"""Apache Iceberg catalog adapter.

Requires the `iceberg` extra. Unlike Delta, Iceberg manifests are Avro, so this
leans on pyiceberg rather than reading the format directly.

Opens tables through `StaticTable`, which needs only a metadata file — no catalog
service, no credentials beyond read access to the table's own directory. That keeps
the adapter usable against a bare S3 prefix.

The interesting case is snapshot expiry. If the token names a snapshot that has been
expired away, we genuinely cannot know what changed since, so the changeset comes
back with every partition and `complete=False`. Reporting an empty diff there would
be the one unforgivable bug in a change detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ..fs import FileSystem, filesystem_for, join
from ..grains import Grain, truncate
from ..ids import dataset_uri
from ..types import (
    ANY,
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    DatasetId,
    ErasureMode,
    KeyPredicate,
    LineageSource,
    PartitionField,
    PartitionSpec,
    Pushdown,
)
from .base import ChangeSet, ObjectMeta, Token, register

__all__ = ["IcebergCatalog", "IcebergUnavailable"]

_EPOCH = datetime(1970, 1, 1)

# Iceberg transform name -> the grain it buckets at. Values arrive as integer offsets
# from the epoch, not as timestamps, so each needs its own decoding.
_TRANSFORM_GRAIN = {"year": Grain.YEAR, "month": Grain.MONTH, "day": Grain.DAY, "hour": Grain.HOUR}


class IcebergUnavailable(RuntimeError):
    """Raised when the `iceberg` extra is not installed."""


def _require_pyiceberg() -> Any:
    try:
        from pyiceberg.table import StaticTable
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise IcebergUnavailable(
            "the Iceberg adapter needs the 'iceberg' extra: pip install 'fathom-data[iceberg]'"
        ) from exc
    return StaticTable


def _decode(value: Any, grain: Grain) -> datetime | None:
    """Turn an Iceberg partition value into a datetime for the given grain."""
    if value is None or isinstance(value, str):
        return None
    try:
        offset = int(value)
    except (TypeError, ValueError):
        return None
    if grain is Grain.YEAR:
        return datetime(1970 + offset, 1, 1)
    if grain is Grain.MONTH:
        return datetime(1970 + offset // 12, offset % 12 + 1, 1)
    if grain is Grain.DAY:
        return _EPOCH + timedelta(days=offset)
    return _EPOCH + timedelta(hours=offset)


@dataclass
class IcebergCatalog:
    """Datasets are Iceberg table roots containing a `metadata/` directory."""

    name: str = "iceberg"
    storage_options: dict[str, Any] = field(default_factory=dict)
    overrides: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.DECLARED,
        change=ChangeSource.SNAPSHOT_DIFF,
        pushdown=Pushdown.NONE,
        erasure=ErasureMode.DELETE_VECTOR,
        partition_aware=True,
    )

    def declare(self, dataset: DatasetId, spec: PartitionSpec) -> None:
        self.overrides[dataset] = spec

    # -- table access ----------------------------------------------------------

    def _root(self, dataset: DatasetId) -> str:
        return dataset_uri(dataset).rstrip("/")

    def _fs(self, dataset: DatasetId) -> FileSystem:
        return filesystem_for(self._root(dataset), **self.storage_options)

    def _metadata_file(self, dataset: DatasetId) -> str | None:
        """The current metadata file: version-hint if present, else highest version."""
        fs = self._fs(dataset)
        directory = join(self._root(dataset), "metadata")
        if not fs.is_dir(directory):
            return None

        hint = join(directory, "version-hint.text")
        if fs.exists(hint):
            version = fs.read_text(hint).strip()
            for candidate in (f"v{version}.metadata.json", f"{version}.metadata.json"):
                target = join(directory, candidate)
                if fs.exists(target):
                    return target

        # No hint: the highest-numbered metadata file wins. Sorting by modification
        # time would pick the wrong one when a table is copied between locations.
        found = [i for i in fs.ls(directory, recursive=False) if i.path.endswith(".metadata.json")]
        return max(found, key=lambda i: i.path).path if found else None

    def is_iceberg_table(self, dataset: DatasetId) -> bool:
        try:
            return self._metadata_file(dataset) is not None
        except (ValueError, OSError):
            return False

    def _table(self, dataset: DatasetId) -> Any:
        static_table = _require_pyiceberg()
        path = self._metadata_file(dataset)
        if path is None:
            raise FileNotFoundError(f"no Iceberg metadata found under {self._root(dataset)}")
        return static_table.from_metadata(str(path), properties=dict(self.storage_options))

    # -- partition spec --------------------------------------------------------

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        if dataset in self.overrides:
            return self.overrides[dataset]
        try:
            table = self._table(dataset)
        except (FileNotFoundError, IcebergUnavailable, ValueError):
            return UNPARTITIONED

        fields = []
        for f in table.spec().fields:
            transform = str(f.transform).split("[")[0].lower()
            grain = _TRANSFORM_GRAIN.get(transform)
            fields.append(
                PartitionField.time(f.name, grain) if grain else PartitionField.value(f.name)
            )
        return PartitionSpec.of(*fields) if fields else UNPARTITIONED

    def _key(self, table: Any, spec: PartitionSpec, partition: Any) -> KeyPredicate:
        """Map an Iceberg partition record onto our key predicate."""
        names = [f.name for f in table.spec().fields]
        values: dict[str, Any] = {}
        for index, name in enumerate(names):
            try:
                values[name] = partition[index]
            except (IndexError, TypeError, KeyError):
                values[name] = None

        bindings: list[tuple[str, object]] = []
        for f in spec.fields:
            raw = values.get(f.name)
            if f.kind == "value":
                bindings.append((f.name, ANY if raw is None else raw))
                continue
            assert f.grain is not None
            decoded = _decode(raw, f.grain)
            bindings.append((f.name, ANY if decoded is None else truncate(decoded, f.grain)))
        return KeyPredicate(bindings=tuple(bindings))

    # -- change detection ------------------------------------------------------

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        table = self._table(dataset)
        spec = self.describe_partitioning(dataset)
        snapshots = sorted(table.metadata.snapshots, key=lambda s: (s.timestamp_ms, s.snapshot_id))
        if not snapshots:
            return ChangeSet(token=since or "", complete=True)

        latest = str(snapshots[-1].snapshot_id)
        start = 0
        complete = True

        if since:
            known = [i for i, s in enumerate(snapshots) if str(s.snapshot_id) == since]
            if known:
                start = known[0] + 1
            else:
                # The token's snapshot has expired. We cannot reconstruct the diff, so
                # report everything and say the answer is not exhaustive.
                complete = False

        if start >= len(snapshots) and complete:
            return ChangeSet(token=latest, complete=True)

        partitions: set[KeyPredicate] = set()
        objects: list[ObjectMeta] = []
        for snapshot in snapshots[start:]:
            for manifest in snapshot.manifests(table.io):
                for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=False):
                    owner = getattr(entry, "snapshot_id", None)
                    if owner is not None and owner != snapshot.snapshot_id:
                        continue  # carried forward from an earlier snapshot
                    data_file = entry.data_file
                    partitions.add(self._key(table, spec, data_file.partition))
                    objects.append(
                        ObjectMeta(
                            path=str(data_file.file_path),
                            size=int(data_file.file_size_in_bytes or 0),
                            modified=datetime.fromtimestamp(snapshot.timestamp_ms / 1000, tz=UTC),
                        )
                    )

        if not complete and not partitions:
            # Expired token and nothing enumerable: widen explicitly rather than
            # returning an empty set that reads as "nothing changed".
            partitions.add(KeyPredicate.unbounded(spec))

        return ChangeSet(
            partitions=frozenset(partitions),
            token=latest,
            complete=complete,
            objects=tuple(objects),
        )

    def files_for(self, dataset: DatasetId, partition: KeyPredicate | None = None) -> list[str]:
        """Live data files, optionally restricted to one partition."""
        table = self._table(dataset)
        spec = self.describe_partitioning(dataset)
        current = table.metadata.current_snapshot_id
        snapshot = next(
            (s for s in table.metadata.snapshots if s.snapshot_id == current),
            None,
        )
        if snapshot is None:
            return []

        out: list[str] = []
        for manifest in snapshot.manifests(table.io):
            for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
                key = self._key(table, spec, entry.data_file.partition)
                if partition is None or partition.is_unbounded or key == partition:
                    out.append(str(entry.data_file.file_path))
        return sorted(out)


register("iceberg")(IcebergCatalog)
