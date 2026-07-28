"""Delta Lake catalog adapter, reading the transaction log directly.

No dependency on `deltalake`. The log is newline-delimited JSON under `_delta_log/`,
and everything change detection needs is in the `add` and `remove` actions: a path,
the partition values, and whether the commit changed data. Checkpoints are Parquet,
which pyarrow already reads.

This is the best change-detection strategy available anywhere — `SNAPSHOT_DIFF`.
Cost is proportional to commits since the last run, not to the size of the table, so
a petabyte table with three commits costs three small file reads.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..grains import Grain
from ..types import (
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

__all__ = ["DeltaCatalog"]

_LOG_DIR = "_delta_log"
_COMMIT = re.compile(r"^(\d{20})\.json$")
_CHECKPOINT = re.compile(r"^(\d{20})\.checkpoint(\.\d+\.\d+)?\.parquet$")

# Delta partition values are always strings on the wire. These are the spellings we
# accept when a column's declared type says it is temporal.
_DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y")

# Delta type name -> the grain that type naturally partitions at.
_TYPE_GRAIN = {"date": Grain.DAY, "timestamp": Grain.HOUR, "timestamp_ntz": Grain.HOUR}


@register("delta")
@dataclass
class DeltaCatalog:
    """Datasets are Delta table roots; partitions come from `partitionValues`."""

    name: str = "delta"
    overrides: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.DECLARED,
        change=ChangeSource.SNAPSHOT_DIFF,
        pushdown=Pushdown.NONE,
        erasure=ErasureMode.DELETE_VECTOR,
        partition_aware=True,
    )

    def declare(self, dataset: DatasetId, spec: PartitionSpec) -> None:
        """Override inferred partitioning. A declaration always wins over inference."""
        self.overrides[dataset] = spec

    # -- log access ------------------------------------------------------------

    def _root(self, dataset: DatasetId) -> Path:
        if dataset.namespace != "file":
            raise ValueError(
                f"{self.name} adapter reads local paths; {dataset.namespace} needs a "
                "storage adapter to fetch the log first"
            )
        return Path(dataset.name)

    def _log_dir(self, dataset: DatasetId) -> Path:
        return self._root(dataset) / _LOG_DIR

    def is_delta_table(self, dataset: DatasetId) -> bool:
        try:
            return self._log_dir(dataset).is_dir()
        except ValueError:
            return False

    def _commits(self, dataset: DatasetId) -> list[tuple[int, Path]]:
        log = self._log_dir(dataset)
        if not log.is_dir():
            return []
        out = []
        for path in log.iterdir():
            m = _COMMIT.match(path.name)
            if m:
                out.append((int(m.group(1)), path))
        return sorted(out)

    def _checkpoints(self, dataset: DatasetId) -> list[tuple[int, Path]]:
        log = self._log_dir(dataset)
        if not log.is_dir():
            return []
        out = []
        for path in log.iterdir():
            m = _CHECKPOINT.match(path.name)
            if m:
                out.append((int(m.group(1)), path))
        return sorted(out)

    def _actions(self, commit: Path) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for line in commit.read_text().splitlines():
            line = line.strip()
            if line:
                actions.append(json.loads(line))
        return actions

    def latest_version(self, dataset: DatasetId) -> int | None:
        commits = self._commits(dataset)
        return commits[-1][0] if commits else None

    # -- partition spec --------------------------------------------------------

    def _metadata(self, dataset: DatasetId) -> dict[str, Any] | None:
        """The most recent `metaData` action. Schema changes rewrite it."""
        for _version, path in reversed(self._commits(dataset)):
            for action in reversed(self._actions(path)):
                if "metaData" in action:
                    return dict(action["metaData"])
        return None

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        """Infer partitioning from the log, unless it was declared.

        Delta records partition column *names* but not how we should bucket time.
        A `date` column is unambiguously daily; a `timestamp` is hourly. A string
        column is a value field even if it happens to hold dates, because guessing
        wrong about grain silently changes what a rebuild covers.
        """
        if dataset in self.overrides:
            return self.overrides[dataset]

        metadata = self._metadata(dataset)
        if not metadata:
            return UNPARTITIONED

        columns = metadata.get("partitionColumns") or []
        if not columns:
            return UNPARTITIONED

        types: dict[str, str] = {}
        raw_schema = metadata.get("schemaString")
        if raw_schema:
            try:
                for f in json.loads(raw_schema).get("fields", []):
                    if isinstance(f.get("type"), str):
                        types[f["name"]] = f["type"]
            except (json.JSONDecodeError, AttributeError):
                pass

        fields = []
        for name in columns:
            grain = _TYPE_GRAIN.get(types.get(name, ""))
            fields.append(PartitionField.time(name, grain) if grain else PartitionField.value(name))
        return PartitionSpec.of(*fields)

    def _key(self, spec: PartitionSpec, values: dict[str, str | None]) -> KeyPredicate:
        from ..types import ANY

        bindings: list[tuple[str, object]] = []
        for f in spec.fields:
            raw = values.get(f.name)
            if raw is None:
                # Delta writes null partition values as JSON null; that is a real
                # partition, not an unknown one, but only if the column is present.
                bindings.append((f.name, None if f.name in values else ANY))
                continue
            if f.kind == "value":
                bindings.append((f.name, raw))
                continue
            assert f.grain is not None
            parsed: datetime | None = None
            for fmt in _DATE_FORMATS:
                try:
                    parsed = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                bindings.append((f.name, ANY))
            else:
                from ..grains import truncate

                bindings.append((f.name, truncate(parsed, f.grain)))
        return KeyPredicate(bindings=tuple(bindings))

    # -- change detection ------------------------------------------------------

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Partitions touched by commits after `since`.

        The token is the last fully processed commit version. Both `add` and `remove`
        count: deleting rows from a partition changes downstream aggregates exactly
        as much as inserting them does.
        """
        spec = self.describe_partitioning(dataset)
        commits = self._commits(dataset)
        if not commits:
            return ChangeSet(token=since or "", complete=True)

        start = -1
        if since:
            try:
                start = int(since)
            except ValueError:
                start = -1

        partitions: set[KeyPredicate] = set()
        objects: list[ObjectMeta] = []
        complete = True
        latest = commits[-1][0]

        if start < 0:
            # No baseline. A checkpoint gives the full file list far more cheaply
            # than replaying every commit from version zero.
            checkpoints = self._checkpoints(dataset)
            if checkpoints:
                version, path = checkpoints[-1]
                found = self._read_checkpoint(path, spec)
                partitions |= found[0]
                objects.extend(found[1])
                complete = found[2]
                start = version

        for version, path in commits:
            if version <= start:
                continue
            for action in self._actions(path):
                for verb in ("add", "remove"):
                    entry = action.get(verb)
                    if not entry:
                        continue
                    if not entry.get("dataChange", True):
                        continue  # compaction and other metadata-only rewrites
                    partitions.add(self._key(spec, entry.get("partitionValues") or {}))
                    if verb == "add":
                        objects.append(
                            ObjectMeta(
                                path=str(self._root(dataset) / entry["path"]),
                                size=int(entry.get("size") or 0),
                                modified=(
                                    datetime.fromtimestamp(entry["modificationTime"] / 1000)
                                    if entry.get("modificationTime")
                                    else None
                                ),
                            )
                        )

        return ChangeSet(
            partitions=frozenset(partitions),
            token=str(latest),
            complete=complete,
            objects=tuple(objects),
        )

    def _read_checkpoint(
        self, path: Path, spec: PartitionSpec
    ) -> tuple[set[KeyPredicate], list[ObjectMeta], bool]:
        """Partitions present as of a checkpoint. Best effort; failure means widen."""
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path, columns=["add"])
        except Exception:  # noqa: BLE001 - checkpoint layouts vary across writers
            return set(), [], False

        partitions: set[KeyPredicate] = set()
        objects: list[ObjectMeta] = []
        for entry in table.column("add").to_pylist():
            if not entry:
                continue
            partitions.add(self._key(spec, entry.get("partitionValues") or {}))
            if entry.get("path"):
                objects.append(
                    ObjectMeta(
                        path=str(path.parent.parent / entry["path"]),
                        size=int(entry.get("size") or 0),
                    )
                )
        return partitions, objects, True

    def files_for(self, dataset: DatasetId, partition: KeyPredicate | None = None) -> list[str]:
        """Live data files, optionally restricted to one partition.

        Replays adds and removes rather than trusting any single commit, because a
        file added in version 3 and removed in version 7 is not live.
        """
        spec = self.describe_partitioning(dataset)
        root = self._root(dataset)
        live: dict[str, KeyPredicate] = {}
        for _version, path in self._commits(dataset):
            for action in self._actions(path):
                if add := action.get("add"):
                    live[add["path"]] = self._key(spec, add.get("partitionValues") or {})
                if remove := action.get("remove"):
                    live.pop(remove["path"], None)
        return [
            str(root / rel)
            for rel, key in sorted(live.items())
            if partition is None or partition.is_unbounded or key == partition
        ]
