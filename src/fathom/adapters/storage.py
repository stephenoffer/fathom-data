"""Object storage adapter: S3, GCS, ADLS, R2, MinIO, HDFS, and local disk.

One implementation for every protocol, because the difference between a bucket and a
directory is fsspec's problem, not the planner's. `LocalStorage` is an alias kept for
readability in tests and docs.

Change detection here is `LIST_DIFF`, the weakest strategy on the ladder, chosen
deliberately: if everything downstream works on LIST plus etag comparison, then a
Delta or Iceberg adapter with real snapshot diffs is strictly faster rather than
differently shaped. At scale you should not use this — see `inventory.py` for the
S3 Inventory path, which costs a manifest read instead of a full bucket listing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..fs import FileInfo, FileSystem, data_files, filesystem_for
from ..ids import dataset_uri
from ..paths import PathTemplate, key_from_path
from ..profile import Profile, profile_parquet
from ..types import (
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    DatasetId,
    ErasureMode,
    KeyPredicate,
    LineageSource,
    PartitionSpec,
    Pushdown,
)
from .base import ChangeSet, ObjectMeta, Token, register

__all__ = ["LocalStorage", "ObjectStorage"]

DATA_SUFFIXES = (".parquet", ".pq")


def _decode_token(token: Token | None) -> tuple[datetime | None, frozenset[str]]:
    """Read a resume token. Unrecognized tokens mean "no baseline", never a crash."""
    if not token:
        return None, frozenset()
    try:
        blob = json.loads(token)
        return datetime.fromisoformat(blob["t"]), frozenset(blob.get("seen", ()))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    try:  # tokens written before boundary etags were carried
        return datetime.fromisoformat(token), frozenset()
    except ValueError:
        return None, frozenset()


def _encode_token(high_water: datetime | None, boundary: Iterable[str]) -> Token:
    if high_water is None:
        return ""
    return json.dumps(
        {"t": high_water.isoformat(), "seen": sorted(boundary)}, separators=(",", ":")
    )


@register("storage")
@dataclass
class ObjectStorage:
    """Datasets are prefixes; partitions come from Hive segments or a declared template.

    `storage_options` passes straight to fsspec, so credentials, an S3-compatible
    endpoint, or a requester-pays flag are all expressible without a code change.
    """

    name: str = "storage"
    storage_options: dict[str, Any] = field(default_factory=dict)
    specs: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    templates: dict[DatasetId, PathTemplate] = field(default_factory=dict)
    suffixes: tuple[str, ...] = DATA_SUFFIXES
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.DECLARED,
        change=ChangeSource.LIST_DIFF,
        pushdown=Pushdown.NONE,
        erasure=ErasureMode.REWRITE,
        partition_aware=True,
    )

    def declare(
        self,
        dataset: DatasetId,
        spec: PartitionSpec,
        template: PathTemplate | None = None,
    ) -> None:
        self.specs[dataset] = spec
        if template is not None:
            self.templates[dataset] = template

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        return self.specs.get(dataset, UNPARTITIONED)

    # -- filesystem ------------------------------------------------------------

    def uri(self, dataset: DatasetId) -> str:
        return dataset_uri(dataset)

    def filesystem(self, dataset: DatasetId) -> FileSystem:
        return filesystem_for(self.uri(dataset), **self.storage_options)

    def _files(self, dataset: DatasetId) -> list[FileInfo]:
        fs = self.filesystem(dataset)
        return data_files(fs, self.uri(dataset), self.suffixes)

    # -- listing ---------------------------------------------------------------

    def list_objects(self, dataset: DatasetId) -> Iterable[ObjectMeta]:
        fs = self.filesystem(dataset)
        unstrip = getattr(fs, "unstrip", lambda p: p)
        for info in self._files(dataset):
            yield ObjectMeta(
                path=unstrip(info.path),
                size=info.size,
                etag=info.etag or f"{info.size}-{info.modified}",
                modified=info.modified,
            )

    def key_for(self, dataset: DatasetId, path: str) -> KeyPredicate:
        """Derive a partition key from an object's path relative to the dataset root."""
        root = self.uri(dataset).rstrip("/")
        relative = path
        for candidate in (root, root.split("://", 1)[-1]):
            marker = candidate.rstrip("/") + "/"
            index = path.find(marker)
            if index >= 0:
                relative = path[index + len(marker) :]
                break
        return key_from_path(
            relative,
            self.describe_partitioning(dataset),
            template=self.templates.get(dataset),
        )

    # -- change detection ------------------------------------------------------

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Objects modified after `since`, and the partitions they belong to.

        The token is a high-water modification time *plus* the etags of the objects
        sitting exactly on that boundary. Timestamps are coarse and object stores
        happily write several objects in the same second, so a strict `>` would miss
        one while a `>=` would re-report the newest forever. Carrying the boundary
        etags gives both properties: nothing is missed, and a quiet dataset converges
        to reporting nothing.
        """
        cutoff, seen = _decode_token(since)

        touched: list[ObjectMeta] = []
        high_water = cutoff
        boundary: set[str] = set()

        for obj in self.list_objects(dataset):
            if obj.modified is None:
                # No timestamp at all: we cannot reason incrementally, so treat it as
                # changed every time rather than silently skipping it.
                touched.append(obj)
                continue
            if high_water is None or obj.modified > high_water:
                high_water = obj.modified
                boundary = set()
            if obj.modified == high_water and obj.etag:
                boundary.add(obj.etag)
            if cutoff is None or (obj.modified >= cutoff and obj.etag not in seen):
                touched.append(obj)

        return ChangeSet(
            partitions=frozenset(self.key_for(dataset, o.path) for o in touched),
            token=_encode_token(high_water, boundary),
            complete=True,
            objects=tuple(touched),
        )

    # -- reads -----------------------------------------------------------------

    def paths(self, objects: Sequence[ObjectMeta]) -> list[str]:
        return [o.path for o in objects]

    def profile(
        self,
        dataset: DatasetId,
        *,
        partition: KeyPredicate | None = None,
    ) -> Profile:
        """Footer-only profile of one partition, or the whole dataset.

        Reads no data pages, so this is cheap enough to run on every partition the
        planner reports dirty rather than on a nightly whole-table schedule.
        """
        objects = list(self.list_objects(dataset))
        if partition is not None and not partition.is_unbounded:
            objects = [o for o in objects if self.key_for(dataset, o.path) == partition]
        return profile_parquet(
            self.paths(objects),
            dataset=dataset,
            partition=partition,
            fs=self.filesystem(dataset),
        )

    def erase_files(self, dataset: DatasetId, paths: Iterable[str]) -> int:
        """Physically remove objects. Used only by an executed erasure plan."""
        fs = self.filesystem(dataset)
        removed = 0
        for path in paths:
            fs.delete(path)
            removed += 1
        return removed


# Local disk is the same adapter with no storage options; the alias keeps call sites
# readable and preserves the name the conformance suite was written against.
LocalStorage = ObjectStorage
