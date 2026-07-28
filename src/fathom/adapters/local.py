"""Local filesystem storage adapter.

The reference implementation of `StorageAdapter`, and the one the conformance suite
runs against. It uses the weakest change-detection strategy on purpose — LIST plus
mtime compare — so that anything built on top works for adapters that can offer
nothing better. Adapters with snapshot diffs or event streams are strictly faster,
never differently shaped.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

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

__all__ = ["LocalStorage"]

DATA_SUFFIXES = {".parquet", ".pq"}


@register("local")
@dataclass
class LocalStorage:
    """Datasets are directories; partitions are Hive segments or a declared template."""

    name: str = "local"
    specs: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    templates: dict[DatasetId, PathTemplate] = field(default_factory=dict)
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

    def _root(self, dataset: DatasetId) -> Path:
        if dataset.namespace != "file":
            raise ValueError(f"{self.name} adapter cannot address {dataset.namespace}")
        return Path(dataset.name)

    def list_objects(self, dataset: DatasetId) -> Iterable[ObjectMeta]:
        root = self._root(dataset)
        if not root.exists():
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in DATA_SUFFIXES:
                continue
            stat = path.stat()
            yield ObjectMeta(
                path=str(path),
                size=stat.st_size,
                etag=f"{stat.st_size}-{int(stat.st_mtime_ns)}",
                modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            )

    def key_for(self, dataset: DatasetId, path: str) -> KeyPredicate:
        root = self._root(dataset)
        try:
            relative = str(Path(path).relative_to(root))
        except ValueError:
            relative = path
        return key_from_path(
            relative,
            self.describe_partitioning(dataset),
            template=self.templates.get(dataset),
        )

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Objects modified after `since`, and the partitions they belong to.

        The token is the high-water modification time. Objects written during the
        same second as the token are re-reported rather than skipped: duplicating a
        rebuild is cheap, missing one is not.
        """
        cutoff: datetime | None = None
        if since:
            try:
                cutoff = datetime.fromisoformat(since)
            except ValueError:
                cutoff = None

        touched: list[ObjectMeta] = []
        high_water = cutoff
        for obj in self.list_objects(dataset):
            if obj.modified is None:
                continue
            if high_water is None or obj.modified > high_water:
                high_water = obj.modified
            if cutoff is None or obj.modified >= cutoff:
                touched.append(obj)

        partitions = frozenset(self.key_for(dataset, o.path) for o in touched)
        return ChangeSet(
            partitions=partitions,
            token=high_water.isoformat() if high_water else (since or ""),
            complete=True,
            objects=tuple(touched),
        )

    def local_paths(self, objects: Sequence[ObjectMeta]) -> list[str | Path]:
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
            self.local_paths(objects),
            dataset=dataset,
            partition=partition,
        )
