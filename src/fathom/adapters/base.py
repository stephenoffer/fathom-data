"""Adapter protocols.

Three surfaces, because "what depends on what" and "what changed" come from
different places and neither assumes SQL:

- **Engine** adapters read execution plans or query logs. Spark, Trino, Flink,
  ClickHouse, DataFusion, DuckDB.
- **Catalog** adapters read table and partition metadata. Iceberg, Delta, Glue,
  Unity, BigQuery INFORMATION_SCHEMA.
- **Storage** adapters read objects, events, and inventory manifests. S3, GCS,
  ADLS, R2, MinIO, local.

Adapters declare capabilities rather than implement everything. One reporting
`LIST_DIFF` and `Pushdown.NONE` still works — it is slower and coarser, and the
planner degrades instead of failing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..types import (
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    DatasetId,
    KeyPredicate,
    LineageSource,
    PartitionSpec,
)

__all__ = [
    "CatalogAdapter",
    "ChangeSet",
    "EngineAdapter",
    "LineageEvent",
    "ObjectMeta",
    "QueryEvent",
    "StorageAdapter",
    "get_adapter",
    "register",
    "registered",
]

# An opaque, adapter-defined resume point: an Iceberg snapshot id, an S3 inventory
# manifest key, a query-history timestamp. Callers store it and hand it back.
Token = str


@dataclass(frozen=True)
class ObjectMeta:
    path: str
    size: int = 0
    etag: str | None = None
    modified: datetime | None = None


@dataclass(frozen=True)
class ChangeSet:
    """What changed since a token, and whether the answer is trustworthy.

    `complete=False` means the source could not enumerate everything — an expired
    event subscription, a truncated query history. The planner treats an incomplete
    changeset as grounds to widen, not as an empty result.
    """

    partitions: frozenset[KeyPredicate] = frozenset()
    token: Token = ""
    complete: bool = True
    objects: tuple[ObjectMeta, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.partitions and not self.objects


@dataclass(frozen=True)
class QueryEvent:
    """One statement recovered from a query log or listener."""

    sql: str
    dialect: str = ""
    query_id: str = ""
    started: datetime | None = None
    default_database: str | None = None
    default_schema: str | None = None


@dataclass(frozen=True)
class LineageEvent:
    """A dependency the platform reported directly, without us parsing SQL."""

    src: DatasetId
    dst: DatasetId
    columns: tuple[tuple[str, str], ...] = ()
    evidence: str = "native"
    observed: datetime | None = None


@runtime_checkable
class StorageAdapter(Protocol):
    """Object storage: what exists, what changed, and how to read it cheaply."""

    name: str
    capabilities: Capabilities

    def list_objects(self, dataset: DatasetId) -> Iterable[ObjectMeta]: ...

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet: ...

    def local_paths(self, objects: Sequence[ObjectMeta]) -> list[str | Path]:
        """Paths pyarrow can open. Remote adapters may return fsspec-style URIs."""
        ...


@runtime_checkable
class CatalogAdapter(Protocol):
    """Table metadata: partition specs and commit-level change detection."""

    name: str
    capabilities: Capabilities

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec: ...

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet: ...


@runtime_checkable
class EngineAdapter(Protocol):
    """Compute engines: where lineage comes from, and where rebuilds are issued."""

    name: str
    capabilities: Capabilities
    dialect: str

    def fetch_lineage(self, since: Token | None) -> Iterable[LineageEvent]:
        """Native lineage, for platforms that maintain it (Unity Catalog, Snowflake)."""
        ...

    def fetch_queries(self, since: Token | None) -> Iterable[QueryEvent]:
        """Historical statements, for everything else."""
        ...

    def render_rebuild(self, dataset: DatasetId, partitions: Iterable[KeyPredicate]) -> list[str]:
        """Statements that rebuild exactly these partitions. Never executed here."""
        ...


_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    """Decorator registering an adapter class under a stable name."""

    def wrap(cls: type) -> type:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"adapter {name!r} already registered by {_REGISTRY[name]!r}")
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_adapter(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"no adapter named {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class DeclaredCatalog:
    """A catalog you configure by hand.

    Every adapter matrix has a long tail. Rather than block on writing an adapter
    for some in-house system, declare its specs and let the rest of the pipeline
    work today.
    """

    name: str = "declared"
    specs: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.DECLARED,
        change=ChangeSource.WATERMARK,
        partition_aware=True,
    )

    def declare(self, dataset: DatasetId, spec: PartitionSpec) -> None:
        self.specs[dataset] = spec

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        return self.specs.get(dataset, UNPARTITIONED)

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        return ChangeSet(token=since or "")
