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
from typing import Protocol, runtime_checkable

from ..core.types import (
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    DatasetId,
    KeyPredicate,
    LineageSource,
    PartitionSpec,
)
from ..core.util.text import did_you_mean

__all__ = [
    "CatalogAdapter",
    "ChangeSet",
    "DeclaredCatalog",
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
    """One object in storage: where it is, how big, and when it last changed."""

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
        """True when nothing changed."""
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

    def list_objects(self, dataset: DatasetId) -> Iterable[ObjectMeta]:
        """Every object under the dataset's prefix, with size and modification time.

        A full listing, and on a large prefix an expensive one — `changed` exists so
        the common case does not have to pay for it.
        """
        ...

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """What changed since `since`, and a token to resume from next time.

        `since` of None means "everything", which is the first run. Returning a
        `ChangeSet` with `complete=False` says the source could not enumerate
        exhaustively, and callers must treat that as "assume everything changed"
        rather than as the empty set.
        """
        ...

    def paths(self, objects: Sequence[ObjectMeta]) -> list[str]:
        """URIs a filesystem can open, in the same order as the objects given."""
        ...


@runtime_checkable
class CatalogAdapter(Protocol):
    """Table metadata: partition specs and commit-level change detection."""

    name: str
    capabilities: Capabilities

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        """How this table is partitioned, as the planner needs to see it.

        Return `UNPARTITIONED` when the catalog does not say, never a guess: a
        fabricated grain makes every mapping composed across this dataset wrong,
        and wrong is the one thing the planner cannot recover from.
        """
        ...

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Partitions touched since `since`, from commit metadata where available.

        Cheaper and more precise than listing objects, because the table format
        already recorded which files each commit wrote.
        """
        ...


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
    """The adapter class registered under `name`.

    Raises `KeyError` naming everything that *is* registered, because the usual
    cause is a typo or a missing optional dependency whose import never ran.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"no adapter named {name!r}{did_you_mean(name, _REGISTRY)}; "
            f"registered right now: {sorted(_REGISTRY)}. An adapter behind an "
            f"optional extra stays absent until its package is installed — "
            f"iceberg needs `pip install 'fathom-data[iceberg]'`, cloud storage "
            f"needs `[s3]`, `[gcs]`, or `[azure]`"
        )
    return _REGISTRY[name]


def registered() -> list[str]:
    """Names of every adapter currently importable, sorted.

    Only reflects modules that have been imported: an adapter behind an optional
    extra is absent until its package is installed and its module loads.
    """
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
        """Record how a dataset is partitioned. Last declaration wins."""
        self.specs[dataset] = spec

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        """The declared spec, or `UNPARTITIONED` for anything never declared."""
        return self.specs.get(dataset, UNPARTITIONED)

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Always empty: a hand-declared catalog has no way to detect change.

        The token is echoed back unchanged so a caller storing it sees no movement,
        rather than a cursor that appears to advance over data nobody examined.
        """
        return ChangeSet(token=since or "")
