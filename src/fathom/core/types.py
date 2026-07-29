"""Core intermediate representation.

Everything downstream of the adapters speaks these types: datasets are identified
uniformly regardless of which engine or storage layer produced them, and partition
state is expressed as predicates rather than concrete key sets so an unbounded
dimension stays representable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from .grains import Grain

__all__ = [
    "ANY",
    "Anything",
    "Capabilities",
    "ChangeSource",
    "ColumnRef",
    "DatasetId",
    "ErasureMode",
    "KeyPredicate",
    "LineageSource",
    "PartitionField",
    "PartitionSpec",
    "Pushdown",
    "UNPARTITIONED",
    "covered_by",
    "subsumes",
]


class Anything:
    """Sentinel for a partition field with no constraint.

    Distinct from `None`, which is a legitimate partition value (a null partition).
    """

    _instance: Anything | None = None

    def __new__(cls) -> Anything:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ANY"

    def __reduce__(self) -> tuple[type[Anything], tuple[()]]:
        return (Anything, ())


ANY = Anything()


@dataclass(frozen=True, order=True)
class DatasetId:
    """An OpenLineage-compatible dataset identity.

    `namespace` locates the system (``s3://bucket``, ``snowflake://acct``), `name`
    locates the dataset within it (``path/to/prefix``, ``db.schema.table``). Two
    references normalize to the same DatasetId exactly when they are the same data,
    which is what makes a Spark job reading ``s3a://`` joinable to a Trino query
    reading the same bytes through a Hive table.
    """

    namespace: str
    name: str

    def __str__(self) -> str:
        # Local paths are already absolute, so the usual separator would double up.
        if self.namespace == "file":
            return f"file://{self.name}"
        return f"{self.namespace}/{self.name}"


@dataclass(frozen=True, order=True)
class ColumnRef:
    dataset: DatasetId
    column: str

    def __str__(self) -> str:
        return f"{self.dataset}#{self.column}"


@dataclass(frozen=True)
class PartitionField:
    """One dimension of a partition spec.

    Time fields carry a grain and support windowed mappings. Value fields (a bucket
    number, a region code) support only passthrough or unconstrained.
    """

    name: str
    kind: Literal["time", "value"] = "value"
    grain: Grain | None = None

    def __post_init__(self) -> None:
        if self.kind == "time" and self.grain is None:
            raise ValueError(f"time partition field {self.name!r} requires a grain")
        if self.kind == "value" and self.grain is not None:
            raise ValueError(f"value partition field {self.name!r} must not carry a grain")

    @classmethod
    def time(cls, name: str, grain: Grain | str) -> PartitionField:
        """A time-partitioned field at the given grain."""
        g = Grain.parse(grain) if isinstance(grain, str) else grain
        return cls(name=name, kind="time", grain=g)

    @classmethod
    def value(cls, name: str) -> PartitionField:
        """A value-partitioned field: a region, a bucket, a tenant."""
        return cls(name=name, kind="value")


@dataclass(frozen=True)
class PartitionSpec:
    """How a dataset is divided. An empty spec means the dataset is a single unit."""

    fields: tuple[PartitionField, ...] = ()

    def __iter__(self) -> Any:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        """Field names in declaration order."""
        return tuple(f.name for f in self.fields)

    def field(self, name: str) -> PartitionField | None:
        """One field by name, or None when this spec has no such field."""
        return next((f for f in self.fields if f.name == name), None)

    @classmethod
    def of(cls, *fields: PartitionField) -> PartitionSpec:
        """Build a spec, rejecting duplicate field names."""
        seen: set[str] = set()
        for f in fields:
            if f.name in seen:
                raise ValueError(f"duplicate partition field {f.name!r}")
            seen.add(f.name)
        return cls(fields=tuple(fields))


UNPARTITIONED = PartitionSpec()


@dataclass(frozen=True)
class KeyPredicate:
    """A constraint over one dataset's partition space.

    Each field is bound either to a concrete value or to ``ANY``. A predicate with
    every field ``ANY`` denotes the whole dataset, which is how the planner represents
    "we could not prove anything narrower, rebuild it all".
    """

    bindings: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings, key=lambda kv: kv[0])))

    @classmethod
    def of(cls, **bindings: Any) -> KeyPredicate:
        """Build a spec, rejecting duplicate field names."""
        return cls(bindings=tuple(bindings.items()))

    @classmethod
    def unbounded(cls, spec: PartitionSpec) -> KeyPredicate:
        """A predicate binding every field of `spec` to ANY — the whole dataset."""
        return cls(bindings=tuple((f.name, ANY) for f in spec.fields))

    def get(self, name: str) -> Any:
        """The value bound to `name`, or ANY when this predicate does not constrain it."""
        for k, v in self.bindings:
            if k == name:
                return v
        return ANY

    @property
    def is_unbounded(self) -> bool:
        """True when nothing is constrained, meaning the whole dataset."""
        return all(v is ANY for _, v in self.bindings)

    def __str__(self) -> str:
        if not self.bindings:
            return "<whole dataset>"
        parts = []
        for k, v in self.bindings:
            if isinstance(v, datetime):
                parts.append(f"{k}={v.isoformat()}")
            else:
                parts.append(f"{k}={v}")
        return "/".join(parts)


def subsumes(outer: KeyPredicate, inner: KeyPredicate) -> bool:
    """True when `outer` covers everything `inner` does.

    An `ANY` binding in `outer` covers any value; a concrete binding covers only
    itself. This is the ordering the whole planner is defined against — the dirty
    set is a set of predicates, and one predicate absorbing another is how the
    worklist detects that it has stopped growing.
    """
    names = {k for k, _ in outer.bindings} | {k for k, _ in inner.bindings}
    for name in names:
        outer_value, inner_value = outer.get(name), inner.get(name)
        if outer_value is ANY:
            continue
        if inner_value is ANY or outer_value != inner_value:
            return False
    return True


def covered_by(candidates: Iterable[KeyPredicate], key: KeyPredicate) -> bool:
    """True when any candidate predicate subsumes `key`."""
    return any(subsumes(c, key) for c in candidates)


class LineageSource(StrEnum):
    """How an adapter learns what depends on what, best first."""

    NATIVE = "native"  # a lineage table the platform maintains for us
    LISTENER = "listener"  # execution-plan hook (Spark, Trino, Flink)
    QUERY_LOG = "query_log"  # parse historical SQL
    DECLARED = "declared"  # the user told us


class ChangeSource(StrEnum):
    """How an adapter learns what changed, cheapest per detected change first."""

    SNAPSHOT_DIFF = "snapshot_diff"  # Iceberg/Delta/Hudi commit metadata
    EVENTS = "events"  # S3 EventBridge, GCS Pub/Sub, ADLS Event Grid
    INVENTORY = "inventory"  # S3 Inventory / Storage Insights manifest diff
    PARTITION_MTIME = "partition_mtime"  # catalog-reported modification times
    LIST_DIFF = "list_diff"  # LIST + etag compare; fine small, ruinous large
    PROFILE_DELTA = "profile_delta"  # last resort
    WATERMARK = "watermark"  # a monotonic column we poll


class Pushdown(StrEnum):
    """What the source can compute for us instead of us reading the data."""

    SKETCHES = "sketches"  # HLL / t-digest state we can merge
    QUANTILES = "quantiles"
    APPROX_DISTINCT = "approx_distinct"
    NONE = "none"


class ErasureMode(StrEnum):
    """How subject data can actually be destroyed here."""

    DELETE_VECTOR = "delete_vector"  # Iceberg/Delta positional deletes + compaction
    CRYPTO_SHRED = "crypto_shred"  # drop the key; the only option under versioning
    REWRITE = "rewrite"  # rewrite affected objects in place
    NONE = "none"  # WORM / Object Lock: refuse, and say so


@dataclass(frozen=True)
class Capabilities:
    """What a given adapter can actually do.

    The planner degrades rather than fails: an adapter that reports `LIST_DIFF` and
    `Pushdown.NONE` still works, it is just slower and coarser.
    """

    lineage: LineageSource
    change: ChangeSource
    pushdown: Pushdown = Pushdown.NONE
    erasure: ErasureMode = ErasureMode.NONE
    column_lineage: bool = False
    partition_aware: bool = False
    # How stale this source's metadata can be. Snowflake's ACCOUNT_USAGE views lag by
    # up to three hours; Databricks system tables by about two. Advancing a resume
    # token past the lag permanently skips rows that had not landed yet, so adapters
    # state it here rather than leaving it to be discovered in production.
    freshness_lag: timedelta | None = None
