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
from .grains import truncate as truncate_to
from .util.text import did_you_mean

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

    Example:
        >>> raw = DatasetId("duckdb", "raw.events")
        >>> str(raw)
        'duckdb/raw.events'
        >>> DatasetId.parse("s3://lake/raw/events")
        DatasetId('s3://lake', 'raw/events')

    See also:
        `fathom.normalize`, which resolves the spelling variations (``s3a://``,
        DBFS mounts, identifier case folding) that `parse` deliberately leaves
        alone. Use `normalize` on anything a user or another system typed; use
        `parse` to read back a string this library printed.
    """

    namespace: str
    name: str

    def __str__(self) -> str:
        # Local paths are already absolute, so the usual separator would double up.
        if self.namespace == "file":
            return f"file://{self.name}"
        return f"{self.namespace}/{self.name}"

    def __repr__(self) -> str:
        # The default dataclass repr spells the field names, which triples the width
        # of any collection of these — and error messages print collections of these.
        return f"DatasetId({self.namespace!r}, {self.name!r})"

    @classmethod
    def parse(cls, text: str) -> DatasetId:
        """Read back the form `__str__` prints.

        Splits on the boundary between the system and the dataset within it: after
        the ``://`` authority for a URI, or at the first ``/`` otherwise.

        Args:
            text: ``s3://lake/raw/events``, ``duckdb/raw.events``, or
                ``file:///tmp/lake/events``.

        Returns:
            The identity that string denotes.

        Raises:
            ValueError: The string carries no namespace, so there is no way to
                tell which system the dataset lives in.

        Example:
            >>> DatasetId.parse("duckdb/raw.events")
            DatasetId('duckdb', 'raw.events')
            >>> DatasetId.parse("file:///tmp/lake/events")
            DatasetId('file', '/tmp/lake/events')
        """
        raw = text.strip()
        if not raw:
            raise ValueError(
                "a dataset identity cannot be empty; expected `system/name`, "
                "for example 'duckdb/raw.events' or 's3://lake/raw/events'"
            )
        if raw.startswith("file://"):
            return cls(namespace="file", name=raw[len("file://") :] or "/")
        scheme, sep, rest = raw.partition("://")
        if sep:
            authority, _, name = rest.partition("/")
            return cls(namespace=f"{scheme}://{authority}", name=name)
        namespace, sep, name = raw.partition("/")
        if not sep:
            raise ValueError(
                f"{text!r} has no namespace, so there is no way to tell which system "
                "it lives in. Write `system/name` — 'duckdb/raw.events', "
                "'snowflake://xy12345/db.schema.orders' — or call "
                "`fathom.normalize(name, system=...)` to build one from a bare table name"
            )
        return cls(namespace=namespace, name=name)


@dataclass(frozen=True, order=True)
class ColumnRef:
    """One column of one dataset — the unit column-level lineage is expressed in.

    Example:
        >>> str(ColumnRef(DatasetId("duckdb", "raw.events"), "amount"))
        'duckdb/raw.events#amount'
    """

    dataset: DatasetId
    column: str

    def __str__(self) -> str:
        return f"{self.dataset}#{self.column}"

    def __repr__(self) -> str:
        return f"ColumnRef({self.dataset!r}, {self.column!r})"

    @classmethod
    def parse(cls, text: str) -> ColumnRef:
        """Read back the ``dataset#column`` form `__str__` prints.

        Example:
            >>> ColumnRef.parse("duckdb/raw.events#amount")
            ColumnRef(DatasetId('duckdb', 'raw.events'), 'amount')
        """
        dataset, sep, column = text.rpartition("#")
        if not sep or not column:
            raise ValueError(
                f"{text!r} is not a column reference; write `dataset#column`, "
                "for example 'duckdb/raw.events#amount'"
            )
        return cls(dataset=DatasetId.parse(dataset), column=column)


@dataclass(frozen=True)
class PartitionField:
    """One dimension of a partition spec.

    Time fields carry a grain and support windowed mappings. Value fields (a bucket
    number, a region code) support only passthrough or unconstrained.

    Build these with the two constructors rather than the raw dataclass — they are
    named for the distinction that matters and cannot produce an invalid pair:

    Example:
        >>> PartitionField.time("dt", "day")
        PartitionField.time('dt', day)
        >>> PartitionField.value("region")
        PartitionField.value('region')
    """

    name: str
    kind: Literal["time", "value"] = "value"
    grain: Grain | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("time", "value"):
            raise ValueError(
                f"partition field {self.name!r} has kind {self.kind!r}; expected "
                "'time' (a date or timestamp, which supports windowed mappings) or "
                "'value' (a region, tenant, or bucket, which does not)"
            )
        if self.kind == "time" and self.grain is None:
            raise ValueError(
                f"time partition field {self.name!r} requires a grain, because a plan "
                f"cannot tell a daily partition from a monthly one without it. "
                f"Write `PartitionField.time({self.name!r}, 'day')`, or declare it in "
                f"fathom.yml as `{{field: {self.name}, grain: day}}`"
            )
        if self.kind == "value" and self.grain is not None:
            raise ValueError(
                f"value partition field {self.name!r} must not carry a grain — only "
                f"time fields have one. Write `PartitionField.value({self.name!r})`, or "
                f"`PartitionField.time({self.name!r}, {self.grain.label!r})` if it is a date"
            )

    def __str__(self) -> str:
        """The compact form `PartitionSpec.parse` reads: ``dt:day`` or ``region``."""
        return f"{self.name}:{self.grain.label}" if self.grain is not None else self.name

    def __repr__(self) -> str:
        if self.kind == "time":
            return f"PartitionField.time({self.name!r}, {self.grain})"
        return f"PartitionField.value({self.name!r})"

    @classmethod
    def time(cls, name: str, grain: Grain | str) -> PartitionField:
        """A time-partitioned field at the given grain.

        Args:
            name: The column the partition is keyed on, e.g. ``dt``.
            grain: How wide one bucket is — a `Grain`, or any name `Grain.parse`
                accepts (``day``, ``daily``, ``d``).

        Example:
            >>> PartitionField.time("dt", Grain.MONTH).grain
            <Grain.MONTH: 3>
        """
        return cls(name=name, kind="time", grain=Grain.parse(grain))

    @classmethod
    def value(cls, name: str) -> PartitionField:
        """A value-partitioned field: a region, a bucket, a tenant.

        Example:
            >>> PartitionField.value("region").kind
            'value'
        """
        return cls(name=name, kind="value")

    @classmethod
    def parse(cls, text: str) -> PartitionField:
        """One field from its compact form: ``dt:day`` is time, ``region`` is value.

        Example:
            >>> PartitionField.parse("dt:day")
            PartitionField.time('dt', day)
            >>> PartitionField.parse("region")
            PartitionField.value('region')
        """
        name, sep, grain = (part.strip() for part in text.strip().partition(":"))
        if not name:
            raise ValueError(
                f"{text!r} is not a partition field; write `name` for a value field "
                "or `name:grain` for a time field, e.g. 'region' or 'dt:day'"
            )
        return cls.time(name, grain) if sep else cls.value(name)


@dataclass(frozen=True)
class PartitionSpec:
    """How a dataset is divided. An empty spec means the dataset is a single unit.

    The spec is the single most important thing you declare about a dataset: a plan
    can only be as precise as the partitioning it knows about, and a dataset with no
    spec is invalidated whole. `fathom doctor` reports the ones still missing.

    Example:
        >>> spec = PartitionSpec.parse("dt:day, region")
        >>> spec.names
        ('dt', 'region')
        >>> str(spec)
        'dt:day, region'
        >>> len(spec), "region" in spec
        (2, True)
    """

    fields: tuple[PartitionField, ...] = ()

    def __iter__(self) -> Any:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __contains__(self, name: object) -> bool:
        """True when this spec has a field of that name."""
        return any(f.name == name for f in self.fields)

    def __str__(self) -> str:
        """The compact form `parse` reads back, or a readable marker when empty."""
        return ", ".join(str(f) for f in self.fields) if self.fields else "<unpartitioned>"

    def __repr__(self) -> str:
        return f"PartitionSpec.parse({str(self)!r})" if self.fields else "UNPARTITIONED"

    @property
    def names(self) -> tuple[str, ...]:
        """Field names in declaration order."""
        return tuple(f.name for f in self.fields)

    @property
    def time_fields(self) -> tuple[PartitionField, ...]:
        """Only the time-partitioned fields — the ones a `TimeWindow` can map."""
        return tuple(f for f in self.fields if f.kind == "time")

    def field(self, name: str) -> PartitionField | None:
        """One field by name, or None when this spec has no such field."""
        return next((f for f in self.fields if f.name == name), None)

    def require(self, name: str) -> PartitionField:
        """One field by name, raising a message that names the alternatives.

        Use this over `field` wherever the absence is a user error rather than a
        branch — the raised message lists what the spec does have, which is the
        question the caller asks next.

        Raises:
            KeyError: No such field.

        Example:
            >>> PartitionSpec.parse("dt:day").require("dt").kind
            'time'
        """
        found = self.field(name)
        if found is None:
            if not self.fields:
                raise KeyError(
                    f"this dataset is unpartitioned, so it has no field {name!r}. "
                    "Declare a partition spec for it in fathom.yml to plan at "
                    "partition granularity rather than whole-dataset"
                )
            raise KeyError(
                f"no partition field {name!r}; this dataset is partitioned by "
                f"{', '.join(repr(n) for n in self.names)}{did_you_mean(name, self.names)}"
            )
        return found

    @classmethod
    def of(cls, *fields: PartitionField) -> PartitionSpec:
        """Build a spec, rejecting duplicate field names.

        Example:
            >>> PartitionSpec.of(PartitionField.time("dt", "day")).names
            ('dt',)
        """
        seen: set[str] = set()
        for f in fields:
            if f.name in seen:
                raise ValueError(
                    f"duplicate partition field {f.name!r}; each field may appear once, "
                    "and the order you declare them in is the order they are printed"
                )
            seen.add(f.name)
        return cls(fields=tuple(fields))

    @classmethod
    def parse(cls, text: str) -> PartitionSpec:
        """Build a spec from the compact form: ``"dt:day, region"``.

        The same information `fathom.yml` spells out as a list of mappings, in one
        line — for tests, notebooks, and anywhere the ceremony outweighs the spec.

        Args:
            text: Comma-separated fields. ``name:grain`` is a time field,
                a bare ``name`` is a value field. An empty string is unpartitioned.

        Example:
            >>> PartitionSpec.parse("dt:day, region").fields
            (PartitionField.time('dt', day), PartitionField.value('region'))
            >>> PartitionSpec.parse("") is UNPARTITIONED
            False
            >>> len(PartitionSpec.parse(""))
            0
        """
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return cls.of(*(PartitionField.parse(p) for p in parts))


UNPARTITIONED = PartitionSpec()


@dataclass(frozen=True)
class KeyPredicate:
    """A constraint over one dataset's partition space.

    Each field is bound either to a concrete value or to ``ANY``. A predicate with
    every field ``ANY`` denotes the whole dataset, which is how the planner represents
    "we could not prove anything narrower, rebuild it all".

    This is what you seed a plan with, and what a plan hands back.

    Example:
        >>> from datetime import datetime
        >>> key = KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
        >>> str(key)
        'dt=2026-03-14T00:00:00/region=eu'
        >>> key.get("region")
        'eu'
        >>> key.get("tenant") is ANY          # unconstrained fields read as ANY
        True
    """

    bindings: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings, key=lambda kv: kv[0])))

    def __repr__(self) -> str:
        return f"KeyPredicate({str(self)!r})"

    @classmethod
    def of(cls, **bindings: Any) -> KeyPredicate:
        """Build a predicate from keyword bindings — the readable constructor.

        Field order does not matter; bindings are sorted by name so two predicates
        over the same partition compare and hash equal however they were written.

        Example:
            >>> KeyPredicate.of(region="eu", dt=None) == KeyPredicate.of(dt=None, region="eu")
            True
        """
        return cls(bindings=tuple(bindings.items()))

    @classmethod
    def parse(cls, text: str, spec: PartitionSpec | None = None) -> KeyPredicate:
        """Build a predicate from the ``field=value,field=value`` form.

        This is the syntax `fathom plan --dirty` takes after the ``@``, and the form
        `__str__` prints, so a partition named in a plan can be pasted straight back
        into the next command.

        When a `spec` is given, fields it calls time fields have their values parsed
        as ISO datetimes and truncated to the field's grain. Without a spec every
        value stays a string — which will not match a partition key read from a
        catalog, so pass the spec whenever you have one.

        Args:
            text: ``dt=2026-03-14,region=eu``. An empty string binds nothing.
            spec: The dataset's partition spec, used to type the values.

        Raises:
            ValueError: A chunk has no ``=``, or a time field's value is not an ISO
                datetime. Both messages quote the offending text.

        Example:
            >>> spec = PartitionSpec.parse("dt:day, region")
            >>> KeyPredicate.parse("dt=2026-03-14,region=eu", spec)
            KeyPredicate('dt=2026-03-14T00:00:00/region=eu')
            >>> KeyPredicate.parse("")
            KeyPredicate('<whole dataset>')
        """
        pairs: list[tuple[str, Any]] = []
        for chunk in (c.strip() for c in text.split(",")):
            if not chunk:
                continue
            name, sep, raw = chunk.partition("=")
            name, raw = name.strip(), raw.strip()
            if not sep or not name:
                raise ValueError(
                    f"{chunk!r} is not a partition binding; write `field=value`, "
                    "for example 'dt=2026-03-14,region=eu'"
                )
            field = spec.field(name) if spec is not None else None
            if field is not None and field.kind == "time":
                assert field.grain is not None
                try:
                    parsed = datetime.fromisoformat(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{raw!r} is not an ISO datetime, and {name!r} is a time field "
                        f"at {field.grain.label} grain. Write it as '2026-03-14' or "
                        f"'2026-03-14T00:00:00'"
                    ) from exc
                pairs.append((name, truncate_to(parsed, field.grain)))
            elif raw == "*":
                pairs.append((name, ANY))
            else:
                pairs.append((name, raw))
        return cls(bindings=tuple(pairs))

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

    Example:
        >>> whole = KeyPredicate.of(dt=ANY, region=ANY)
        >>> one = KeyPredicate.of(dt="2026-03-14", region="eu")
        >>> subsumes(whole, one)      # the whole dataset covers one partition
        True
        >>> subsumes(one, whole)      # but not the other way round
        False
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
    """True when any candidate predicate subsumes `key`.

    How the planner asks "is this partition already in the dirty set?" — including
    the case where it is covered by a wider predicate rather than listed itself.

    Example:
        >>> already = [KeyPredicate.of(dt=ANY, region="eu")]
        >>> covered_by(already, KeyPredicate.of(dt="2026-03-14", region="eu"))
        True
        >>> covered_by(already, KeyPredicate.of(dt="2026-03-14", region="us"))
        False
    """
    return any(subsumes(c, key) for c in candidates)


class _Described(StrEnum):
    """A capability enum whose members can explain themselves.

    These four enums are the vocabulary of the adapter capability matrix, and the
    matrix is the first thing a new user reads when a plan comes back coarser than
    they expected. A name alone does not answer "is `list_diff` a problem?", so each
    member carries a sentence, and `fathom explain` and `fathom adapters --verbose`
    print it rather than restating the constant.
    """

    @property
    def description(self) -> str:
        """One sentence on what this member means in practice."""
        return _DESCRIPTIONS.get(f"{type(self).__name__}.{self.name}", "")

    def __format__(self, spec: str) -> str:
        # StrEnum formats as its value already; keeping that explicit means a change
        # to the base class cannot silently alter every message that interpolates one.
        return format(self.value, spec)

    @classmethod
    def describe(cls) -> dict[str, str]:
        """Every member of this enum mapped to its explanation, best first.

        Example:
            >>> Pushdown.describe()["none"]
            'The source computes nothing for us, so profiling reads the data itself.'
        """
        return {m.value: m.description for m in cls}


class LineageSource(_Described):
    """How an adapter learns what depends on what, best first.

    Example:
        >>> LineageSource.NATIVE.description
        'The platform maintains a lineage table and we read it. Most accurate.'
    """

    NATIVE = "native"  # a lineage table the platform maintains for us
    LISTENER = "listener"  # execution-plan hook (Spark, Trino, Flink)
    QUERY_LOG = "query_log"  # parse historical SQL
    DECLARED = "declared"  # the user told us


class ChangeSource(_Described):
    """How an adapter learns what changed, cheapest per detected change first."""

    SNAPSHOT_DIFF = "snapshot_diff"  # Iceberg/Delta/Hudi commit metadata
    EVENTS = "events"  # S3 EventBridge, GCS Pub/Sub, ADLS Event Grid
    INVENTORY = "inventory"  # S3 Inventory / Storage Insights manifest diff
    PARTITION_MTIME = "partition_mtime"  # catalog-reported modification times
    LIST_DIFF = "list_diff"  # LIST + etag compare; fine small, ruinous large
    PROFILE_DELTA = "profile_delta"  # last resort
    WATERMARK = "watermark"  # a monotonic column we poll


class Pushdown(_Described):
    """What the source can compute for us instead of us reading the data."""

    SKETCHES = "sketches"  # HLL / t-digest state we can merge
    QUANTILES = "quantiles"
    APPROX_DISTINCT = "approx_distinct"
    NONE = "none"


class ErasureMode(_Described):
    """How subject data can actually be destroyed here."""

    DELETE_VECTOR = "delete_vector"  # Iceberg/Delta positional deletes + compaction
    CRYPTO_SHRED = "crypto_shred"  # drop the key; the only option under versioning
    REWRITE = "rewrite"  # rewrite affected objects in place
    NONE = "none"  # WORM / Object Lock: refuse, and say so


# Kept beside the enums rather than inside them: StrEnum treats any assignment in the
# class body as a member, so a per-member docstring has to live out here.
_DESCRIPTIONS: dict[str, str] = {
    "LineageSource.NATIVE": (
        "The platform maintains a lineage table and we read it. Most accurate."
    ),
    "LineageSource.LISTENER": (
        "A hook on the execution plan reports inputs and outputs as jobs run. "
        "Accurate, but only covers jobs that ran while the listener was installed."
    ),
    "LineageSource.QUERY_LOG": (
        "Historical SQL is parsed. Covers everything that ran, and loses whatever "
        "the parser cannot read — a MERGE or an opaque UDF widens to unbounded."
    ),
    "LineageSource.DECLARED": (
        "You told us, in fathom.yml. As right as what you wrote, and it does not "
        "notice when the pipeline changes underneath it."
    ),
    "ChangeSource.SNAPSHOT_DIFF": (
        "Table-format commit metadata names the files each commit touched. Exact, "
        "cheap, and partition-scoped — the best case."
    ),
    "ChangeSource.EVENTS": (
        "The object store pushes a notification per object written. Exact and cheap, "
        "but only from the moment the subscription existed."
    ),
    "ChangeSource.INVENTORY": (
        "A daily storage inventory manifest is diffed. Cheap at any scale, and no "
        "finer than the inventory's own schedule — usually a day behind."
    ),
    "ChangeSource.PARTITION_MTIME": (
        "The catalog reports a modification time per partition. Partition-scoped, "
        "and blind to a rewrite that preserved the timestamp."
    ),
    "ChangeSource.LIST_DIFF": (
        "Every object is listed and its etag compared against last time. Correct "
        "anywhere, and the cost grows with the dataset rather than with the change."
    ),
    "ChangeSource.PROFILE_DELTA": (
        "Change is inferred from a profile moving. Last resort: it sees only changes "
        "large enough to move a statistic."
    ),
    "ChangeSource.WATERMARK": (
        "A monotonic column is polled for its high-water mark. Cheap, and it cannot "
        "see a restatement of history — only new rows past the mark."
    ),
    "Pushdown.SKETCHES": (
        "The source hands back mergeable sketch state (HLL, t-digest), so profiles "
        "combine across partitions without re-reading anything."
    ),
    "Pushdown.QUANTILES": "The source computes quantiles for us; distributions cost no scan.",
    "Pushdown.APPROX_DISTINCT": (
        "The source computes approximate distinct counts, which is what cardinality "
        "and re-identification risk need."
    ),
    "Pushdown.NONE": "The source computes nothing for us, so profiling reads the data itself.",
    "ErasureMode.DELETE_VECTOR": (
        "Positional deletes mark the subject's rows, and compaction removes them. "
        "The rows are gone once compaction runs, and not before."
    ),
    "ErasureMode.CRYPTO_SHRED": (
        "The subject's encryption key is destroyed, leaving unreadable bytes. The "
        "only option where the data itself cannot be rewritten, such as under "
        "object versioning."
    ),
    "ErasureMode.REWRITE": (
        "Affected objects are rewritten without the subject's rows. Correct, and it "
        "costs a rewrite of every file that held them."
    ),
    "ErasureMode.NONE": (
        "Nothing here can destroy data — WORM or Object Lock. An erasure targeting "
        "it is refused rather than reported complete."
    ),
}


@dataclass(frozen=True)
class Capabilities:
    """What a given adapter can actually do.

    The planner degrades rather than fails: an adapter that reports `LIST_DIFF` and
    `Pushdown.NONE` still works, it is just slower and coarser.

    Read this before blaming a coarse plan on the planner. Most surprises — no
    column lineage, whole-dataset invalidation, a profile that costs a full scan —
    are the adapter's declared limits showing through, and they are printed by
    `fathom adapters`.

    Example:
        >>> caps = Capabilities(LineageSource.NATIVE, ChangeSource.SNAPSHOT_DIFF)
        >>> print(caps.summary())
        lineage native, change snapshot_diff, no pushdown, erasure unsupported, \
dataset-level lineage, not partition-aware
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

    def summary(self) -> str:
        """One line naming what this adapter can and cannot do."""
        parts = [
            f"lineage {self.lineage.value}",
            f"change {self.change.value}",
            "no pushdown" if self.pushdown is Pushdown.NONE else f"pushdown {self.pushdown.value}",
            "erasure unsupported"
            if self.erasure is ErasureMode.NONE
            else f"erasure via {self.erasure.value}",
            "column-level lineage" if self.column_lineage else "dataset-level lineage",
            "partition-aware" if self.partition_aware else "not partition-aware",
        ]
        if self.freshness_lag is not None:
            parts.append(f"metadata lags up to {self.freshness_lag}")
        return ", ".join(parts)

    def explain(self) -> list[str]:
        """A line per capability, each saying what it means for your plans.

        What `fathom adapters --verbose` prints. The point is that "list_diff" is
        not self-explanatory, and the consequence — cost that grows with the dataset
        rather than with the change — is the thing worth knowing before adopting it.
        """
        lines = [
            f"lineage    {self.lineage.value:<16} {self.lineage.description}",
            f"change     {self.change.value:<16} {self.change.description}",
            f"pushdown   {self.pushdown.value:<16} {self.pushdown.description}",
            f"erasure    {self.erasure.value:<16} {self.erasure.description}",
        ]
        lines.append(
            "columns    "
            + (
                "yes              Column-level edges, so drift attributes to a column."
                if self.column_lineage
                else "no               Dataset-level edges only; drift attributes to a table."
            )
        )
        lines.append(
            "partitions "
            + (
                "yes              Change is reported per partition."
                if self.partition_aware
                else "no               Change is reported for the whole dataset."
            )
        )
        if self.freshness_lag is not None:
            lines.append(
                f"lag        {str(self.freshness_lag):<16} Metadata can be this stale; "
                "resume tokens are held back by it."
            )
        return lines
