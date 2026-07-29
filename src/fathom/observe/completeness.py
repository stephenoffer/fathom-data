"""Whether a dataset has the partitions it was supposed to have.

Every other check in `observe` reads data that arrived and asks whether it looks
right. This one asks the question those cannot: **what if nothing arrived at all?**

A partition that was never written and a partition that legitimately holds no rows
are indistinguishable from downstream. Both contribute nothing to a join, both make
a `SUM` smaller, and neither raises. Drift detection cannot see it either, because
drift compares profiles and a partition that does not exist has no profile to
compare. The gap is found weeks later by a person noticing a dip in a chart.

The fix is not clever, only absent from most stacks: enumerate what *should* exist
from the partition spec and a date range, compare against what does, and report the
difference as contiguous runs rather than a flat list of keys. Seven consecutive
missing days is one incident with a start and an end, not seven alerts.

**Where the expected set comes from.** Time fields enumerate from the spec's grain
over the requested range — that part is exact. Value fields cannot be enumerated
from a spec, because nothing in a spec says which regions exist. Callers may declare
a domain; absent one, the domain is inferred from the values actually observed
across the range. That inference has a specific and stated blind spot, kept in
`CompletenessReport.assumed_domains`: a value that has *never* appeared cannot be
missed, so a region that was switched off on day one is invisible here. Declaring
the domain is what closes it.

**Arrivals** are the other half. Knowing a partition exists says nothing about
whether it arrived on time, arrived twice, or arrived twice with different contents.
The last is a restatement and is the one that silently double-counts revenue, so it
is separated from an idempotent replay by comparing digests rather than by counting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.grains import Grain, span, step, truncate
from ..core.types import ANY, DatasetId, KeyPredicate, PartitionSpec
from ..core.util.clock import as_utc

__all__ = [
    "Arrival",
    "CompletenessReport",
    "Gap",
    "arrival_lag",
    "coverage_ratio",
    "duplicate_arrivals",
    "expected_keys",
    "gaps",
    "late_arrivals",
    "longest_gap",
    "missing",
    "observed_domains",
    "replays",
    "report",
    "restatements",
    "unexpected",
]

# Ceiling on the enumerated expected set. A five-year hourly range crossed with three
# value dimensions is millions of keys, and a completeness check that exhausts memory
# is worse than one that refuses. `expected_keys` raises rather than truncating,
# because a silently shortened expected set reports a dataset as complete.
MAX_EXPECTED_KEYS = 200_000


@dataclass(frozen=True)
class Arrival:
    """One observation that a partition was written.

    `digest` is what separates a replay from a restatement. Anything stable under
    rewrite works — a content hash, a Delta version, an etag. Empty means unknown,
    and unknown is treated as "cannot tell", never as "unchanged".
    """

    dataset: DatasetId
    key: KeyPredicate
    observed: datetime
    digest: str = ""
    row_count: int | None = None

    def __str__(self) -> str:
        return f"{self.dataset} {self.key} at {as_utc(self.observed).isoformat()}"


@dataclass(frozen=True)
class Gap:
    """A contiguous run of expected-but-absent time buckets.

    Reported as a run because that is what an incident is. `count` is the number of
    buckets, so a one-bucket gap has `start == end` and `count == 1`.
    """

    dataset: DatasetId
    field: str
    start: datetime
    end: datetime
    grain: Grain
    count: int
    within: tuple[tuple[str, object], ...] = ()  # the value-field slice this run sits in

    @property
    def is_single(self) -> bool:
        """True when the window covers exactly one partition."""
        return self.count == 1

    def __str__(self) -> str:
        where = ""
        if self.within:
            where = " [" + ", ".join(f"{k}={v}" for k, v in self.within) + "]"
        if self.is_single:
            return f"{self.field}={self.start.isoformat()}{where}"
        return (
            f"{self.field}={self.start.isoformat()}..{self.end.isoformat()} "
            f"({self.count} {self.grain.label}s){where}"
        )


def _time_field(spec: PartitionSpec) -> tuple[str, Grain] | None:
    """The first time field of a spec, which is the one gaps are reported along."""
    for f in spec.fields:
        if f.kind == "time" and f.grain is not None:
            return (f.name, f.grain)
    return None


def observed_domains(
    spec: PartitionSpec, present: Iterable[KeyPredicate]
) -> dict[str, list[object]]:
    """Value-field domains inferred from the keys that exist.

    `ANY` bindings are skipped: a key unconstrained on a dimension says nothing about
    that dimension's domain.
    """
    found: dict[str, set[object]] = {f.name: set() for f in spec.fields if f.kind == "value"}
    for key in present:
        for name in found:
            value = key.get(name)
            if value is not ANY:
                found[name].add(value)
    return {name: sorted(values, key=repr) for name, values in found.items()}


def expected_keys(
    spec: PartitionSpec,
    *,
    start: datetime,
    end: datetime,
    domains: Mapping[str, Sequence[object]] | None = None,
    max_keys: int = MAX_EXPECTED_KEYS,
) -> list[KeyPredicate]:
    """Every partition key that should exist between `start` and `end`, inclusive.

    Time fields enumerate from their grain. Value fields enumerate from `domains`;
    one absent from `domains` collapses to a single `ANY` binding, which matches any
    value of that field rather than claiming to know them.

    Raises `ValueError` past `max_keys` rather than truncating — a shortened expected
    set makes an incomplete dataset look complete, which is the failure this module
    exists to prevent.
    """
    if end < start:
        return []

    domains = domains or {}
    per_field: list[tuple[str, list[object]]] = []
    for f in spec.fields:
        if f.kind == "time":
            assert f.grain is not None
            per_field.append((f.name, list(span(start, end, f.grain))))
        elif f.name in domains:
            per_field.append((f.name, list(domains[f.name])))
        else:
            per_field.append((f.name, [ANY]))

    total = 1
    for _, values in per_field:
        total *= max(len(values), 1)
    if total > max_keys:
        raise ValueError(
            f"expected set of {total} keys exceeds max_keys={max_keys}; "
            "narrow the range or declare fewer value-field domains"
        )

    combos: list[list[tuple[str, object]]] = [[]]
    for name, values in per_field:
        combos = [prefix + [(name, v)] for prefix in combos for v in values]
    return [KeyPredicate(bindings=tuple(c)) for c in combos]


def missing(
    expected: Iterable[KeyPredicate], present: Iterable[KeyPredicate]
) -> list[KeyPredicate]:
    """Expected keys with no matching present key.

    Comparison is exact rather than by subsumption: a present key bound to `ANY` on
    some dimension is a key whose partitioning we could not read, and letting it
    absorb every expected key would report a dataset with one unreadable partition
    as complete.
    """
    have = set(present)
    return [k for k in expected if k not in have]


def unexpected(
    expected: Iterable[KeyPredicate], present: Iterable[KeyPredicate]
) -> list[KeyPredicate]:
    """Present keys nobody expected — usually a spec that has drifted from the data."""
    want = set(expected)
    return [k for k in present if k not in want]


def gaps(dataset: DatasetId, spec: PartitionSpec, absent: Iterable[KeyPredicate]) -> list[Gap]:
    """Collapse missing keys into contiguous runs along the spec's time field.

    Runs are computed within each distinct combination of the value fields, because
    `region=eu` missing Monday through Wednesday and `region=us` missing only Tuesday
    are two incidents, and merging them would misreport both.
    """
    time_field = _time_field(spec)
    if time_field is None:
        return []
    name, grain = time_field
    value_names = tuple(f.name for f in spec.fields if f.kind == "value")

    by_slice: dict[tuple[tuple[str, object], ...], list[datetime]] = {}
    for key in absent:
        when = key.get(name)
        if not isinstance(when, datetime):
            continue
        slice_key = tuple((v, key.get(v)) for v in value_names)
        by_slice.setdefault(slice_key, []).append(truncate(when, grain))

    out: list[Gap] = []
    for slice_key, moments in by_slice.items():
        ordered = sorted(set(moments))
        run_start = run_end = ordered[0]
        run_len = 1
        for current in ordered[1:]:
            # Contiguous exactly when nothing sits between them at this grain.
            if len(span(run_end, current, grain)) == 2:
                run_end = current
                run_len += 1
                continue
            out.append(Gap(dataset, name, run_start, run_end, grain, run_len, slice_key))
            run_start = run_end = current
            run_len = 1
        out.append(Gap(dataset, name, run_start, run_end, grain, run_len, slice_key))

    return sorted(out, key=lambda g: (g.start, repr(g.within)))


@dataclass
class CompletenessReport:
    """What should exist against what does."""

    dataset: DatasetId
    expected: int = 0
    present: int = 0
    absent: list[KeyPredicate] = field(default_factory=list)
    runs: list[Gap] = field(default_factory=list)
    surplus: list[KeyPredicate] = field(default_factory=list)
    assumed_domains: dict[str, list[object]] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """True when every expected partition is present."""
        return not self.absent

    @property
    def ratio(self) -> float:
        """Fraction of expected partitions that are present. 1.0 when nothing expected."""
        if self.expected == 0:
            return 1.0
        return (self.expected - len(self.absent)) / self.expected

    def summary(self) -> str:
        """The report as text, missing partitions named."""
        if self.is_complete:
            return f"complete: {self.present}/{self.expected} partitions present"
        lines = [
            f"incomplete: {len(self.absent)} of {self.expected} partitions missing "
            f"({self.ratio:.0%} present), in {len(self.runs)} run(s)"
        ]
        lines.extend(f"    {run}" for run in self.runs[:10])
        if len(self.runs) > 10:
            lines.append(f"    +{len(self.runs) - 10} more")
        if self.assumed_domains:
            inferred = ", ".join(
                f"{k} ({len(v)} value(s))" for k, v in sorted(self.assumed_domains.items())
            )
            lines.append(
                f"    domains inferred from observed data: {inferred} — "
                "a value that never appeared cannot be reported missing"
            )
        if self.surplus:
            lines.append(f"    {len(self.surplus)} unexpected partition(s) present")
        return "\n".join(lines)


def report(
    dataset: DatasetId,
    spec: PartitionSpec,
    present: Iterable[KeyPredicate],
    *,
    start: datetime,
    end: datetime,
    domains: Mapping[str, Sequence[object]] | None = None,
    max_keys: int = MAX_EXPECTED_KEYS,
) -> CompletenessReport:
    """Compare the partitions a dataset should have against the ones it does.

    Value-field domains not given in `domains` are inferred from `present` and
    recorded in `assumed_domains`, so the report states the assumption it made rather
    than presenting an inferred domain as a known one.
    """
    have = list(present)
    inferred = observed_domains(spec, have)
    declared = dict(domains or {})
    assumed = {k: v for k, v in inferred.items() if k not in declared and v}
    effective: dict[str, Sequence[object]] = {**assumed, **declared}

    want = expected_keys(spec, start=start, end=end, domains=effective, max_keys=max_keys)
    absent = missing(want, have)
    return CompletenessReport(
        dataset=dataset,
        expected=len(want),
        present=len(have),
        absent=absent,
        runs=gaps(dataset, spec, absent),
        surplus=unexpected(want, have),
        assumed_domains=assumed,
    )


def coverage_ratio(result: CompletenessReport) -> float:
    """Alias for `CompletenessReport.ratio`, for symmetry with `metrics`."""
    return result.ratio


def longest_gap(result: CompletenessReport) -> Gap | None:
    """The longest contiguous run of missing buckets, which is the one to triage."""
    return max(result.runs, key=lambda g: g.count, default=None)


# -- arrivals ------------------------------------------------------------------


def arrival_lag(arrival: Arrival, *, field_name: str, grain: Grain) -> timedelta | None:
    """How long after its own bucket closed a partition actually landed.

    Measured from the *end* of the bucket, so a day partition written at 02:00 the
    following morning has a lag of two hours rather than twenty-six. Returns `None`
    when the key carries no datetime on that field.
    """
    when = arrival.key.get(field_name)
    if not isinstance(when, datetime):
        return None
    # Measured from where the bucket closes, which is where the next one begins.
    bucket_end = step(truncate(when, grain), 1, grain)
    return as_utc(arrival.observed).replace(tzinfo=None) - bucket_end


def late_arrivals(
    arrivals: Iterable[Arrival],
    *,
    field_name: str,
    grain: Grain,
    tolerance: timedelta,
) -> list[tuple[Arrival, timedelta]]:
    """Arrivals that landed more than `tolerance` after their bucket closed."""
    out: list[tuple[Arrival, timedelta]] = []
    for arrival in arrivals:
        lag = arrival_lag(arrival, field_name=field_name, grain=grain)
        if lag is not None and lag > tolerance:
            out.append((arrival, lag))
    return sorted(out, key=lambda pair: pair[1], reverse=True)


def _by_key(arrivals: Iterable[Arrival]) -> dict[tuple[DatasetId, KeyPredicate], list[Arrival]]:
    grouped: dict[tuple[DatasetId, KeyPredicate], list[Arrival]] = {}
    for arrival in arrivals:
        grouped.setdefault((arrival.dataset, arrival.key), []).append(arrival)
    for group in grouped.values():
        group.sort(key=lambda a: as_utc(a.observed))
    return grouped


def duplicate_arrivals(arrivals: Iterable[Arrival]) -> list[list[Arrival]]:
    """Every partition written more than once, as the full group of its arrivals.

    Says nothing about whether that was harmful — `replays` and `restatements` split
    it into the harmless and the harmful case.
    """
    return [group for group in _by_key(arrivals).values() if len(group) > 1]


def replays(arrivals: Iterable[Arrival]) -> list[list[Arrival]]:
    """Repeat arrivals whose contents were identical — an idempotent rewrite.

    Requires a digest on every arrival in the group. A group with a missing digest is
    not a proven replay, so it is excluded here and reported by `restatements`.
    """
    out: list[list[Arrival]] = []
    for group in duplicate_arrivals(arrivals):
        digests = {a.digest for a in group}
        if len(digests) == 1 and "" not in digests:
            out.append(group)
    return out


def restatements(arrivals: Iterable[Arrival]) -> list[list[Arrival]]:
    """Repeat arrivals whose contents changed, or cannot be proven not to have.

    This is the one that silently double-counts. A group with an unknown digest lands
    here rather than in `replays`, because treating "cannot tell" as "unchanged" is
    exactly the assumption that lets a restatement through unnoticed.
    """
    out: list[list[Arrival]] = []
    for group in duplicate_arrivals(arrivals):
        digests = {a.digest for a in group}
        if len(digests) > 1 or "" in digests:
            out.append(group)
    return out
