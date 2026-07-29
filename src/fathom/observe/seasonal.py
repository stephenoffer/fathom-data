"""Baselines that know Tuesday from Sunday.

`quality.learn` derives bounds from what was observed, which is the right default and
wrong for anything with a weekly or daily cycle. A B2B events table does a fifth of
its Tuesday volume on Sunday. Learn one flat band across both and you get a band wide
enough to admit Tuesday's floor and Sunday's ceiling — which is to say, wide enough to
catch nothing. Narrow it to Tuesday and it fires every weekend.

Teams resolve this by muting the check. That is the actual failure mode: not a wrong
alert, an absent one.

The fix is to bucket observations by their position in the cycle and learn a band per
bucket. Three commitments keep it from being worse than the flat version:

- **A bucket with too few observations is not modelled.** Two Sundays is not a
  baseline for Sunday. Such buckets go in `unmodelled` and are *not checked* — an
  invented band is worse than no band, because it carries the same authority.
- **Seasonality is measured before it is assumed.** `strength` compares between-bucket
  variation against within-bucket variation. Near zero means the data has no cycle on
  this period and a flat bound from `quality.learn` is the better tool. The number is
  reported so that stays a decision rather than a default.
- **Bounds widen, never tighten** — the same rule the rest of `quality` follows, for
  the same reason.

This module learns bands and checks against them. It does not schedule, alert, or
decide severity beyond what it was given; those belong to whatever runs it.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from ..core.types import DatasetId
from .profile import Finding, Profile, Severity
from .quality import DEFAULT_MARGIN

__all__ = [
    "Cycle",
    "Observation",
    "SeasonalBaseline",
    "SeasonalBand",
    "bands_for",
    "bucket_of",
    "check_seasonal",
    "learn_seasonal",
    "observations_from",
    "strength",
    "unmodelled_buckets",
]

# Below this many observations a bucket is left unmodelled. Four is the smallest count
# from which a range plus a margin is more signal than accident — one holiday Monday in
# three would otherwise set the floor for every Monday.
MIN_OBSERVATIONS = 4


class Cycle(StrEnum):
    """Which repeating position an observation is bucketed by."""

    HOUR_OF_DAY = "hour_of_day"  # 0..23 — intraday load
    DAY_OF_WEEK = "day_of_week"  # 0..6, Monday 0 — the common case
    DAY_OF_MONTH = "day_of_month"  # 1..31 — billing and close cycles
    MONTH_OF_YEAR = "month_of_year"  # 1..12 — retail seasonality


def bucket_of(when: datetime, cycle: Cycle) -> int:
    """Which bucket of `cycle` a moment falls in."""
    if cycle is Cycle.HOUR_OF_DAY:
        return when.hour
    if cycle is Cycle.DAY_OF_WEEK:
        return when.weekday()
    if cycle is Cycle.DAY_OF_MONTH:
        return when.day
    return when.month


def _label(cycle: Cycle, index: int) -> str:
    if cycle is Cycle.DAY_OF_WEEK:
        return ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[index]
    if cycle is Cycle.HOUR_OF_DAY:
        return f"{index:02d}:00"
    if cycle is Cycle.MONTH_OF_YEAR:
        return (
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        )[index - 1]
    return f"day {index}"


@dataclass(frozen=True)
class Observation:
    """One profile, and when the data it describes belongs to.

    `when` is the partition's own moment, not when profiling ran. Bucketing by run
    time would put a Monday partition backfilled on Saturday into the Saturday band,
    which is how a backfill starts failing its own checks.
    """

    when: datetime
    profile: Profile


@dataclass(frozen=True)
class SeasonalBand:
    """The learned range of one metric, in one bucket of the cycle."""

    column: str | None  # None for dataset-level metrics such as row_count
    metric: str
    bucket: int
    low: float
    high: float
    observations: int
    label: str = ""

    def contains(self, value: float) -> bool:
        """True when a value falls inside the modelled band."""
        return self.low <= value <= self.high

    def __str__(self) -> str:
        where = f"{self.column}.{self.metric}" if self.column else self.metric
        return f"{where} @ {self.label or self.bucket}: [{self.low:g}, {self.high:g}]"


@dataclass
class SeasonalBaseline:
    """Learned bands per cycle bucket, and an explicit record of what was not learned."""

    dataset: DatasetId
    cycle: Cycle
    bands: dict[tuple[str | None, str, int], SeasonalBand] = field(default_factory=dict)
    unmodelled: dict[int, int] = field(default_factory=dict)  # bucket -> observations seen
    margin: float = DEFAULT_MARGIN

    def band(self, column: str | None, metric: str, bucket: int) -> SeasonalBand | None:
        """Lower and upper bound for one bucket."""
        return self.bands.get((column, metric, bucket))

    @property
    def modelled_buckets(self) -> list[int]:
        """Buckets with enough history to model."""
        return sorted({b for _, _, b in self.bands})

    @property
    def is_usable(self) -> bool:
        """True when at least one bucket was modelled."""
        return bool(self.bands)

    def summary(self) -> str:
        """The model as text, with how much history backs it."""
        if not self.is_usable:
            seen = sum(self.unmodelled.values())
            return (
                f"no seasonal baseline for {self.dataset}: {seen} observation(s) across "
                f"{len(self.unmodelled)} {self.cycle.value} bucket(s), none reaching the minimum"
            )
        lines = [
            f"seasonal baseline for {self.dataset} by {self.cycle.value}: "
            f"{len(self.bands)} band(s) across {len(self.modelled_buckets)} bucket(s)"
        ]
        if self.unmodelled:
            skipped = ", ".join(
                f"{_label(self.cycle, b)} ({n})" for b, n in sorted(self.unmodelled.items())
            )
            lines.append(
                f"    not modelled, too few observations: {skipped} — these are not checked"
            )
        return "\n".join(lines)


def _metrics(profile: Profile) -> dict[tuple[str | None, str], float]:
    """The numeric metrics of one profile, flattened to (column, metric) keys."""
    out: dict[tuple[str | None, str], float] = {(None, "row_count"): float(profile.row_count)}
    for column in profile.columns:
        if column.null_rate is not None:
            out[(column.name, "null_rate")] = column.null_rate
        if column.distinct_estimate is not None:
            out[(column.name, "distinct")] = float(column.distinct_estimate)
        for name, value in (("min", column.min), ("max", column.max)):
            if isinstance(value, int | float) and not isinstance(value, bool):
                out[(column.name, name)] = float(value)
    return out


def learn_seasonal(
    history: Sequence[Observation],
    *,
    cycle: Cycle = Cycle.DAY_OF_WEEK,
    margin: float = DEFAULT_MARGIN,
    min_observations: int = MIN_OBSERVATIONS,
) -> SeasonalBaseline:
    """Learn a band per metric per cycle bucket from observed history.

    Buckets with fewer than `min_observations` are recorded in `unmodelled` and left
    without bands, so `check_seasonal` skips them rather than testing against a range
    invented from two Sundays.
    """
    if not history:
        raise ValueError("cannot learn a seasonal baseline from no observations")

    dataset = history[0].profile.dataset
    grouped: dict[int, list[Observation]] = {}
    for item in history:
        grouped.setdefault(bucket_of(item.when, cycle), []).append(item)

    baseline = SeasonalBaseline(dataset=dataset, cycle=cycle, margin=margin)
    for index, items in sorted(grouped.items()):
        if len(items) < min_observations:
            baseline.unmodelled[index] = len(items)
            continue
        values: dict[tuple[str | None, str], list[float]] = {}
        for item in items:
            for key, value in _metrics(item.profile).items():
                values.setdefault(key, []).append(value)
        for (column, metric), series in values.items():
            if len(series) < min_observations:
                # A metric that only appeared in some of the bucket's profiles — a
                # column added partway through. Not enough of it to bound.
                continue
            low, high = min(series), max(series)
            pad = max(abs(high - low) * margin, abs(high) * margin, 1e-9)
            baseline.bands[(column, metric, index)] = SeasonalBand(
                column=column,
                metric=metric,
                bucket=index,
                low=low - pad,
                high=high + pad,
                observations=len(series),
                label=_label(cycle, index),
            )
    return baseline


def check_seasonal(
    observation: Observation,
    baseline: SeasonalBaseline,
    *,
    severity: Severity = Severity.WARN,
) -> list[Finding]:
    """Check one profile against the band for its own bucket of the cycle.

    A bucket with no band produces no findings. That is deliberate and is the whole
    reason `unmodelled` is part of the baseline: silence here means "not modelled",
    and the baseline says which buckets those are.
    """
    index = bucket_of(observation.when, baseline.cycle)
    label = _label(baseline.cycle, index)
    findings: list[Finding] = []

    for (column, metric), value in sorted(_metrics(observation.profile).items(), key=repr):
        band = baseline.band(column, metric, index)
        if band is None or band.contains(value):
            continue
        direction = "above" if value > band.high else "below"
        bound = band.high if value > band.high else band.low
        where = f"{column}.{metric}" if column else metric
        findings.append(
            Finding(
                column=column,
                kind=f"seasonal_{metric}",
                severity=severity,
                detail=(
                    f"{where} is {value:g}, {direction} the {label} band "
                    f"[{band.low:g}, {band.high:g}] learned from {band.observations} observation(s)"
                ),
                before=bound,
                after=value,
            )
        )
    return findings


def strength(
    history: Sequence[Observation],
    *,
    cycle: Cycle = Cycle.DAY_OF_WEEK,
    column: str | None = None,
    metric: str = "row_count",
) -> float | None:
    """How much of the variation in a metric is explained by the cycle, in `[0, 1]`.

    The ratio of between-bucket variance to total variance. Near 1 means the cycle
    explains almost everything and a seasonal baseline is clearly right; near 0 means
    it explains nothing and `quality.learn`'s flat bound is the better tool with less
    machinery behind it.

    Returns `None` when there is not enough spread to compute a ratio — fewer than two
    buckets, or a metric that never varies. That is not zero seasonality; it is no
    answer, and the difference matters when the number is used to choose an approach.
    """
    series: dict[int, list[float]] = {}
    everything: list[float] = []
    for item in history:
        value = _metrics(item.profile).get((column, metric))
        if value is None:
            continue
        series.setdefault(bucket_of(item.when, cycle), []).append(value)
        everything.append(value)

    if len(series) < 2 or len(everything) < 3:
        return None
    total = statistics.pvariance(everything)
    if total == 0:
        return None

    grand = statistics.fmean(everything)
    between = sum(len(v) * (statistics.fmean(v) - grand) ** 2 for v in series.values())
    return max(0.0, min(1.0, between / (total * len(everything))))


def unmodelled_buckets(baseline: SeasonalBaseline) -> list[str]:
    """Human-readable labels of the buckets that were not modelled."""
    return [_label(baseline.cycle, b) for b in sorted(baseline.unmodelled)]


def bands_for(baseline: SeasonalBaseline, bucket: int) -> list[SeasonalBand]:
    """Every band learned for one bucket of the cycle."""
    return sorted(
        (b for (_, _, index), b in baseline.bands.items() if index == bucket),
        key=lambda b: (b.column or "", b.metric),
    )


def observations_from(
    profiles: Iterable[Profile], *, when: Sequence[datetime]
) -> list[Observation]:
    """Zip profiles with the moments they describe.

    A convenience for callers holding two parallel sequences out of a store query;
    raises rather than zipping short, since a silent truncation would drop the most
    recent observations and skew every band it learned.
    """
    materialized = list(profiles)
    if len(materialized) != len(when):
        raise ValueError(
            f"{len(materialized)} profile(s) against {len(when)} moment(s); "
            "they must correspond one to one"
        )
    return [Observation(w, p) for w, p in zip(when, materialized, strict=True)]
