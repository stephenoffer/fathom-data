"""Time grain arithmetic.

Partition invalidation is mostly reasoning about time: a day of source data lands,
and we need to know which months or days of derived data it dirties. That requires
truncating to a grain, stepping by whole grain units, and re-expressing a window of
offsets from one grain in another.

"Conservative" always means outward. When a conversion is inexact we widen the
range rather than narrow it, because the planner's core invariant is that it may
over-invalidate but must never under-invalidate.

Conversions only ever go fine to coarse. Re-expressing a coarse window at a finer
grain is possible in principle but the result is so wide it is indistinguishable
from "the whole dataset", so callers treat refinement as unbounded instead.
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from math import ceil

__all__ = ["Grain", "truncate", "step", "span", "convert_window"]


class Grain(IntEnum):
    """Time partition granularity, ordered fine to coarse."""

    HOUR = 1
    DAY = 2
    MONTH = 3
    YEAR = 4

    @property
    def label(self) -> str:
        """Lower-case name of this grain, as it appears in config."""
        return self.name.lower()

    @classmethod
    def parse(cls, s: str) -> Grain:
        """Resolve a grain name. Raises `ValueError` on anything unrecognised."""
        try:
            return cls[s.strip().upper()]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"unknown grain {s!r}") from exc


# Minimum number of `fine` units contained in one `coarse` unit. Used when
# converting a fine-grained offset up to a coarser grain, where dividing by the
# smallest possible divisor yields the largest (safest) result.
_MIN_FINE_PER_COARSE: dict[tuple[Grain, Grain], int] = {
    (Grain.DAY, Grain.HOUR): 23,  # 23 on a spring-forward DST day
    (Grain.MONTH, Grain.HOUR): 28 * 24 - 1,
    (Grain.MONTH, Grain.DAY): 28,
    (Grain.YEAR, Grain.HOUR): 365 * 24 - 1,
    (Grain.YEAR, Grain.DAY): 365,
    (Grain.YEAR, Grain.MONTH): 12,
}


def truncate(dt: datetime, grain: Grain) -> datetime:
    """Round `dt` down to the start of its enclosing `grain` bucket, in naive UTC.

    Partition keys are always naive UTC, and this is the single place that is
    enforced, because every source of a time partition value funnels through here.

    The reason is that `datetime(2026, 3, 14)` and
    `datetime(2026, 3, 14, tzinfo=UTC)` are unequal, hash differently, and print
    identically. Warehouse drivers hand back aware datetimes; Parquet paths and
    Delta partition values produce naive ones. Without normalizing, a plan seeded
    from Snowflake would never match a partition profiled from files, and the
    symptom would be a planner that quietly rebuilds nothing.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    if grain is Grain.HOUR:
        return dt.replace(minute=0, second=0, microsecond=0)
    if grain is Grain.DAY:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if grain is Grain.MONTH:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def step(dt: datetime, n: int, grain: Grain) -> datetime:
    """Advance `dt` by `n` whole `grain` units. `n` may be negative."""
    if n == 0:
        return dt
    if grain is Grain.HOUR:
        return dt + timedelta(hours=n)
    if grain is Grain.DAY:
        return dt + timedelta(days=n)
    if grain is Grain.MONTH:
        total = dt.year * 12 + (dt.month - 1) + n
        year, month = divmod(total, 12)
        # Clamp the day so stepping off the end of a short month stays valid.
        day = min(dt.day, calendar.monthrange(year, month + 1)[1])
        return dt.replace(year=year, month=month + 1, day=day)
    # Clamp against the target year's own month length: 29 February plus one year
    # is 28 February, and a hard-coded 29 would raise.
    target_year = dt.year + n
    return dt.replace(
        year=target_year, day=min(dt.day, calendar.monthrange(target_year, dt.month)[1])
    )


def span(start: datetime, end: datetime, grain: Grain) -> list[datetime]:
    """Every `grain` bucket start from `start` through `end`, inclusive."""
    if end < start:
        return []
    out: list[datetime] = []
    cur = truncate(start, grain)
    stop = truncate(end, grain)
    # Bound the walk so a pathological range can't hang the planner.
    for _ in range(100_000):
        out.append(cur)
        if cur >= stop:
            return out
        cur = step(cur, 1, grain)
    raise ValueError(f"span from {start} to {end} at {grain.label} grain is too large")


def convert_window(lo: int, hi: int, frm: Grain, to: Grain) -> tuple[int, int] | None:
    """Re-express the offset window `[lo, hi]` from `frm` units into `to` units.

    Returns `None` when `to` is finer than `frm`, which callers treat as unbounded.

    Two sources of widening, both deliberate:

    - **Straddle.** Six days from an arbitrary start can land in the next month, so
      `+6 day` becomes `+2 month` rather than `+0`.
    - **Anchor slack.** The window is measured from the input's own bucket start,
      which can sit up to one `to`-unit after the coarser bucket start we re-anchor
      on. The extra `+1` on the upper bound absorbs that.
    """
    if frm is to:
        return (lo, hi)
    if frm > to:  # coarse -> fine; refuse rather than emit a near-total range
        return None

    divisor = _MIN_FINE_PER_COARSE[(to, frm)]
    out_lo = 0 if lo >= 0 else -(ceil(-lo / divisor) + 1)
    out_hi = (0 if hi <= 0 else ceil(hi / divisor) + 1) + 1
    return (out_lo, out_hi)
