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

from .util.text import did_you_mean, options

__all__ = ["Grain", "truncate", "step", "span", "convert_window"]


class Grain(IntEnum):
    """Time partition granularity, ordered fine to coarse.

    Ordering is the point: `Grain.DAY < Grain.MONTH` is true, and the planner leans
    on it constantly to decide whether an edge is a rollup (safe to reason about) or
    a refinement (widened to unbounded).

    Example:
        >>> Grain.parse("daily") is Grain.DAY
        True
        >>> Grain.DAY < Grain.MONTH
        True
        >>> str(Grain.MONTH)
        'month'
    """

    HOUR = 1
    DAY = 2
    MONTH = 3
    YEAR = 4

    @property
    def label(self) -> str:
        """Lower-case name of this grain, as it appears in config."""
        return self.name.lower()

    def __str__(self) -> str:
        """The config spelling, so f-strings read `day` rather than `Grain.DAY`."""
        return self.label

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Every grain spelling this accepts, fine to coarse.

        Example:
            >>> Grain.names()
            ('hour', 'day', 'month', 'year')
        """
        return tuple(g.label for g in cls)

    @classmethod
    def parse(cls, s: str | Grain) -> Grain:
        """Resolve a grain from its name, in any of the spellings people write.

        Accepts the canonical name (``day``), the adjective (``daily``), the plural
        (``days``), and the single-letter abbreviation (``d``) — because config
        files, CLI flags, and cron descriptions each favour a different one, and
        rejecting the other three teaches nothing. A `Grain` passes through
        unchanged, so callers can accept ``Grain | str`` without branching.

        Args:
            s: A grain name, or a `Grain` to return unchanged.

        Returns:
            The matching `Grain`.

        Raises:
            ValueError: The name is not a grain. The message lists every accepted
                value and suggests the nearest one.

        Example:
            >>> Grain.parse("day"), Grain.parse("hourly"), Grain.parse("M")
            (<Grain.DAY: 2>, <Grain.HOUR: 1>, <Grain.MONTH: 3>)
            >>> Grain.parse(Grain.YEAR) is Grain.YEAR
            True
        """
        if isinstance(s, Grain):
            return s
        if not isinstance(s, str):
            raise ValueError(
                f"a grain must be a string or a Grain, not {type(s).__name__}. "
                f"Pass {options(cls.names())}, or a `Grain` member such as `Grain.DAY`"
            )
        key = s.strip().lower()
        found = _GRAIN_ALIASES.get(key)
        if found is not None:
            return found
        raise ValueError(
            # Suggest from the canonical names only: proposing 'daily' when 'day' is
            # the spelling every example uses sends the reader to the wrong one.
            f"unknown grain {s!r}; expected {options(cls.names())}{did_you_mean(key, cls.names())}"
        )


# Every spelling of a grain we accept. Config files write `day`, cron descriptions
# write `daily`, retention policies write `days`, and abbreviations turn up in flags.
# They all mean one thing, so they all resolve to one member rather than to four
# different error messages.
_GRAIN_ALIASES: dict[str, Grain] = {
    alias: grain
    for grain, aliases in (
        (Grain.HOUR, ("hour", "hourly", "hours", "hr", "hrs", "h")),
        (Grain.DAY, ("day", "daily", "days", "d")),
        (Grain.MONTH, ("month", "monthly", "months", "mon", "m")),
        (Grain.YEAR, ("year", "yearly", "years", "annual", "annually", "yr", "y")),
    )
    for alias in aliases
}


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

    Args:
        dt: Any datetime, aware or naive. Aware values convert to UTC first.
        grain: The bucket size to round down to.

    Returns:
        The start of the enclosing bucket, always naive UTC.

    Example:
        >>> from datetime import datetime
        >>> truncate(datetime(2026, 3, 14, 17, 42), Grain.DAY)
        datetime.datetime(2026, 3, 14, 0, 0)
        >>> truncate(datetime(2026, 3, 14, 17, 42), Grain.MONTH)
        datetime.datetime(2026, 3, 1, 0, 0)
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
    """Advance `dt` by `n` whole `grain` units. `n` may be negative.

    Calendar-aware, so stepping a month lands on the same day of the next month
    rather than 30 days later, and the day is clamped when the target month is
    shorter — 31 January plus one month is 28 February, not an exception.

    Example:
        >>> from datetime import datetime
        >>> step(datetime(2026, 1, 31), 1, Grain.MONTH)
        datetime.datetime(2026, 2, 28, 0, 0)
        >>> step(datetime(2026, 3, 14), -2, Grain.DAY)
        datetime.datetime(2026, 3, 12, 0, 0)
    """
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
    """Every `grain` bucket start from `start` through `end`, inclusive.

    An `end` before `start` is an empty range rather than an error, because it is
    the natural result of an empty window and callers should not have to guard it.

    Args:
        start: First moment of interest; truncated to its bucket.
        end: Last moment of interest; truncated to its bucket, and included.
        grain: Bucket size to walk in.

    Returns:
        Bucket starts in ascending order. Empty when `end` precedes `start`.

    Raises:
        ValueError: The range covers more than 100,000 buckets.

    Example:
        >>> from datetime import datetime
        >>> span(datetime(2026, 3, 14), datetime(2026, 3, 16), Grain.DAY)
        [datetime.datetime(2026, 3, 14, 0, 0), datetime.datetime(2026, 3, 15, 0, 0), \
datetime.datetime(2026, 3, 16, 0, 0)]
        >>> span(datetime(2026, 3, 16), datetime(2026, 3, 14), Grain.DAY)
        []
    """
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
    raise ValueError(
        f"span from {start} to {end} at {grain.label} grain exceeds 100,000 buckets. "
        f"Narrow the range, or use a coarser grain — at {grain.label} grain this range "
        f"would enumerate every bucket individually, which is never what a plan wants"
    )


def convert_window(lo: int, hi: int, frm: Grain, to: Grain) -> tuple[int, int] | None:
    """Re-express the offset window `[lo, hi]` from `frm` units into `to` units.

    Returns `None` when `to` is finer than `frm`, which callers treat as unbounded.

    Two sources of widening, both deliberate:

    - **Straddle.** Six days from an arbitrary start can land in the next month, so
      `+6 day` becomes `+2 month` rather than `+0`.
    - **Anchor slack.** The window is measured from the input's own bucket start,
      which can sit up to one `to`-unit after the coarser bucket start we re-anchor
      on. The extra `+1` on the upper bound absorbs that.

    Both widenings are why a converted window is bigger than arithmetic suggests.
    That is the invariant working, not a bug: the result must cover every bucket
    the finer window could touch, whatever day of the month the input lands on.

    Example:
        >>> convert_window(0, 0, Grain.DAY, Grain.DAY)      # same grain, unchanged
        (0, 0)
        >>> convert_window(0, 6, Grain.DAY, Grain.MONTH)    # 7 days can straddle
        (0, 3)
        >>> convert_window(0, 0, Grain.MONTH, Grain.DAY) is None   # refinement
        True
    """
    if frm is to:
        return (lo, hi)
    if frm > to:  # coarse -> fine; refuse rather than emit a near-total range
        return None

    divisor = _MIN_FINE_PER_COARSE[(to, frm)]
    out_lo = 0 if lo >= 0 else -(ceil(-lo / divisor) + 1)
    out_hi = (0 if hi <= 0 else ceil(hi / divisor) + 1) + 1
    return (out_lo, out_hi)
