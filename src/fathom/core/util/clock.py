"""Timestamps, and the one rule about them.

Four modules independently grew a private `_aware()` that attaches UTC to a naive
datetime. They had to, because comparing a naive datetime to an aware one raises,
and half the timestamps in this system come from a provider SDK that returns naive
values while the other half come from `datetime.now(UTC)`.

The rule: **a naive timestamp is UTC.** Not local time. Assuming local would make
freshness and retention wrong by hours for anyone west of Greenwich, silently, and
only in production.

`now()` exists so tests can pass a fixed reference through the same parameter every
caller already takes, rather than patching the clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

__all__ = ["age", "as_utc", "is_older_than", "now"]


def as_utc(stamp: datetime) -> datetime:
    """Attach UTC to a naive timestamp. Aware timestamps pass through untouched."""
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def now(reference: datetime | None = None) -> datetime:
    """The current time, or the reference a caller supplied instead."""
    return as_utc(reference) if reference is not None else datetime.now(UTC)


def age(stamp: datetime | None, *, reference: datetime | None = None) -> timedelta | None:
    """How long ago `stamp` was, or None when there is no timestamp.

    None means unknown, and callers must not read it as zero. A dataset that was
    never built is not a dataset that was built just now.
    """
    return None if stamp is None else now(reference) - as_utc(stamp)


def is_older_than(
    stamp: datetime | None, limit: timedelta, *, reference: datetime | None = None
) -> bool:
    """True when `stamp` is past its budget. An absent timestamp is always past it."""
    elapsed = age(stamp, reference=reference)
    return True if elapsed is None else elapsed > limit
