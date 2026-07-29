"""Grain arithmetic, including the timezone normalization every source depends on."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from fathom.core.grains import Grain, convert_window, span, step, truncate


def test_grain_ordering_is_fine_to_coarse():
    assert Grain.HOUR < Grain.DAY < Grain.MONTH < Grain.YEAR


def test_grain_parsing_is_case_insensitive():
    assert Grain.parse("Day") is Grain.DAY
    with pytest.raises(ValueError, match="unknown grain"):
        Grain.parse("fortnight")


@pytest.mark.parametrize(
    ("grain", "expected"),
    [
        (Grain.HOUR, datetime(2026, 3, 14, 15)),
        (Grain.DAY, datetime(2026, 3, 14)),
        (Grain.MONTH, datetime(2026, 3, 1)),
        (Grain.YEAR, datetime(2026, 1, 1)),
    ],
)
def test_truncation_to_each_grain(grain, expected):
    assert truncate(datetime(2026, 3, 14, 15, 45, 30), grain) == expected


# -- timezone normalization ----------------------------------------------------


def test_aware_and_naive_inputs_produce_equal_keys():
    """Warehouse drivers return aware datetimes; file paths produce naive ones.

    If these differ, a plan seeded from Snowflake never matches a partition
    profiled from Parquet, and the planner silently rebuilds nothing.
    """
    aware = truncate(datetime(2026, 3, 14, 8, tzinfo=UTC), Grain.DAY)
    naive = truncate(datetime(2026, 3, 14, 8), Grain.DAY)
    assert aware == naive
    assert hash(aware) == hash(naive)


def test_partition_keys_are_always_naive():
    assert truncate(datetime(2026, 3, 14, tzinfo=UTC), Grain.DAY).tzinfo is None


def test_offsets_convert_to_utc_before_truncating():
    """01:00 at +02:00 is 23:00 the previous day in UTC, and belongs to that day."""
    east = datetime(2026, 3, 14, 1, tzinfo=timezone(timedelta(hours=2)))
    assert truncate(east, Grain.DAY) == datetime(2026, 3, 13)


def test_negative_offsets_convert_too():
    west = datetime(2026, 3, 14, 23, tzinfo=timezone(timedelta(hours=-5)))
    assert truncate(west, Grain.DAY) == datetime(2026, 3, 15)


# -- stepping ------------------------------------------------------------------


def test_stepping_by_months_handles_year_rollover():
    assert step(datetime(2026, 11, 1), 3, Grain.MONTH) == datetime(2027, 2, 1)
    assert step(datetime(2026, 2, 1), -3, Grain.MONTH) == datetime(2025, 11, 1)


def test_stepping_off_the_end_of_a_short_month_clamps():
    assert step(datetime(2026, 1, 31), 1, Grain.MONTH) == datetime(2026, 2, 28)


def test_stepping_a_leap_day_by_a_year_clamps():
    assert step(datetime(2024, 2, 29), 1, Grain.YEAR) == datetime(2025, 2, 28)


def test_zero_step_is_identity():
    when = datetime(2026, 3, 14, 7)
    assert step(when, 0, Grain.MONTH) is when


# -- spans ---------------------------------------------------------------------


def test_span_is_inclusive_of_both_ends():
    got = span(datetime(2026, 3, 14), datetime(2026, 3, 16), Grain.DAY)
    assert got == [datetime(2026, 3, 14), datetime(2026, 3, 15), datetime(2026, 3, 16)]


def test_backwards_span_is_empty():
    assert span(datetime(2026, 3, 16), datetime(2026, 3, 14), Grain.DAY) == []


def test_a_single_bucket_span_has_one_entry():
    assert span(datetime(2026, 3, 14, 1), datetime(2026, 3, 14, 5), Grain.DAY) == [
        datetime(2026, 3, 14)
    ]


def test_absurd_spans_are_refused_rather_than_hanging():
    with pytest.raises(ValueError, match="exceeds 100,000 buckets"):
        span(datetime(1000, 1, 1), datetime(3000, 1, 1), Grain.HOUR)


def test_the_refusal_says_how_to_fix_it():
    """A ceiling nobody can act on just moves the confusion one step later."""
    with pytest.raises(ValueError, match="coarser grain"):
        span(datetime(1000, 1, 1), datetime(3000, 1, 1), Grain.HOUR)


# -- window conversion ---------------------------------------------------------


def test_same_grain_conversion_is_identity():
    assert convert_window(-2, 3, Grain.DAY, Grain.DAY) == (-2, 3)


def test_refinement_is_refused():
    assert convert_window(0, 0, Grain.MONTH, Grain.DAY) is None


def test_coarsening_brackets_the_source_bucket():
    lo, hi = convert_window(0, 0, Grain.DAY, Grain.MONTH)
    assert lo <= 0 < hi
