"""Properties of completeness, because the prose invariants deserve a machine check.

A completeness check that is *usually* right reports a dataset as complete some of
the time, and a false "complete" is indistinguishable from a real one. These are the
four claims the module's docstring makes, stated as properties:

1. Expected partitions the caller has are exactly present plus missing.
2. Collapsing missing keys into runs neither invents nor drops a bucket.
3. A run is contiguous — every bucket between its ends is also missing.
4. Runs never straddle two value slices.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from fathom.core.grains import Grain, span
from fathom.core.types import DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.observe import completeness

RAW = DatasetId("duckdb", "raw.events")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
DAY_REGION = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))

START = datetime(2026, 3, 1)


def days(offsets: list[int]) -> list[KeyPredicate]:
    return [KeyPredicate.of(dt=START + timedelta(days=n)) for n in sorted(set(offsets))]


# Offsets inside a bounded month, so the expected set stays small and the shrinker
# produces readable counterexamples.
offsets = st.lists(st.integers(min_value=0, max_value=27), max_size=28)


@given(present=offsets)
@settings(max_examples=200)
def test_present_and_missing_partition_the_expected_set(present):
    """Nothing is both there and absent, and nothing is neither."""
    expected = completeness.expected_keys(DAY, start=START, end=START + timedelta(days=27))
    have = days(present)
    absent = completeness.missing(expected, have)

    assert set(absent).isdisjoint(have)
    assert set(absent) | (set(expected) & set(have)) == set(expected)
    assert len(absent) + len(set(expected) & set(have)) == len(expected)


@given(present=offsets)
@settings(max_examples=200)
def test_runs_account_for_every_missing_bucket_exactly_once(present):
    """Collapsing into runs must neither invent nor drop a bucket."""
    expected = completeness.expected_keys(DAY, start=START, end=START + timedelta(days=27))
    absent = completeness.missing(expected, days(present))
    runs = completeness.gaps(RAW, DAY, absent)

    assert sum(run.count for run in runs) == len(absent)

    covered: set[datetime] = set()
    for run in runs:
        covered.update(span(run.start, run.end, Grain.DAY))
    assert covered == {k.get("dt") for k in absent}


@given(present=offsets)
@settings(max_examples=200)
def test_every_run_is_contiguous(present):
    """A run claims its ends are adjacent; every bucket between them must be missing."""
    expected = completeness.expected_keys(DAY, start=START, end=START + timedelta(days=27))
    absent = completeness.missing(expected, days(present))
    inside = {k.get("dt") for k in absent}

    for run in completeness.gaps(RAW, DAY, absent):
        buckets = span(run.start, run.end, Grain.DAY)
        assert len(buckets) == run.count
        assert all(bucket in inside for bucket in buckets)


@given(present=offsets)
@settings(max_examples=200)
def test_runs_are_ordered_and_do_not_overlap(present):
    expected = completeness.expected_keys(DAY, start=START, end=START + timedelta(days=27))
    absent = completeness.missing(expected, days(present))
    runs = completeness.gaps(RAW, DAY, absent)

    assert [r.start for r in runs] == sorted(r.start for r in runs)
    for earlier, later in zip(runs, runs[1:], strict=False):
        assert earlier.end < later.start


@given(
    eu=st.lists(st.integers(min_value=0, max_value=9), max_size=10),
    us=st.lists(st.integers(min_value=0, max_value=9), max_size=10),
)
@settings(max_examples=150)
def test_a_run_never_straddles_two_value_slices(eu, us):
    """Two regions missing different days are separate incidents with separate causes."""
    absent = [
        KeyPredicate.of(dt=START + timedelta(days=n), region=region)
        for region, offsets_for in (("eu", eu), ("us", us))
        for n in sorted(set(offsets_for))
    ]
    runs = completeness.gaps(RAW, DAY_REGION, absent)

    for run in runs:
        assert len(run.within) == 1
        region = dict(run.within)["region"]
        source = eu if region == "eu" else us
        for bucket in span(run.start, run.end, Grain.DAY):
            assert (bucket - START).days in set(source)


@given(present=offsets)
@settings(max_examples=200)
def test_the_ratio_matches_the_counts(present):
    result = completeness.report(
        RAW, DAY, days(present), start=START, end=START + timedelta(days=27)
    )
    assert result.expected == 28
    assert result.ratio == (result.expected - len(result.absent)) / result.expected
    assert result.is_complete == (len(result.absent) == 0)


@given(present=offsets)
@settings(max_examples=100)
def test_a_report_never_claims_complete_while_holding_missing_keys(present):
    """The one failure mode this module exists to prevent."""
    result = completeness.report(
        RAW, DAY, days(present), start=START, end=START + timedelta(days=27)
    )
    if result.is_complete:
        assert result.absent == []
        assert "complete" in result.summary()
    else:
        assert result.absent
        assert "incomplete" in result.summary()


@given(stamps=st.lists(st.integers(min_value=0, max_value=48), min_size=1, max_size=6, unique=True))
@settings(max_examples=150)
def test_replays_and_restatements_partition_the_duplicates(stamps):
    """Every repeated arrival is exactly one of the two, never both or neither."""
    arrivals = [
        completeness.Arrival(
            RAW,
            KeyPredicate.of(dt=START),
            START + timedelta(hours=h),
            digest="same" if h % 2 == 0 else "",
        )
        for h in stamps
    ]
    duplicates = completeness.duplicate_arrivals(arrivals)
    replayed = completeness.replays(arrivals)
    restated = completeness.restatements(arrivals)

    assert len(replayed) + len(restated) == len(duplicates)
    keys = {id(g) for g in replayed} & {id(g) for g in restated}
    assert not keys


@given(hours=st.integers(min_value=-48, max_value=200))
@settings(max_examples=200)
def test_arrival_lag_is_monotone_in_observation_time(hours):
    """Landing later is never reported as landing earlier."""
    base = completeness.Arrival(RAW, KeyPredicate.of(dt=START), START + timedelta(hours=hours))
    later = completeness.Arrival(RAW, KeyPredicate.of(dt=START), START + timedelta(hours=hours + 1))
    first = completeness.arrival_lag(base, field_name="dt", grain=Grain.DAY)
    second = completeness.arrival_lag(later, field_name="dt", grain=Grain.DAY)
    assert first is not None and second is not None
    assert second > first
