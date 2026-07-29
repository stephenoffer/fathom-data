"""The partition that never arrived, and the one that arrived twice."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.grains import Grain
from fathom.core.types import ANY, DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.observe import completeness

RAW = DatasetId("duckdb", "raw.events")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
DAY_REGION = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
HOUR = PartitionSpec.of(PartitionField.time("ts", Grain.HOUR))


def days(*numbers: int) -> list[KeyPredicate]:
    return [KeyPredicate.of(dt=datetime(2026, 3, n)) for n in numbers]


# -- expected sets -------------------------------------------------------------


def test_expected_keys_enumerates_the_time_grain():
    keys = completeness.expected_keys(DAY, start=datetime(2026, 3, 1), end=datetime(2026, 3, 5))
    assert keys == days(1, 2, 3, 4, 5)


def test_expected_keys_crosses_declared_value_domains():
    keys = completeness.expected_keys(
        DAY_REGION,
        start=datetime(2026, 3, 1),
        end=datetime(2026, 3, 2),
        domains={"region": ["eu", "us"]},
    )
    assert len(keys) == 4
    assert KeyPredicate.of(dt=datetime(2026, 3, 1), region="eu") in keys


def test_an_undeclared_value_field_collapses_to_any_rather_than_guessing():
    keys = completeness.expected_keys(
        DAY_REGION, start=datetime(2026, 3, 1), end=datetime(2026, 3, 1)
    )
    assert len(keys) == 1
    assert keys[0].get("region") is ANY


def test_a_reversed_range_is_empty_not_an_error():
    assert (
        completeness.expected_keys(DAY, start=datetime(2026, 3, 5), end=datetime(2026, 3, 1)) == []
    )


def test_an_oversized_expected_set_refuses_rather_than_truncating():
    """A shortened expected set reports an incomplete dataset as complete."""
    with pytest.raises(ValueError, match="exceeds max_keys"):
        completeness.expected_keys(
            HOUR, start=datetime(2020, 1, 1), end=datetime(2026, 1, 1), max_keys=1000
        )


# -- missing partitions --------------------------------------------------------


def test_missing_finds_the_absent_day():
    absent = completeness.missing(days(1, 2, 3), days(1, 3))
    assert absent == days(2)


def test_an_unreadable_any_partition_does_not_absorb_every_expected_key():
    """Subsumption here would report a dataset with one unreadable partition complete."""
    present = [KeyPredicate.of(dt=ANY)]
    assert completeness.missing(days(1, 2, 3), present) == days(1, 2, 3)


def test_unexpected_finds_partitions_outside_the_spec_range():
    surplus = completeness.unexpected(days(1, 2), days(1, 2, 9))
    assert surplus == days(9)


# -- gaps as runs --------------------------------------------------------------


def test_consecutive_missing_days_collapse_into_one_run():
    runs = completeness.gaps(RAW, DAY, days(4, 5, 6))
    assert len(runs) == 1
    assert (runs[0].start, runs[0].end, runs[0].count) == (
        datetime(2026, 3, 4),
        datetime(2026, 3, 6),
        3,
    )


def test_a_split_gap_is_two_runs():
    runs = completeness.gaps(RAW, DAY, days(2, 5, 6))
    assert [r.count for r in runs] == [1, 3 - 1]
    assert runs[0].is_single


def test_runs_are_computed_within_each_value_slice_separately():
    """`region=eu` missing three days and `region=us` missing one are two incidents."""
    absent = [KeyPredicate.of(dt=datetime(2026, 3, d), region="eu") for d in (4, 5, 6)] + [
        KeyPredicate.of(dt=datetime(2026, 3, 5), region="us")
    ]
    runs = completeness.gaps(RAW, DAY_REGION, absent)
    assert len(runs) == 2
    by_region = {dict(r.within)["region"]: r.count for r in runs}
    assert by_region == {"eu": 3, "us": 1}


def test_a_spec_with_no_time_field_reports_no_runs():
    spec = PartitionSpec.of(PartitionField.value("region"))
    assert completeness.gaps(RAW, spec, [KeyPredicate.of(region="eu")]) == []


# -- the report ----------------------------------------------------------------


def test_a_complete_dataset_says_so():
    result = completeness.report(
        RAW, DAY, days(1, 2, 3), start=datetime(2026, 3, 1), end=datetime(2026, 3, 3)
    )
    assert result.is_complete
    assert result.ratio == 1.0
    assert "complete" in result.summary()


def test_an_incomplete_dataset_reports_the_run_and_the_ratio():
    result = completeness.report(
        RAW, DAY, days(1, 3), start=datetime(2026, 3, 1), end=datetime(2026, 3, 3)
    )
    assert not result.is_complete
    assert result.ratio == pytest.approx(2 / 3)
    assert len(result.runs) == 1
    assert "incomplete" in result.summary()


def test_the_report_states_which_domains_it_inferred():
    """An inferred domain has a blind spot, so it is never presented as a known one."""
    present = [KeyPredicate.of(dt=datetime(2026, 3, 1), region=r) for r in ("eu", "us")]
    result = completeness.report(
        RAW, DAY_REGION, present, start=datetime(2026, 3, 1), end=datetime(2026, 3, 2)
    )
    assert result.assumed_domains == {"region": ["eu", "us"]}
    assert "inferred" in result.summary()
    assert len(result.absent) == 2  # both regions missing on the 2nd


def test_a_declared_domain_is_not_reported_as_assumed():
    present = [KeyPredicate.of(dt=datetime(2026, 3, 1), region="eu")]
    result = completeness.report(
        RAW,
        DAY_REGION,
        present,
        start=datetime(2026, 3, 1),
        end=datetime(2026, 3, 1),
        domains={"region": ["eu", "us"]},
    )
    assert result.assumed_domains == {}
    assert len(result.absent) == 1  # us, which was never observed but was declared


def test_longest_gap_picks_the_run_to_triage():
    result = completeness.report(
        RAW, DAY, days(1, 8), start=datetime(2026, 3, 1), end=datetime(2026, 3, 8)
    )
    worst = completeness.longest_gap(result)
    assert worst is not None and worst.count == 6


def test_longest_gap_of_a_complete_dataset_is_none():
    result = completeness.report(
        RAW, DAY, days(1), start=datetime(2026, 3, 1), end=datetime(2026, 3, 1)
    )
    assert completeness.longest_gap(result) is None


def test_coverage_ratio_of_an_empty_range_is_one():
    result = completeness.report(RAW, DAY, [], start=datetime(2026, 3, 5), end=datetime(2026, 3, 1))
    assert completeness.coverage_ratio(result) == 1.0


# -- arrivals ------------------------------------------------------------------


def arrival(day: int, observed: datetime, digest: str = "", dataset: DatasetId = RAW):
    return completeness.Arrival(
        dataset, KeyPredicate.of(dt=datetime(2026, 3, day)), observed, digest
    )


def test_arrival_lag_is_measured_from_when_the_bucket_closed():
    """A day partition written at 02:00 the next morning is two hours late, not 26."""
    landed = arrival(14, datetime(2026, 3, 15, 2))
    lag = completeness.arrival_lag(landed, field_name="dt", grain=Grain.DAY)
    assert lag == timedelta(hours=2)


def test_an_on_time_arrival_has_a_non_positive_lag():
    landed = arrival(14, datetime(2026, 3, 14, 23))
    lag = completeness.arrival_lag(landed, field_name="dt", grain=Grain.DAY)
    assert lag is not None and lag < timedelta(0)


def test_arrival_lag_of_a_non_time_field_is_none():
    landed = completeness.Arrival(RAW, KeyPredicate.of(region="eu"), datetime(2026, 3, 15))
    assert completeness.arrival_lag(landed, field_name="region", grain=Grain.DAY) is None


def test_late_arrivals_are_ranked_worst_first():
    late = completeness.late_arrivals(
        [arrival(14, datetime(2026, 3, 15, 3)), arrival(13, datetime(2026, 3, 16, 6))],
        field_name="dt",
        grain=Grain.DAY,
        tolerance=timedelta(hours=1),
    )
    assert [a.key.get("dt").day for a, _ in late] == [13, 14]


def test_an_arrival_within_tolerance_is_not_late():
    late = completeness.late_arrivals(
        [arrival(14, datetime(2026, 3, 15, 1))],
        field_name="dt",
        grain=Grain.DAY,
        tolerance=timedelta(hours=4),
    )
    assert late == []


def test_an_aware_observation_compares_against_a_naive_bucket():
    """Half the timestamps in this system are naive; comparing them must not raise."""
    landed = arrival(14, datetime(2026, 3, 15, 2, tzinfo=UTC))
    assert completeness.arrival_lag(landed, field_name="dt", grain=Grain.DAY) == timedelta(hours=2)


def test_a_single_arrival_is_not_a_duplicate():
    assert completeness.duplicate_arrivals([arrival(14, datetime(2026, 3, 15))]) == []


def test_identical_digests_are_a_replay_not_a_restatement():
    group = [arrival(14, datetime(2026, 3, 15), "abc"), arrival(14, datetime(2026, 3, 16), "abc")]
    assert len(completeness.replays(group)) == 1
    assert completeness.restatements(group) == []


def test_differing_digests_are_a_restatement():
    group = [arrival(14, datetime(2026, 3, 15), "abc"), arrival(14, datetime(2026, 3, 16), "def")]
    assert completeness.replays(group) == []
    assert len(completeness.restatements(group)) == 1


def test_an_unknown_digest_is_a_restatement_because_it_cannot_be_proven_otherwise():
    """Treating 'cannot tell' as 'unchanged' is what lets a restatement through."""
    group = [arrival(14, datetime(2026, 3, 15), "abc"), arrival(14, datetime(2026, 3, 16), "")]
    assert completeness.replays(group) == []
    assert len(completeness.restatements(group)) == 1


def test_duplicates_are_grouped_per_dataset_not_just_per_key():
    other = DatasetId("duckdb", "silver.events")
    group = [
        arrival(14, datetime(2026, 3, 15), "a"),
        arrival(14, datetime(2026, 3, 15), "a", other),
    ]
    assert completeness.duplicate_arrivals(group) == []


def test_arrivals_in_a_group_are_ordered_by_observation_time():
    group = [arrival(14, datetime(2026, 3, 16), "a"), arrival(14, datetime(2026, 3, 15), "b")]
    (found,) = completeness.restatements(group)
    assert [a.observed.day for a in found] == [15, 16]
