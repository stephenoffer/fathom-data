"""Baselines bucketed by a cycle, and the buckets deliberately left unmodelled."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fathom.core.types import DatasetId
from fathom.observe import seasonal
from fathom.observe.profile import ColumnProfile, Profile, Severity

EVENTS = DatasetId("duckdb", "raw.events")

# A Monday, so weekday() lines up with the offsets below.
MONDAY = datetime(2026, 3, 2)


def profile(rows: int, *, nulls: int = 0) -> Profile:
    return Profile(
        dataset=EVENTS,
        row_count=rows,
        columns=(
            ColumnProfile("amount", "double", row_count=rows, null_count=nulls, min=0, max=10),
        ),
    )


def weekly(weeks: int, *, weekday_rows: int, weekend_rows: int) -> list[seasonal.Observation]:
    """`weeks` weeks of daily observations with a strong weekday/weekend split."""
    out: list[seasonal.Observation] = []
    for week in range(weeks):
        for offset in range(7):
            when = MONDAY + timedelta(days=week * 7 + offset)
            rows = weekend_rows if offset >= 5 else weekday_rows
            out.append(seasonal.Observation(when, profile(rows + offset)))
    return out


# -- bucketing -----------------------------------------------------------------


def test_day_of_week_buckets_monday_to_zero():
    assert seasonal.bucket_of(MONDAY, seasonal.Cycle.DAY_OF_WEEK) == 0
    assert seasonal.bucket_of(MONDAY + timedelta(days=6), seasonal.Cycle.DAY_OF_WEEK) == 6


def test_the_other_cycles_bucket_on_their_own_field():
    when = datetime(2026, 7, 14, 9)
    assert seasonal.bucket_of(when, seasonal.Cycle.HOUR_OF_DAY) == 9
    assert seasonal.bucket_of(when, seasonal.Cycle.DAY_OF_MONTH) == 14
    assert seasonal.bucket_of(when, seasonal.Cycle.MONTH_OF_YEAR) == 7


# -- learning ------------------------------------------------------------------


def test_learning_from_no_history_refuses():
    with pytest.raises(ValueError, match="no observations"):
        seasonal.learn_seasonal([])


def test_each_weekday_gets_its_own_band():
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    monday = baseline.band(None, "row_count", 0)
    saturday = baseline.band(None, "row_count", 5)
    assert monday is not None and saturday is not None
    assert monday.low > saturday.high


def test_a_bucket_below_the_minimum_is_recorded_not_modelled():
    """Two Sundays is not a baseline for Sunday."""
    history = weekly(2, weekday_rows=1000, weekend_rows=200)
    baseline = seasonal.learn_seasonal(history, min_observations=4)
    assert baseline.bands == {}
    assert set(baseline.unmodelled) == set(range(7))
    assert all(n == 2 for n in baseline.unmodelled.values())


def test_an_unmodelled_baseline_says_so_rather_than_looking_empty():
    baseline = seasonal.learn_seasonal(weekly(1, weekday_rows=10, weekend_rows=5))
    assert not baseline.is_usable
    assert "none reaching the minimum" in baseline.summary()


def test_the_summary_names_the_buckets_it_will_not_check():
    history = weekly(4, weekday_rows=1000, weekend_rows=200)
    # Drop most Wednesdays so that bucket alone falls under the minimum.
    thinned = [o for o in history if o.when.weekday() != 2 or o.when.day < 10]
    baseline = seasonal.learn_seasonal(thinned)
    assert 2 in baseline.unmodelled
    assert "Wed" in baseline.summary()
    assert "not checked" in baseline.summary()


def test_bounds_widen_past_what_was_observed():
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    band = baseline.band(None, "row_count", 0)
    assert band is not None
    assert band.low < 1000 and band.high > 1000


def test_column_metrics_are_banded_too():
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    assert baseline.band("amount", "null_rate", 0) is not None
    assert baseline.band("amount", "max", 0) is not None


def test_unmodelled_buckets_reports_labels():
    baseline = seasonal.learn_seasonal(weekly(2, weekday_rows=10, weekend_rows=5))
    assert "Mon" in seasonal.unmodelled_buckets(baseline)


def test_bands_for_returns_one_buckets_bands():
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    bands = seasonal.bands_for(baseline, 0)
    assert bands and all(b.bucket == 0 for b in bands)


# -- checking ------------------------------------------------------------------


def test_a_normal_saturday_passes_against_the_saturday_band():
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    saturday = MONDAY + timedelta(days=5 * 7 + 5)
    assert seasonal.check_seasonal(seasonal.Observation(saturday, profile(205)), baseline) == []


def test_a_weekday_volume_on_a_saturday_is_flagged():
    """The whole point: 1000 rows is normal on Monday and an anomaly on Saturday."""
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    saturday = MONDAY + timedelta(days=5 * 7 + 5)
    findings = seasonal.check_seasonal(seasonal.Observation(saturday, profile(1000)), baseline)
    assert any(f.kind == "seasonal_row_count" for f in findings)
    assert "above" in next(f for f in findings if f.kind == "seasonal_row_count").detail


def test_the_same_volume_on_a_monday_is_not_flagged():
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    monday = MONDAY + timedelta(days=5 * 7)
    findings = seasonal.check_seasonal(seasonal.Observation(monday, profile(1000)), baseline)
    assert [f for f in findings if f.kind == "seasonal_row_count"] == []


def test_an_unmodelled_bucket_produces_no_findings():
    """Silence means 'not modelled', and the baseline says which buckets those are."""
    history = weekly(2, weekday_rows=1000, weekend_rows=200)
    baseline = seasonal.learn_seasonal(history)
    absurd = seasonal.Observation(MONDAY + timedelta(days=14), profile(10**9))
    assert seasonal.check_seasonal(absurd, baseline) == []


def test_a_finding_carries_the_bound_it_crossed():
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    saturday = MONDAY + timedelta(days=5 * 7 + 5)
    finding = next(
        f
        for f in seasonal.check_seasonal(seasonal.Observation(saturday, profile(1000)), baseline)
        if f.kind == "seasonal_row_count"
    )
    assert finding.after == 1000
    assert finding.before is not None and finding.before < 1000


def test_severity_is_the_callers_choice():
    baseline = seasonal.learn_seasonal(weekly(5, weekday_rows=1000, weekend_rows=200))
    saturday = MONDAY + timedelta(days=5 * 7 + 5)
    findings = seasonal.check_seasonal(
        seasonal.Observation(saturday, profile(1000)), baseline, severity=Severity.ERROR
    )
    assert all(f.severity is Severity.ERROR for f in findings)


# -- is seasonality even there -------------------------------------------------


def test_strong_weekly_variation_scores_high():
    assert seasonal.strength(weekly(5, weekday_rows=1000, weekend_rows=100)) > 0.7


def test_flat_data_scores_low():
    history = [
        seasonal.Observation(MONDAY + timedelta(days=d), profile(1000 + (d % 3))) for d in range(35)
    ]
    score = seasonal.strength(history)
    assert score is not None and score < 0.5


def test_a_constant_metric_has_no_answer_rather_than_zero():
    """No spread is not 'no seasonality'; it is no answer, and the caller must tell."""
    history = [seasonal.Observation(MONDAY + timedelta(days=d), profile(1000)) for d in range(35)]
    assert seasonal.strength(history) is None


def test_too_few_buckets_has_no_answer():
    history = [seasonal.Observation(MONDAY, profile(10 + d)) for d in range(5)]
    assert seasonal.strength(history) is None


# -- zipping helper ------------------------------------------------------------


def test_observations_from_pairs_profiles_with_moments():
    made = seasonal.observations_from([profile(1), profile(2)], when=[MONDAY, MONDAY])
    assert [o.profile.row_count for o in made] == [1, 2]


def test_mismatched_lengths_refuse_rather_than_truncating():
    with pytest.raises(ValueError, match="one to one"):
        seasonal.observations_from([profile(1), profile(2)], when=[MONDAY])
