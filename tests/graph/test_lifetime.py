"""Lifetime cost, and the difference between unmeasured and free."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.types import DatasetId
from fathom.graph.plan import lifetime
from fathom.graph.plan.cost import CostModel

BUSY = DatasetId("duckdb", "gold.busy")
QUIET = DatasetId("duckdb", "gold.quiet")
CHEAP = DatasetId("duckdb", "gold.cheap")
UNKNOWN = DatasetId("duckdb", "gold.unknown")

MODEL = CostModel(price_per_partition=1.0)
MARCH = datetime(2026, 3, 1, tzinfo=UTC)


def run(ds: DatasetId, *, day: int = 1, partitions: int = 10) -> lifetime.RunRecord:
    return lifetime.RunRecord(ds, MARCH + timedelta(days=day), partitions=partitions)


# -- accumulating --------------------------------------------------------------


def test_runs_total_into_one_lifetime_cost():
    totals = lifetime.accumulate([run(BUSY, day=1), run(BUSY, day=2)], MODEL)
    assert totals[BUSY].runs == 2
    assert totals[BUSY].spend == pytest.approx(20.0)
    assert totals[BUSY].partitions == 20


def test_first_and_last_run_bracket_the_history():
    totals = lifetime.accumulate([run(BUSY, day=5), run(BUSY, day=1)], MODEL)
    assert totals[BUSY].first_run == MARCH + timedelta(days=1)
    assert totals[BUSY].last_run == MARCH + timedelta(days=5)
    assert totals[BUSY].span == timedelta(days=4)


def test_a_dataset_with_no_runs_is_absent_rather_than_zero():
    """Inventing a zero would make an unmeasured table the cheapest in the warehouse."""
    totals = lifetime.accumulate([run(BUSY)], MODEL)
    assert UNKNOWN not in totals


def test_an_unmeasured_total_says_so():
    empty = lifetime.LifetimeCost(dataset=UNKNOWN)
    assert not empty.is_measured
    assert "not measured, which is not free" in empty.summary()


def test_per_run_divides_the_spend():
    totals = lifetime.accumulate([run(BUSY, day=1), run(BUSY, day=2)], MODEL)
    assert totals[BUSY].per_run == pytest.approx(10.0)


def test_per_run_of_an_unmeasured_dataset_is_none():
    assert lifetime.LifetimeCost(dataset=UNKNOWN).per_run is None


def test_the_summary_reports_the_span():
    totals = lifetime.accumulate([run(BUSY, day=1), run(BUSY, day=31)], MODEL)
    assert "over 30 day(s)" in totals[BUSY].summary()


def test_naive_and_aware_run_timestamps_mix():
    records = [run(BUSY, day=1), lifetime.RunRecord(BUSY, datetime(2026, 3, 9), partitions=1)]
    assert lifetime.accumulate(records, MODEL)[BUSY].runs == 2


# -- rates and rollups ---------------------------------------------------------


def test_burn_rate_projects_over_the_observed_span():
    totals = lifetime.accumulate([run(BUSY, day=1), run(BUSY, day=11)], MODEL)
    # 20 spent across 10 days -> 60 over 30 days
    assert lifetime.burn_rate(totals[BUSY]) == pytest.approx(60.0)


def test_burn_rate_of_a_single_instant_is_none():
    """A rate needs a span to divide by, and one point does not have one."""
    totals = lifetime.accumulate([run(BUSY, day=1)], MODEL)
    assert lifetime.burn_rate(totals[BUSY]) is None


def test_total_spend_sums_the_measured_part():
    totals = lifetime.accumulate([run(BUSY), run(QUIET)], MODEL)
    assert lifetime.total_spend(totals) == pytest.approx(20.0)


def test_most_expensive_lifetime_ranks_by_spend():
    totals = lifetime.accumulate([run(BUSY, partitions=100), run(QUIET, partitions=1)], MODEL)
    assert [t.dataset for t in lifetime.most_expensive_lifetime(totals)] == [BUSY, QUIET]


def test_most_expensive_lifetime_respects_the_limit():
    totals = lifetime.accumulate([run(BUSY), run(QUIET)], MODEL)
    assert len(lifetime.most_expensive_lifetime(totals, limit=1)) == 1


# -- cost against usage --------------------------------------------------------


def totals_fixture() -> dict:
    return lifetime.accumulate(
        [
            run(BUSY, partitions=100),
            run(QUIET, partitions=100),
            run(CHEAP, partitions=1),
        ],
        MODEL,
    )


def test_a_read_dataset_is_earning_whatever_it_costs():
    found = lifetime.value(totals_fixture(), {BUSY: 40}, threshold=50.0)
    verdict = next(f for f in found if f.dataset == BUSY)
    assert verdict.verdict is lifetime.Verdict.EARNING
    assert not verdict.is_actionable


def test_an_unread_expensive_dataset_is_worth_a_review():
    found = lifetime.value(totals_fixture(), {BUSY: 40}, threshold=50.0)
    verdict = next(f for f in found if f.dataset == QUIET)
    assert verdict.verdict is lifetime.Verdict.REVIEW
    assert verdict.is_actionable


def test_an_unread_cheap_dataset_is_not_worth_the_review():
    found = lifetime.value(totals_fixture(), {BUSY: 40}, threshold=50.0)
    verdict = next(f for f in found if f.dataset == CHEAP)
    assert verdict.verdict is lifetime.Verdict.CHEAP_AND_QUIET
    assert not verdict.is_actionable


def test_a_dataset_with_no_cost_history_is_unmeasured_not_judged():
    found = lifetime.value(totals_fixture(), {UNKNOWN: 0}, threshold=50.0)
    verdict = next(f for f in found if f.dataset == UNKNOWN)
    assert verdict.verdict is lifetime.Verdict.UNMEASURED
    assert verdict.spend is None
    assert "unmeasured, not free" in str(verdict)


def test_the_threshold_moves_the_verdict():
    cheap = next(
        f for f in lifetime.value(totals_fixture(), {}, threshold=0.5) if f.dataset == CHEAP
    )
    assert cheap.verdict is lifetime.Verdict.REVIEW


def test_actionable_findings_sort_most_expensive_first():
    totals = lifetime.accumulate([run(BUSY, partitions=100), run(QUIET, partitions=200)], MODEL)
    found = lifetime.actionable(lifetime.value(totals, {}, threshold=10.0))
    assert [f.dataset for f in found] == [QUIET, BUSY]


def test_actionable_comes_before_everything_else():
    found = lifetime.value(totals_fixture(), {BUSY: 5}, threshold=50.0)
    assert found[0].dataset == QUIET


def test_unmeasured_lists_the_gap_to_close():
    found = lifetime.value(totals_fixture(), {UNKNOWN: 0}, threshold=50.0)
    assert lifetime.unmeasured(found) == [UNKNOWN]


# -- the review list -----------------------------------------------------------


def test_the_summary_states_the_blind_spot_once():
    """Cost is measured; usage is observed. The output must say which is which."""
    text = lifetime.summarize(lifetime.value(totals_fixture(), {BUSY: 5}, threshold=50.0))
    assert "unread and above the threshold" in text
    assert "read once a year for a" in text


def test_the_window_appears_in_the_finding_when_given():
    found = lifetime.value(totals_fixture(), {}, threshold=50.0, window=timedelta(days=90))
    assert "in 90 day(s)" in str(next(f for f in found if f.dataset == QUIET))


def test_a_clean_summary_says_nothing_qualified():
    text = lifetime.summarize(
        lifetime.value(totals_fixture(), {QUIET: 1, CHEAP: 1, BUSY: 1}, threshold=50.0)
    )
    assert "no dataset is both unread and above the threshold" in text


def test_the_summary_reports_unmeasured_datasets_separately():
    text = lifetime.summarize(lifetime.value(totals_fixture(), {UNKNOWN: 0}, threshold=50.0))
    assert "have no cost history and were not judged" in text
