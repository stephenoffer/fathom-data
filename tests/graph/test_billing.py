"""Reconciling a cost model against the actual invoice.

Every savings figure rests on declared rates that nothing checks. These tests care
about the three refusals: no comparison against a zero bill, no calibration from too
few periods, and no per-dataset figure without attribution to support it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.types import DatasetId
from fathom.graph.plan import billing
from fathom.graph.plan.cost import CostModel

GOLD = DatasetId("duckdb", "gold.monthly")
RAW = DatasetId("duckdb", "raw.events")

MARCH = datetime(2026, 3, 1, tzinfo=UTC)
MODEL = CostModel(price_per_partition=1.0, price_per_tb_scanned=5.0)


def rows(n: int, *, amount: float = 100.0, dataset: DatasetId | None = None):
    return [
        billing.BillingRecord(MARCH + timedelta(days=d), amount, source="wh_main", dataset=dataset)
        for d in range(n)
    ]


# -- the records ---------------------------------------------------------------


def test_total_billed_sums_the_rows():
    assert billing.total_billed(rows(3)) == pytest.approx(300.0)


def test_an_unattributed_row_is_the_common_case():
    (record,) = rows(1)
    assert not record.is_attributed
    assert "(unattributed)" in str(record)


def test_an_attributed_row_names_its_dataset():
    (record,) = rows(1, dataset=GOLD)
    assert record.is_attributed
    assert "gold.monthly" in str(record)


def test_attributed_share_is_the_assignable_fraction():
    records = [*rows(1, amount=75.0, dataset=GOLD), *rows(1, amount=25.0)]
    assert billing.attributed_share(records) == pytest.approx(0.75)


def test_attributed_share_of_a_zero_bill_is_zero():
    assert billing.attributed_share(rows(2, amount=0.0)) == 0.0


def test_unattributed_returns_only_the_unassignable():
    records = [*rows(1, dataset=GOLD), *rows(1)]
    assert len(billing.unattributed(records)) == 1


# -- bias ----------------------------------------------------------------------


def test_a_correct_model_has_no_bias():
    assert billing.bias(100.0, 100.0) == pytest.approx(0.0)


def test_over_prediction_is_positive():
    assert billing.bias(150.0, 100.0) == pytest.approx(0.5)


def test_under_prediction_is_negative():
    assert billing.bias(50.0, 100.0) == pytest.approx(-0.5)


def test_bias_against_a_zero_bill_is_no_answer_not_a_large_error():
    assert billing.bias(100.0, 0.0) is None


# -- reconciliation ------------------------------------------------------------


def test_reconcile_compares_the_totals():
    result = billing.reconcile(1200.0, rows(14))
    assert result.billed == pytest.approx(1400.0)
    assert result.periods == 14
    assert result.bias == pytest.approx((1200.0 - 1400.0) / 1400.0)


def test_the_correction_is_what_to_multiply_by():
    result = billing.reconcile(1000.0, rows(14, amount=200.0))
    assert result.correction == pytest.approx(2.8)


def test_a_model_that_priced_nothing_has_no_correction():
    """Infinitely wrong and priced nothing are different readings."""
    assert billing.reconcile(0.0, rows(14)).correction is None


def test_a_zero_bill_makes_no_comparison():
    result = billing.reconcile(500.0, rows(3, amount=0.0))
    assert result.bias is None
    assert "no comparison to make" in result.summary()


def test_the_window_brackets_the_records():
    result = billing.reconcile(100.0, rows(5))
    assert result.window == (MARCH, MARCH + timedelta(days=4))


def test_a_short_window_is_reported_unreliable():
    result = billing.reconcile(100.0, rows(3))
    assert not result.is_reliable
    assert "rounding artifact" in result.summary()


def test_a_long_enough_window_is_reliable():
    assert billing.reconcile(1400.0, rows(14)).is_reliable


def test_the_summary_names_the_direction():
    assert "over-predicts by 50%" in billing.reconcile(2100.0, rows(14)).summary()
    assert "under-predicts by 50%" in billing.reconcile(700.0, rows(14)).summary()


def test_the_attribution_caveat_reaches_the_summary():
    result = billing.reconcile(1400.0, rows(14))
    assert "0% of the bill carries a dataset" in result.summary()
    assert "evidence for themselves" in result.summary()


def test_a_fully_attributed_bill_drops_the_caveat():
    result = billing.reconcile(1400.0, rows(14, dataset=GOLD))
    assert "carries a dataset" not in result.summary()


# -- per dataset, only where attribution supports it ---------------------------


def test_per_dataset_appears_when_the_bill_attributes():
    records = rows(14, amount=100.0, dataset=GOLD)
    result = billing.reconcile(1000.0, records, per_dataset_modelled={GOLD: 1000.0})
    assert result.per_dataset[GOLD] == (1000.0, pytest.approx(1400.0))


def test_a_dataset_the_bill_never_named_is_absent_not_zeroed():
    """Showing it against zero would invent a 100% error out of missing attribution."""
    records = rows(14, dataset=GOLD)
    result = billing.reconcile(2000.0, records, per_dataset_modelled={GOLD: 1000.0, RAW: 1000.0})
    assert GOLD in result.per_dataset
    assert RAW not in result.per_dataset


def test_no_per_dataset_figures_without_a_modelled_breakdown():
    assert billing.reconcile(1400.0, rows(14, dataset=GOLD)).per_dataset == {}


def test_drifted_datasets_ranks_the_worst_first():
    records = [
        *rows(7, amount=100.0, dataset=GOLD),
        *rows(7, amount=100.0, dataset=RAW),
    ]
    result = billing.reconcile(1400.0, records, per_dataset_modelled={GOLD: 2100.0, RAW: 750.0})
    drifted = billing.drifted_datasets(result)
    assert drifted[0][0] == GOLD  # +200% beats -~-7%


def test_a_dataset_inside_tolerance_is_not_drifted():
    records = rows(14, amount=100.0, dataset=GOLD)
    result = billing.reconcile(1400.0, records, per_dataset_modelled={GOLD: 1450.0})
    assert billing.drifted_datasets(result) == []


# -- calibration ---------------------------------------------------------------


def test_calibration_scales_every_rate():
    result = billing.reconcile(700.0, rows(14, amount=100.0))
    corrected = billing.calibrate(MODEL, result)
    assert corrected is not None
    assert corrected.price_per_partition == pytest.approx(2.0)
    assert corrected.price_per_tb_scanned == pytest.approx(10.0)


def test_calibration_leaves_the_physical_constants_alone():
    """The bill says nothing about grid intensity."""
    result = billing.reconcile(700.0, rows(14, amount=100.0))
    corrected = billing.calibrate(MODEL, result)
    assert corrected is not None
    assert corrected.grid_intensity == MODEL.grid_intensity
    assert corrected.kwh_per_tb_scanned == MODEL.kwh_per_tb_scanned


def test_too_few_periods_refuses_to_calibrate():
    assert billing.calibrate(MODEL, billing.reconcile(700.0, rows(3, amount=100.0))) is None


def test_a_model_already_within_tolerance_is_left_alone():
    result = billing.reconcile(1450.0, rows(14, amount=100.0))
    assert result.within()
    assert billing.calibrate(MODEL, result) is None


def test_a_model_that_priced_nothing_cannot_be_calibrated():
    assert billing.calibrate(MODEL, billing.reconcile(0.0, rows(14))) is None


def test_the_tolerance_is_the_callers_choice():
    result = billing.reconcile(1200.0, rows(14, amount=100.0))
    assert billing.calibrate(MODEL, result, tolerance=0.5) is None
    assert billing.calibrate(MODEL, result, tolerance=0.01) is not None


def test_within_tolerance_matches_the_report():
    result = billing.reconcile(1450.0, rows(14, amount=100.0))
    assert billing.within_tolerance(result)


# -- slicing -------------------------------------------------------------------


def test_in_window_compares_like_with_like():
    """A month of modelled cost against six weeks of billing is a 50% window artifact."""
    records = rows(40)
    inside = billing.in_window(records, start=MARCH, end=MARCH + timedelta(days=29))
    assert len(inside) == 30


def test_by_period_totals_each_day():
    records = [*rows(1, amount=50.0), *rows(1, amount=70.0)]
    assert billing.by_period(records)[MARCH] == pytest.approx(120.0)


def test_by_source_ranks_the_biggest_spender():
    records = [
        billing.BillingRecord(MARCH, 300.0, source="wh_big"),
        billing.BillingRecord(MARCH, 100.0, source="wh_small"),
    ]
    assert list(billing.by_source(records)) == ["wh_big", "wh_small"]


def test_an_unnamed_source_is_labelled_not_dropped():
    assert "(unnamed)" in billing.by_source([billing.BillingRecord(MARCH, 10.0)])


def test_coverage_days_spans_the_records():
    assert billing.coverage_days(rows(5)) == timedelta(days=4)


def test_coverage_of_no_records_is_none():
    assert billing.coverage_days([]) is None
