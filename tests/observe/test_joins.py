"""Join keys: the fan-out that doubles revenue, and the orphans that hide a loss."""

from __future__ import annotations

import pytest

from fathom.core.types import DatasetId
from fathom.observe import joins
from fathom.observe.profile import ColumnProfile, Profile, Severity

ORDERS = DatasetId("duckdb", "silver.orders")
CUSTOMERS = DatasetId("duckdb", "silver.customers")
JOINED = DatasetId("duckdb", "gold.orders_enriched")


def profile(dataset: DatasetId, rows: int, **columns: int | None) -> Profile:
    return Profile(
        dataset=dataset,
        row_count=rows,
        columns=tuple(
            ColumnProfile(name, "string", row_count=rows, distinct_estimate=distinct)
            for name, distinct in columns.items()
        ),
    )


# -- key shape -----------------------------------------------------------------


def test_a_unique_key_has_fan_out_of_one():
    shape = joins.shape_of(profile(ORDERS, 1000, customer_id=1000), "customer_id")
    assert shape is not None
    assert shape.fan_out == pytest.approx(1.0)
    assert shape.is_unique


def test_a_duplicated_key_has_fan_out_above_one():
    shape = joins.shape_of(profile(ORDERS, 1000, customer_id=250), "customer_id")
    assert shape is not None
    assert shape.fan_out == pytest.approx(4.0)
    assert not shape.is_unique


def test_a_sketch_rounding_artifact_still_counts_as_unique():
    """A hair over 1.0 on a genuinely unique key is the estimator, not a duplicate."""
    assert joins.is_unique_key(profile(ORDERS, 1000, k=995), "k")


def test_an_unprofiled_column_has_no_shape():
    """'Not profiled' and 'no distinct values' must not be the same answer."""
    assert joins.shape_of(profile(ORDERS, 1000, k=None), "k") is None
    assert joins.fan_out(profile(ORDERS, 1000, k=None), "k") is None


def test_an_absent_column_has_no_shape():
    assert joins.shape_of(profile(ORDERS, 1000, k=10), "missing") is None


def test_an_empty_dataset_has_no_shape():
    assert joins.shape_of(profile(ORDERS, 0, k=0), "k") is None


def test_the_shape_reads_as_prose():
    shape = joins.shape_of(profile(ORDERS, 1000, k=250), "k")
    assert "250 distinct over 1000 rows" in str(shape)
    assert "fan-out 4.00" in str(shape)


# -- uniqueness lost -----------------------------------------------------------


def test_a_key_that_stops_being_unique_is_detected():
    """The single most common cause of a revenue total silently doubling."""
    before = profile(CUSTOMERS, 1000, customer_id=1000)
    after = profile(CUSTOMERS, 2000, customer_id=1000)
    assert joins.uniqueness_lost(before, after, "customer_id")


def test_a_key_that_stays_unique_is_not_flagged():
    before = profile(CUSTOMERS, 1000, customer_id=1000)
    after = profile(CUSTOMERS, 1500, customer_id=1500)
    assert not joins.uniqueness_lost(before, after, "customer_id")


def test_a_key_that_was_never_unique_is_not_a_loss():
    before = profile(ORDERS, 1000, customer_id=250)
    after = profile(ORDERS, 2000, customer_id=250)
    assert not joins.uniqueness_lost(before, after, "customer_id")


def test_an_unmeasurable_key_is_not_reported_as_lost():
    before = profile(CUSTOMERS, 1000, customer_id=1000)
    after = profile(CUSTOMERS, 2000, customer_id=None)
    assert not joins.uniqueness_lost(before, after, "customer_id")


# -- shape drift ---------------------------------------------------------------


def test_a_doubled_fan_out_is_an_error():
    before = profile(CUSTOMERS, 1000, customer_id=1000)
    after = profile(CUSTOMERS, 2000, customer_id=1000)
    (finding,) = joins.shape_drift(before, after, ["customer_id"])
    assert finding.kind == "join_key_fan_out"
    assert finding.severity is Severity.ERROR
    assert "was unique and is not any more" in finding.detail


def test_a_small_move_is_below_tolerance():
    before = profile(CUSTOMERS, 1000, customer_id=1000)
    after = profile(CUSTOMERS, 1050, customer_id=1000)
    assert joins.shape_drift(before, after, ["customer_id"]) == []


def test_the_tolerance_is_the_callers_choice():
    before = profile(CUSTOMERS, 1000, customer_id=1000)
    after = profile(CUSTOMERS, 1100, customer_id=1000)
    assert joins.shape_drift(before, after, ["customer_id"], tolerance=0.5) == []
    assert joins.shape_drift(before, after, ["customer_id"], tolerance=0.05)


def test_a_falling_fan_out_is_only_a_warning():
    """Fewer duplicates does not duplicate anything downstream."""
    before = profile(ORDERS, 1000, customer_id=250)
    after = profile(ORDERS, 1000, customer_id=1000)
    (finding,) = joins.shape_drift(before, after, ["customer_id"])
    assert finding.severity is Severity.WARN
    assert "fell" in finding.detail


def test_an_unmeasurable_key_produces_no_drift_finding():
    before = profile(CUSTOMERS, 1000, customer_id=None)
    after = profile(CUSTOMERS, 2000, customer_id=1000)
    assert joins.shape_drift(before, after, ["customer_id"]) == []


def test_the_finding_carries_both_fan_outs():
    before = profile(CUSTOMERS, 1000, customer_id=1000)
    after = profile(CUSTOMERS, 4000, customer_id=1000)
    (finding,) = joins.shape_drift(before, after, ["customer_id"])
    assert finding.before == pytest.approx(1.0)
    assert finding.after == pytest.approx(4.0)


# -- orphans, provable in one direction only -----------------------------------


def test_more_keys_on_the_left_proves_some_cannot_match():
    left = profile(ORDERS, 1000, customer_id=900)
    right = profile(CUSTOMERS, 400, customer_id=400)
    assert joins.orphan_floor(left, right, "customer_id") == 500


def test_a_smaller_left_side_proves_nothing():
    """Equal or fewer keys may still be entirely disjoint — that needs a scan."""
    left = profile(ORDERS, 1000, customer_id=400)
    right = profile(CUSTOMERS, 900, customer_id=900)
    assert joins.orphan_floor(left, right, "customer_id") == 0


def test_the_floor_is_none_when_either_side_is_unmeasurable():
    left = profile(ORDERS, 1000, customer_id=None)
    right = profile(CUSTOMERS, 900, customer_id=900)
    assert joins.orphan_floor(left, right, "customer_id") is None


# -- amplification -------------------------------------------------------------


def test_an_output_larger_than_its_inputs_amplifies():
    inputs = [profile(ORDERS, 1000, k=1000), profile(CUSTOMERS, 500, k=500)]
    assert joins.amplification(inputs, profile(JOINED, 4000, k=1000)) == pytest.approx(4.0)


def test_an_output_no_larger_than_its_inputs_does_not():
    inputs = [profile(ORDERS, 1000, k=1000)]
    assert joins.amplification(inputs, profile(JOINED, 800, k=800)) == pytest.approx(0.8)


def test_amplification_of_no_inputs_is_none():
    assert joins.amplification([], profile(JOINED, 100, k=100)) is None


# -- the whole check -----------------------------------------------------------


def test_a_clean_one_to_one_join_proves_nothing():
    left = profile(ORDERS, 1000, customer_id=1000)
    right = profile(CUSTOMERS, 1000, customer_id=1000)
    out = profile(JOINED, 1000, customer_id=1000)
    risk = joins.join_risks(left, right, out, "customer_id")
    assert risk.is_clear


def test_a_fanned_out_join_reports_both_the_key_and_the_amplification():
    left = profile(ORDERS, 1000, customer_id=1000)
    right = profile(CUSTOMERS, 1000, customer_id=250)  # duplicated dimension rows
    out = profile(JOINED, 4000, customer_id=250)
    risk = joins.join_risks(left, right, out, "customer_id")

    kinds = {f.kind for f in risk.findings}
    assert "join_key_not_unique" in kinds
    assert "join_amplification" in kinds
    assert not risk.is_clear


def test_a_join_that_will_drop_rows_says_how_many_at_least():
    left = profile(ORDERS, 1000, customer_id=900)
    right = profile(CUSTOMERS, 400, customer_id=400)
    out = profile(JOINED, 400, customer_id=400)
    risk = joins.join_risks(left, right, out, "customer_id")
    assert risk.orphans == 500
    assert "at least 500 key(s) cannot match" in risk.summary()


def test_an_unmeasurable_side_is_named_not_skipped():
    left = profile(ORDERS, 1000, customer_id=None)
    right = profile(CUSTOMERS, 1000, customer_id=1000)
    out = profile(JOINED, 1000, customer_id=1000)
    risk = joins.join_risks(left, right, out, "customer_id")
    assert any("silver.orders" in u for u in risk.unmeasurable)
    assert "not measurable" in risk.summary()


def test_a_previous_output_profile_enables_drift_detection():
    left = profile(ORDERS, 1000, customer_id=1000)
    right = profile(CUSTOMERS, 1000, customer_id=1000)
    before = profile(JOINED, 1000, customer_id=1000)
    after = profile(JOINED, 5000, customer_id=1000)
    risk = joins.join_risks(left, right, after, "customer_id", previous=before)
    assert any(f.kind == "join_key_fan_out" for f in risk.findings)


def test_a_clear_result_never_claims_the_join_is_safe():
    left = profile(ORDERS, 1000, customer_id=1000)
    right = profile(CUSTOMERS, 1000, customer_id=1000)
    out = profile(JOINED, 1000, customer_id=1000)
    summary = joins.join_risks(left, right, out, "customer_id").summary()
    assert "no join risk proven" in summary
    assert "needs a scan this does not do" in summary


# -- rollups -------------------------------------------------------------------


def test_risky_keys_returns_only_what_was_proven():
    clean = joins.join_risks(
        profile(ORDERS, 100, k=100),
        profile(CUSTOMERS, 100, k=100),
        profile(JOINED, 100, k=100),
        "k",
    )
    bad = joins.join_risks(
        profile(ORDERS, 100, k=100), profile(CUSTOMERS, 100, k=10), profile(JOINED, 900, k=10), "k"
    )
    assert joins.risky_keys([clean, bad]) == [bad]


def test_candidate_keys_are_the_unique_columns():
    made = profile(ORDERS, 1000, order_id=1000, customer_id=250, status=4)
    assert joins.candidate_keys(made) == ["order_id"]
