"""Promises between teams, and who a breach is owed to."""

from __future__ import annotations

from datetime import timedelta

from fathom.core.types import DatasetId
from fathom.govern import contracts
from fathom.observe.profile import ColumnProfile, Profile, Severity
from fathom.observe.quality import Suite, not_null

ORDERS = DatasetId("duckdb", "gold.orders")
EVENTS = DatasetId("duckdb", "raw.events")


def profile(*names: str, dataset: DatasetId = ORDERS, rows: int = 10, nulls: int = 0) -> Profile:
    return Profile(
        dataset=dataset,
        row_count=rows,
        columns=tuple(ColumnProfile(n, "string", row_count=rows, null_count=nulls) for n in names),
    )


def contract(**overrides) -> contracts.Contract:
    base = {
        "dataset": ORDERS,
        "producer": "platform",
        "consumers": ("finance",),
        "columns": ("order_id", "amount"),
    }
    return contracts.Contract(**{**base, **overrides})


# -- promised columns ----------------------------------------------------------


def test_a_met_contract_says_met():
    report = contracts.verify(contract(), profile=profile("order_id", "amount"))
    assert report.is_met
    assert report.summary().endswith("met")


def test_a_missing_column_is_a_breach():
    report = contracts.verify(contract(), profile=profile("order_id"))
    assert not report.is_met
    assert report.breaches[0].kind == "missing_column"
    assert "'amount'" in report.breaches[0].detail


def test_extra_columns_are_permitted():
    report = contracts.verify(contract(), profile=profile("order_id", "amount", "extra"))
    assert report.is_met


def test_a_breach_names_the_producer_and_the_consumers():
    """The difference between an alert and an escalation."""
    report = contracts.verify(contract(consumers=("finance", "ml")), profile=profile("order_id"))
    rendered = str(report.breaches[0])
    assert "owed by platform to finance, ml" in rendered


def test_a_contract_with_no_consumer_says_so_in_the_breach():
    report = contracts.verify(contract(consumers=()), profile=profile("order_id"))
    assert "no named consumer" in str(report.breaches[0])


# -- severity follows the blast radius -----------------------------------------


def test_a_breaking_schema_change_with_consumers_is_an_error():
    report = contracts.verify(
        contract(columns=()),
        profile=profile("order_id"),
        previous=profile("order_id", "amount"),
    )
    assert [b.severity for b in report.breaches] == [Severity.ERROR]


def test_the_same_change_with_no_consumer_is_only_a_warning():
    """Severity is a property of the blast radius, not of the change."""
    report = contracts.verify(
        contract(columns=(), consumers=()),
        profile=profile("order_id"),
        previous=profile("order_id", "amount"),
    )
    assert [b.severity for b in report.breaches] == [Severity.WARN]


def test_errors_are_separable_from_warnings():
    report = contracts.verify(
        contract(columns=(), consumers=()),
        profile=profile("order_id"),
        previous=profile("order_id", "amount"),
    )
    assert report.breaches and report.errors == []


# -- staleness -----------------------------------------------------------------


def test_an_age_past_the_promise_is_a_breach():
    report = contracts.verify(
        contract(columns=(), max_staleness=timedelta(hours=6)),
        profile=profile("order_id"),
        age=timedelta(hours=9),
    )
    assert report.breaches[0].kind == "staleness"
    assert "past the promised" in report.breaches[0].detail


def test_an_age_within_the_promise_is_met():
    report = contracts.verify(
        contract(columns=(), max_staleness=timedelta(hours=6)),
        profile=profile("order_id"),
        age=timedelta(hours=2),
    )
    assert report.is_met


def test_a_long_staleness_renders_in_days():
    report = contracts.verify(
        contract(columns=(), max_staleness=timedelta(days=1)),
        profile=profile("order_id"),
        age=timedelta(days=5),
    )
    assert "5d" in report.breaches[0].detail


# -- expectations --------------------------------------------------------------


def test_a_failing_expectation_is_a_breach():
    suite = Suite(dataset=ORDERS, name="orders").add(not_null("order_id"))
    report = contracts.verify(
        contract(columns=(), suite=suite), profile=profile("order_id", rows=10, nulls=4)
    )
    assert [b.kind for b in report.breaches] == ["expectation"]


def test_a_passing_expectation_is_met():
    suite = Suite(dataset=ORDERS, name="orders").add(not_null("order_id"))
    report = contracts.verify(contract(columns=(), suite=suite), profile=profile("order_id"))
    assert report.is_met


# -- what could not be checked -------------------------------------------------


def test_a_promise_with_no_evidence_is_unchecked_not_passed():
    """A report that looks met because the caller forgot the profile is the bug."""
    report = contracts.verify(contract())
    assert report.is_met
    assert any("columns" in item for item in report.unchecked)
    assert "not checked" in report.summary()


def test_staleness_without_an_age_is_unchecked():
    report = contracts.verify(
        contract(columns=(), max_staleness=timedelta(hours=6)), profile=profile("order_id")
    )
    assert any("staleness" in item for item in report.unchecked)


def test_expectations_without_a_profile_are_unchecked():
    suite = Suite(dataset=ORDERS, name="orders").add(not_null("order_id"))
    report = contracts.verify(contract(columns=(), suite=suite))
    assert any("expectations" in item for item in report.unchecked)


def test_a_fully_unchecked_report_does_not_claim_to_be_met():
    report = contracts.verify(contract())
    assert "not checked" in report.summary()
    assert not report.summary().endswith("met")


# -- across a set of contracts -------------------------------------------------


def test_contracts_for_filters_by_dataset():
    a, b = contract(), contract(dataset=EVENTS)
    assert contracts.contracts_for([a, b], ORDERS) == [a]


def test_two_contracts_on_one_dataset_are_both_returned():
    """Two consumers may legitimately hold different promises about the same table."""
    a = contract(consumers=("finance",))
    b = contract(consumers=("ml",))
    assert len(contracts.contracts_for([a, b], ORDERS)) == 2


def test_consumers_of_unions_across_contracts():
    a = contract(consumers=("finance",))
    b = contract(consumers=("ml", "finance"))
    assert contracts.consumers_of([a, b], ORDERS) == ["finance", "ml"]


def test_unowned_finds_datasets_with_no_contract():
    assert contracts.unowned([ORDERS, EVENTS], [contract()]) == [EVENTS]


def test_a_contract_with_no_consumers_promises_nothing_to_anyone():
    assert contract(consumers=()).is_unconsumed
    assert not contract().is_unconsumed


def test_breaches_puts_errors_first():
    strict = contracts.verify(contract(), profile=profile("order_id"))
    loose = contracts.verify(
        contract(columns=(), consumers=()),
        profile=profile("order_id"),
        previous=profile("order_id", "amount"),
    )
    ordered = contracts.breaches([loose, strict])
    assert ordered[0].severity is Severity.ERROR


def test_producers_groups_datasets_by_owning_team():
    made = contracts.producers([contract(), contract(dataset=EVENTS, producer="ingest")])
    assert made == {"ingest": [EVENTS], "platform": [ORDERS]}
