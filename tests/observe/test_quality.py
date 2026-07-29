"""Expectations over profiles, and suites learned from them."""

from __future__ import annotations

from datetime import datetime

import pytest

from fathom import quality
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph
from fathom.observe.profile import ColumnProfile, Profile, Severity

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.events")
GOLD = DatasetId("duckdb", "gold.monthly")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MONTH = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))


@pytest.fixture
def graph() -> Graph:
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(SILVER, DAY)
    g.add_dataset(GOLD, MONTH)
    g.add_edge(Edge(RAW, SILVER, PartitionMapping.identity(DAY), evidence="sql:1"))
    g.add_edge(Edge(SILVER, GOLD, PartitionMapping.rollup(DAY, MONTH), evidence="sql:2"))
    return g


@pytest.fixture
def plan(graph):
    days = [KeyPredicate.of(dt=datetime(2026, 3, d)) for d in range(1, 8)]
    return graph.invalidate({RAW: days})


# -- quality -------------------------------------------------------------------


@pytest.fixture
def profile() -> Profile:
    return Profile(
        dataset=RAW,
        row_count=1000,
        columns=(
            ColumnProfile("id", "string", row_count=1000, null_count=0, min="a", max="z"),
            ColumnProfile("amount", "double", row_count=1000, null_count=50, min=0.0, max=100.0),
        ),
    )


def test_expectations_pass_on_the_profile_they_describe(profile):
    result = quality.check(
        profile,
        [
            quality.not_null("id"),
            quality.dtype_is("amount", "double"),
            quality.in_range("amount", 0.0, 200.0),
            quality.row_count_between(500, 2000),
        ],
    )
    assert result.passed
    assert result.checked == 4
    assert result.findings == []


def test_expectations_catch_the_failures(profile):
    result = quality.check(
        profile,
        [
            quality.not_null("amount"),
            quality.dtype_is("amount", "int64"),
            quality.max_below("amount", 50.0),
            quality.schema_matches(["id", "missing_column"]),
        ],
    )
    assert not result.passed
    kinds = {f.kind for f in result.findings}
    assert kinds == {"null_rate", "dtype_change", "max_above_bound", "schema_mismatch"}


def test_a_missing_column_is_a_finding_not_a_crash(profile):
    result = quality.check(profile, [quality.not_null("nope")])
    assert result.findings[0].kind == "column_missing"


def test_unverifiable_expectations_are_skipped_not_passed(profile):
    # No distinct count in the footer, so uniqueness cannot be checked either way.
    result = quality.check(profile, [quality.unique("id")])
    assert result.findings == []
    # Reporting PASS over an expectation nothing could decide is read as assurance.
    assert result.skipped == [quality.unique("id")]
    assert result.checked == 0
    assert not result.complete
    assert "unverifiable" in result.summary()


def test_a_fully_unverifiable_suite_does_not_read_as_assurance(profile):
    """Green and incomplete are different answers, and the second one matters."""
    result = quality.check(profile, [quality.unique("id"), quality.unique("amount")])
    assert result.passed  # nothing failed...
    assert not result.complete  # ...because nothing was checked
    assert result.checked == 0


def test_learned_suites_widen_and_are_marked(profile):
    suite = quality.learn(profile)
    assert suite.learned == suite.expectations
    assert not suite.asserted

    # The suite it was learned from must pass against itself.
    assert quality.run(suite, profile).passed

    ranges = [e for e in suite.expectations if e.kind == "in_range" and e.column == "amount"]
    assert ranges and ranges[0].params["low"] < 0.0 < 100.0 < ranges[0].params["high"]


def test_a_learned_suite_still_catches_a_type_change(profile):
    suite = quality.learn(profile)
    changed = Profile(
        dataset=RAW,
        row_count=1000,
        columns=(
            ColumnProfile("id", "string", row_count=1000, null_count=0),
            ColumnProfile("amount", "int64", row_count=1000, null_count=50),
        ),
    )
    result = quality.run(suite, changed)
    assert not result.passed
    assert any(f.kind == "dtype_change" for f in result.errors)


def test_merge_keeps_asserted_expectations_over_learned(profile):
    learned = quality.learn(profile)
    asserted = quality.Suite(dataset=RAW, expectations=[quality.not_null("amount")])
    merged = quality.merge(learned, asserted)
    null_checks = [e for e in merged.expectations if e.kind == "null_rate_below"]
    assert any(not e.learned and e.column == "amount" for e in null_checks)


def test_suite_round_trips_through_a_dict(profile):
    suite = quality.learn(profile)
    restored = quality.from_dict(quality.to_dict(suite), RAW)
    assert len(restored.expectations) == len(suite.expectations)
    assert restored.expectations[0].severity is Severity.ERROR
