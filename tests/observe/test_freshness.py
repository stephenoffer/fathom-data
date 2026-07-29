"""Transitive freshness: a table is only as fresh as its oldest input."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom import freshness
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph

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


# -- freshness -----------------------------------------------------------------


def test_effective_age_takes_the_oldest_input(graph):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    built = {
        RAW: datetime(2026, 3, 10, tzinfo=UTC),
        SILVER: datetime(2026, 3, 20, tzinfo=UTC),
        GOLD: datetime(2026, 3, 20, tzinfo=UTC),
    }
    # Gold was rebuilt minutes ago and still carries ten-day-old information.
    assert freshness.effective_age(graph, GOLD, built, now=now) == timedelta(days=10)
    assert freshness.age(built[GOLD], now=now) == timedelta(0)


def test_blame_names_the_responsible_input(graph):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    built = {RAW: datetime(2026, 3, 10, tzinfo=UTC), GOLD: datetime(2026, 3, 20, tzinfo=UTC)}
    culprit = freshness.blame(graph, GOLD, built, now=now)
    assert culprit is not None and culprit[0] == RAW
    assert freshness.freshness_path(graph, GOLD, built, now=now) == [RAW, SILVER, GOLD]


def test_unmeasured_is_not_fresh(graph):
    sla = freshness.SLA(GOLD, max_age=timedelta(hours=1))
    assert not freshness.is_fresh(graph, sla, {})
    assert freshness.effective_age(graph, GOLD, {}) is None


def test_stale_and_late_are_reported_separately(graph):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    built = {RAW: datetime(2026, 3, 1, tzinfo=UTC), GOLD: datetime(2026, 3, 20, tzinfo=UTC)}
    sla = freshness.SLA(GOLD, max_age=timedelta(days=1), expected_interval=timedelta(days=1))
    report = freshness.report(graph, [sla], built, now=now)
    assert report.stale and not report.late
    assert report.stale[0][2] == RAW
    assert "STALE" in report.summary()

    # Fresh upstream, but this dataset's own build has not run: late, not stale.
    fresh_built = {RAW: now, GOLD: datetime(2026, 3, 1, tzinfo=UTC)}
    late = freshness.report(graph, [sla], fresh_built, now=now)
    assert late.late == [GOLD]


def test_propagate_freshness_takes_the_oldest(graph):
    built = {RAW: datetime(2026, 3, 1, tzinfo=UTC), SILVER: datetime(2026, 3, 20, tzinfo=UTC)}
    resolved = freshness.propagate_freshness(graph, built)
    assert resolved[GOLD] == datetime(2026, 3, 1, tzinfo=UTC)


def test_worst_offenders_weights_by_reach(graph):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    built = {
        RAW: datetime(2026, 3, 18, tzinfo=UTC),
        GOLD: datetime(2026, 3, 17, tzinfo=UTC),
    }
    ranked = freshness.worst_offenders(graph, built, now=now)
    # Raw is newer but holds back two datasets, so it outranks the older leaf.
    assert ranked[0][0] == RAW


def test_slas_from_covers_the_leaves(graph):
    slas = freshness.slas_from(graph, max_age=timedelta(hours=6))
    assert [s.dataset for s in slas] == [GOLD]


def test_late_and_stale_are_different_questions(graph):
    """A slow upstream and a broken scheduler have different fixes."""
    now = datetime(2026, 3, 20, tzinfo=UTC)
    sla = freshness.SLA(GOLD, expected_interval=timedelta(days=1))

    on_time = {GOLD: datetime(2026, 3, 20, tzinfo=UTC)}
    assert not freshness.is_late(sla, on_time, now=now)

    overdue = {GOLD: datetime(2026, 3, 1, tzinfo=UTC)}
    assert freshness.is_late(sla, overdue, now=now)

    # Never built is always late, and never silently on time.
    assert freshness.is_late(sla, {}, now=now)

    # No schedule declared means the question does not apply.
    assert not freshness.is_late(freshness.SLA(GOLD), overdue, now=now)


def test_stale_closure_names_every_contributor(graph):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    built = {RAW: datetime(2026, 3, 1, tzinfo=UTC), SILVER: now, GOLD: now}

    late = freshness.stale_closure(graph, GOLD, built, max_age=timedelta(days=2), now=now)
    assert RAW in late
    assert SILVER not in late


def test_expected_next_build_reads_the_schedule():
    sla = freshness.SLA(GOLD, expected_interval=timedelta(hours=6))
    built = {GOLD: datetime(2026, 3, 20, tzinfo=UTC)}

    assert freshness.expected_next_build(sla, built) == datetime(2026, 3, 20, 6, tzinfo=UTC)
    assert freshness.expected_next_build(freshness.SLA(GOLD), built) is None
    assert freshness.expected_next_build(sla, {}) is None
