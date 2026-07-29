"""Purpose limitation and residency, intersecting downstream."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.ai import assets, training
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, PartitionField, PartitionSpec
from fathom.govern import consent
from fathom.graph import Edge, Graph

SCRAPED = DatasetId("s3://eu-west-1-lake", "corpus.scraped")
INTERNAL = DatasetId("s3://eu-west-1-lake", "raw.users")
JOINED = DatasetId("s3://us-east-1-lake", "gold.combined")
MODEL = assets.model("assistant", registry="internal")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))


@pytest.fixture
def graph() -> Graph:
    """Two sources joining into one derived table that then trains a model."""
    g = Graph()
    g.add_dataset(SCRAPED, DAY)
    g.add_dataset(INTERNAL, DAY)
    g.add_dataset(JOINED, DAY)
    g.add_edge(Edge(SCRAPED, JOINED, PartitionMapping.identity(DAY), evidence="sql:1"))
    g.add_edge(Edge(INTERNAL, JOINED, PartitionMapping.identity(DAY), evidence="sql:1"))
    run = training.TrainingRun(model=MODEL, code_version="abc")
    run.add_input(JOINED, snapshot="v1")
    training.record_training_run(g, run)
    return g


# -- consent -------------------------------------------------------------------


def test_purposes_intersect_rather_than_union(graph):
    declared = {
        SCRAPED: consent.ConsentScope(SCRAPED, frozenset({consent.Purpose.TRAINING})),
        INTERNAL: consent.ConsentScope(
            INTERNAL, frozenset({consent.Purpose.TRAINING, consent.Purpose.FRAUD})
        ),
    }
    permitted = consent.permitted_purposes(graph, JOINED, declared)
    assert permitted == frozenset({consent.Purpose.TRAINING})


def test_a_disjoint_join_permits_nothing(graph):
    declared = {
        SCRAPED: consent.ConsentScope(SCRAPED, frozenset({consent.Purpose.RESEARCH})),
        INTERNAL: consent.ConsentScope(INTERNAL, frozenset({consent.Purpose.FRAUD})),
    }
    assert consent.permitted_purposes(graph, JOINED, declared) == frozenset()
    assert not consent.purpose_allowed(graph, JOINED, consent.Purpose.FRAUD, declared)


def test_undeclared_data_fails_closed(graph):
    assert consent.permitted_purposes(graph, JOINED, {}) == frozenset()
    assert not consent.purpose_allowed(graph, MODEL, consent.Purpose.TRAINING, {})


def test_blocking_sources_name_the_fix(graph):
    declared = {
        SCRAPED: consent.ConsentScope(SCRAPED, frozenset({consent.Purpose.RESEARCH})),
        INTERNAL: consent.ConsentScope(INTERNAL, frozenset({consent.Purpose.TRAINING})),
    }
    blockers = consent.blocking_sources(graph, MODEL, consent.Purpose.TRAINING, declared)
    assert blockers == [SCRAPED]


def test_unconsented_uses_are_reported(graph):
    declared = {
        SCRAPED: consent.ConsentScope(SCRAPED, frozenset({consent.Purpose.ANALYTICS})),
        INTERNAL: consent.ConsentScope(INTERNAL, frozenset({consent.Purpose.ANALYTICS})),
    }
    notes = consent.unconsented_uses(graph, declared, intended={MODEL: consent.Purpose.TRAINING})
    assert notes and "not permitted by" in notes[0]


def test_propagate_purposes_covers_the_graph(graph):
    declared = {
        SCRAPED: consent.ConsentScope(SCRAPED, frozenset({consent.Purpose.TRAINING})),
        INTERNAL: consent.ConsentScope(INTERNAL, frozenset({consent.Purpose.TRAINING})),
    }
    resolved = consent.propagate_purposes(graph, declared)
    assert resolved[MODEL] == frozenset({consent.Purpose.TRAINING})


def test_region_is_read_off_the_identity():
    assert consent.region_of(SCRAPED) == "eu-west-1"
    assert consent.region_of(JOINED) == "us-east-1"
    assert consent.region_of(DatasetId("duckdb", "x")) == ""


def test_residency_violation_when_eu_data_lands_in_us(graph):
    constraints = {SCRAPED: consent.Residency(SCRAPED, frozenset({"eu-west-1"}))}
    notes = consent.residency_violations(graph, constraints)
    assert any(str(JOINED) in note for note in notes)
    assert consent.transfer_paths(graph, constraints)


def test_retention_expiry_reaches_downstream(graph):
    declared = {
        INTERNAL: consent.ConsentScope(
            INTERNAL,
            frozenset({consent.Purpose.SERVICE}),
            collected=datetime(2020, 1, 1, tzinfo=UTC),
            retention=timedelta(days=365),
        )
    }
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert consent.expired(declared, now=now) == [INTERNAL]
    notes = consent.retention_violations(graph, declared, now=now)
    assert any(str(JOINED) in note for note in notes)


def test_consent_report_assembles_the_position(graph):
    declared = {
        SCRAPED: consent.ConsentScope(SCRAPED, frozenset({consent.Purpose.RESEARCH})),
        INTERNAL: consent.ConsentScope(INTERNAL, frozenset({consent.Purpose.RESEARCH})),
    }
    report = consent.report(graph, MODEL, declared, intended=[consent.Purpose.TRAINING])
    assert not report.is_clear
    assert report.permitted == frozenset({consent.Purpose.RESEARCH})
    assert "blocked by" in report.denied[0]


def test_lapsed_retention_withdraws_permission():
    """Retention running out is a withdrawal, not a note for a separate report.

    `expired()` already identified these datasets; the permission path simply never
    asked, so training on data whose consent lapsed years ago read as permitted.
    """
    ds = DatasetId("duckdb", "raw.signups")
    lapsed = consent.ConsentScope(
        dataset=ds,
        purposes=frozenset({consent.Purpose.TRAINING}),
        collected=datetime(2020, 1, 1, tzinfo=UTC),
        retention=timedelta(days=365),
    )
    graph = Graph()
    graph.add_dataset(ds)

    assert consent.expired({ds: lapsed}) == [ds]
    assert lapsed.is_expired()
    assert not lapsed.allows(consent.Purpose.TRAINING)
    assert not consent.purpose_allowed(graph, ds, consent.Purpose.TRAINING, {ds: lapsed})
    assert consent.permitted_purposes(graph, ds, {ds: lapsed}) == frozenset()

    # A scope still inside its retention window is unaffected.
    live = consent.ConsentScope(
        dataset=ds,
        purposes=frozenset({consent.Purpose.TRAINING}),
        collected=datetime(2020, 1, 1, tzinfo=UTC),
        retention=timedelta(days=365),
    )
    at_the_time = datetime(2020, 6, 1, tzinfo=UTC)
    assert live.allows(consent.Purpose.TRAINING, now=at_the_time)
