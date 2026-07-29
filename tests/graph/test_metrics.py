"""Coverage and health — how much of the graph is worth planning on."""

from __future__ import annotations

import pytest

from fathom import metrics
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.events")
GOLD = DatasetId("duckdb", "gold.monthly")
SIDE = DatasetId("duckdb", "gold.side")
ORPHAN = DatasetId("s3://lake", "orphan")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
MONTH = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH), PartitionField.value("region"))


def identity() -> PartitionMapping:
    return PartitionMapping.identity(DAY)


def rollup() -> PartitionMapping:
    return PartitionMapping.rollup(DAY, MONTH)


@pytest.fixture
def graph() -> Graph:
    """raw -> silver -> {gold, side}, plus one disconnected dataset."""
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(SILVER, DAY)
    g.add_dataset(GOLD, MONTH)
    g.add_dataset(SIDE, DAY)
    g.add_dataset(ORPHAN)
    g.add_edge(Edge(RAW, SILVER, identity(), columns=(("amount", "amount"),), evidence="sql:1"))
    g.add_edge(Edge(SILVER, GOLD, rollup(), columns=(("amount", "revenue"),), evidence="sql:2"))
    g.add_edge(Edge(SILVER, SIDE, identity(), evidence="sql:3"))
    return g


# -- metrics -------------------------------------------------------------------


def test_graph_stats(graph):
    stats = metrics.graph_stats(graph)
    assert stats.datasets == 5
    assert stats.edges == 3
    assert stats.isolated == 1
    assert stats.components == 2
    assert stats.max_depth == 2


def test_coverage_reports_what_is_provable(graph):
    result = metrics.coverage(graph)
    assert result.datasets == 5
    assert result.specced == 4
    assert result.spec_ratio == pytest.approx(0.8)
    assert result.edge_ratio == 1.0  # every edge has at least one bounded field
    assert result.column_ratio == pytest.approx(2 / 3)
    assert result.field_ratio == 1.0


def test_coverage_falls_when_edges_are_unbounded():
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(SILVER, DAY)
    g.add_edge(Edge(RAW, SILVER, PartitionMapping.unknown(DAY)))
    assert metrics.bounded_edge_ratio(g) == 0.0
    assert metrics.precision_ceiling(g) == 0.0


def test_health_report_recommends_the_gaps(graph):
    report = metrics.health_report(graph)
    assert 0.0 < report.score <= 1.0
    assert any("partition specs" in note for note in report.recommendations)
    assert ORPHAN in report.suspicious["isolated"]


def test_importance_rankings(graph):
    assert metrics.most_depended_on(graph)[0] == (RAW, 3)
    assert metrics.bottlenecks(graph)[0][0] == SILVER
    assert metrics.longest_chain(graph) == [RAW, SILVER, GOLD]


def test_evidence_and_namespace_breakdowns(graph):
    assert metrics.evidence_breakdown(graph) == {"sql": 3}
    assert metrics.namespace_breakdown(graph) == {"duckdb": 4, "s3://lake": 1}


def test_reachability_counts_match_the_traversal_on_a_dag():
    """The bitset dynamic program must agree with walking from every node.

    Running a traversal per dataset is O(V x (V+E)) and takes minutes on a
    warehouse-sized graph, so the fast path matters — but only if it is exact.
    """
    from fathom.core.partitions import PartitionMapping
    from fathom.graph.query import ancestors, descendants

    g = Graph()
    previous = [DatasetId("duckdb", "raw.root")]
    for layer in range(4):
        current = [DatasetId("duckdb", f"l{layer}.n{i}") for i in range(4)]
        for child in current:
            for parent in previous:
                g.add_edge(Edge(parent, child, PartitionMapping()))
        previous = current

    assert metrics.reach_score(g) == {ds: len(descendants(g, ds)) for ds in g.datasets}
    expected = {
        ds: len(ancestors(g, ds)) * len(descendants(g, ds))
        for ds in g.datasets
        if ancestors(g, ds) and descendants(g, ds)
    }
    assert dict(metrics.bottlenecks(g, limit=len(g.datasets))) == expected


def test_reachability_still_correct_when_the_graph_has_a_cycle():
    """Reachability inside a cycle is not a fold over an order, so it falls back."""
    from fathom.core.partitions import PartitionMapping
    from fathom.graph.query import descendants

    a, b, c = (DatasetId("duckdb", n) for n in ("x.a", "x.b", "x.c"))
    g = Graph()
    g.add_edge(Edge(a, b, PartitionMapping()))
    g.add_edge(Edge(b, c, PartitionMapping()))
    g.add_edge(Edge(c, b, PartitionMapping()))  # cycle

    assert metrics.reach_score(g) == {ds: len(descendants(g, ds)) for ds in g.datasets}


# -- shape measures ------------------------------------------------------------


def test_density_and_average_degree(graph):
    # 3 edges over 5 datasets: 3 / (5*4)
    assert metrics.density(graph) == pytest.approx(0.15)
    assert metrics.average_degree(graph) == pytest.approx(6 / 5)
    assert metrics.density(Graph()) == 0.0
    assert metrics.average_degree(Graph()) == 0.0


def test_diameter_and_width(graph):
    # raw -> silver -> gold is two hops.
    assert metrics.diameter(graph) == 2
    # level 2 holds gold and side.
    assert metrics.width(graph) == 2


def test_degree_centrality_normalises_by_the_largest_possible(graph):
    centrality = metrics.degree_centrality(graph)
    # silver touches raw, gold and side: 3 of a possible 4.
    assert centrality[SILVER] == pytest.approx(0.75)
    assert centrality[ORPHAN] == 0.0
    assert metrics.degree_centrality(Graph()) == {}


def test_hubs_ranks_by_direct_neighbours(graph):
    assert metrics.hubs(graph, limit=1) == [(SILVER, 3)]


def test_coverage_ratios_are_exposed_individually(graph):
    assert metrics.spec_coverage(graph) == pytest.approx(0.8)  # 4 of 5 specced
    assert metrics.column_lineage_coverage(graph) == pytest.approx(2 / 3)


def test_plan_efficiency_reports_what_a_plan_avoided(graph):
    from datetime import datetime

    from fathom.core.types import KeyPredicate

    plan = graph.invalidate({RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")]})
    efficiency = metrics.plan_efficiency(graph, plan)
    assert efficiency["datasets_total"] == 5.0
    assert efficiency["datasets_affected"] == len(plan.dirty)
    assert efficiency["skip_ratio"] == pytest.approx((5 - len(plan.dirty)) / 5)
    assert 0.0 <= efficiency["widened_ratio"] <= 1.0


def test_summaries_render_without_raising(graph):
    assert "dataset(s)" in metrics.graph_stats(graph).summary()
    assert "coverage" in metrics.coverage(graph).summary()
    assert "lineage health" in metrics.health_report(graph).summary()


def test_connectivity_reports_the_largest_component(graph):
    """Below about 0.8 in a real warehouse, suspect identity normalization."""
    assert metrics.connectivity(graph) == pytest.approx(4 / 5)  # the orphan is the fifth
    assert metrics.connectivity(Graph()) == 0.0


def test_suspicious_datasets_names_categories_not_errors(graph):
    """Every category here is legitimate sometimes; they are where to look first."""
    found = metrics.suspicious_datasets(graph)
    assert ORPHAN in found["isolated"]
    assert "derived_without_spec" not in found  # every derived dataset here has a spec


def test_a_self_referencing_model_is_flagged():
    g = Graph()
    g.add_edge(Edge(RAW, RAW, PartitionMapping()))
    assert RAW in metrics.suspicious_datasets(g)["self_referencing"]


def test_graph_stats_are_descriptive_only(graph):
    stats = metrics.graph_stats(graph)
    assert stats.roots and stats.leaves
    assert "dataset(s)" in stats.summary()
    assert "disconnected" in stats.summary()  # two components, so it says so
