"""The last hop: what has already been told to whom."""

from __future__ import annotations

import pytest

from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph, sinks

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.revenue")
GOLD = DatasetId("duckdb", "gold.monthly")
SIDE = DatasetId("duckdb", "gold.unpublished")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))

EXEC = sinks.dashboard("revenue/exec", tool="looker")
BOARD = sinks.report("board/monthly", publisher="finance")
TENK = sinks.filing("10-K/2026", regulator="sec")


@pytest.fixture
def graph() -> Graph:
    g = Graph()
    for ds in (RAW, SILVER, GOLD, SIDE):
        g.add_dataset(ds, DAY)
    mapping = PartitionMapping.identity(DAY)
    g.add_edge(Edge(RAW, SILVER, mapping, evidence="sql:1"))
    g.add_edge(Edge(SILVER, GOLD, mapping, evidence="sql:2"))
    g.add_edge(Edge(RAW, SIDE, mapping, evidence="sql:3"))
    sinks.record_publication(g, EXEC, [GOLD])
    sinks.record_publication(g, BOARD, [GOLD])
    sinks.record_publication(g, TENK, [SILVER])
    return g


# -- identity ------------------------------------------------------------------


def test_each_constructor_makes_its_own_kind():
    assert sinks.kind_of(EXEC) is sinks.SinkKind.DASHBOARD
    assert sinks.kind_of(BOARD) is sinks.SinkKind.REPORT
    assert sinks.kind_of(TENK) is sinks.SinkKind.FILING
    assert sinks.kind_of(sinks.export("eu/partner")) is sinks.SinkKind.EXPORT
    assert sinks.kind_of(sinks.endpoint("scorer")) is sinks.SinkKind.ENDPOINT
    assert sinks.kind_of(sinks.notebook("adhoc")) is sinks.SinkKind.NOTEBOOK


def test_a_table_is_not_a_sink():
    assert sinks.kind_of(RAW) is None
    assert not sinks.is_sink(RAW)
    assert sinks.is_sink(EXEC)


def test_an_unnamed_sink_refuses():
    with pytest.raises(ValueError, match="needs a name"):
        sinks.dashboard("   ")


def test_the_instance_lands_in_the_namespace():
    assert sinks.dashboard("x", tool="Looker").namespace == "dashboard://looker"


def test_describe_reads_as_prose():
    assert sinks.describe(EXEC) == "dashboard revenue/exec on looker"
    assert sinks.describe(RAW) == str(RAW)


# -- recording -----------------------------------------------------------------


def test_publication_adds_edges_into_the_sink(graph):
    assert {e.src for e in graph.in_edges(EXEC)} == {GOLD}


def test_the_mapping_into_a_sink_is_unknown(graph):
    """A dashboard has no partitions; claiming a relationship would be unearned."""
    edge = graph.in_edges(EXEC)[0]
    assert edge.mapping.is_unbounded or not edge.mapping.fields


def test_publishing_to_a_non_sink_refuses():
    g = Graph()
    with pytest.raises(ValueError, match="not a sink identity"):
        sinks.record_publication(g, GOLD, [RAW])


def test_a_sink_cannot_be_an_input():
    g = Graph()
    with pytest.raises(ValueError, match="cannot be an input"):
        sinks.record_publication(g, EXEC, [BOARD])


def test_a_sink_is_terminal(graph):
    """A sink feeding a table would extend every restatement cone through it forever."""
    g = Graph()
    sinks.record_publication(g, EXEC, [GOLD])
    g.add_edge(Edge(EXEC, SIDE, PartitionMapping.unknown(DAY), evidence="bad"))
    with pytest.raises(ValueError, match="sinks are terminal"):
        sinks.record_publication(g, EXEC, [SILVER])


def test_declare_many_records_several_at_once():
    g = Graph()
    made = sinks.declare_many(g, [(EXEC, [GOLD]), (BOARD, [GOLD])])
    assert len(made) == 2


# -- the restatement question --------------------------------------------------


def test_sinks_of_reaches_through_intermediate_tables(graph):
    """The whole point: RAW is two hops from the dashboard and still reaches it."""
    assert sinks.sinks_of(graph, RAW) == sorted([EXEC, BOARD, TENK], key=str)


def test_a_dataset_with_no_publication_downstream_has_no_sinks(graph):
    assert sinks.sinks_of(graph, SIDE) == []


def test_restatement_impact_separates_artefacts_from_tables(graph):
    impact = sinks.restatement_impact(graph, SILVER)
    assert set(impact.tables) == {GOLD}
    assert set(impact.sinks) == {EXEC, BOARD, TENK}


def test_restatement_impact_groups_by_kind(graph):
    impact = sinks.restatement_impact(graph, GOLD)
    assert impact.by_kind[sinks.SinkKind.DASHBOARD] == [EXEC]
    assert impact.by_kind[sinks.SinkKind.REPORT] == [BOARD]


def test_filings_are_reported_separately_from_dashboards(graph):
    """A wrong dashboard is embarrassing; a wrong filing is a legal event."""
    impact = sinks.restatement_impact(graph, SILVER)
    assert TENK in impact.regulatory
    assert EXEC not in impact.regulatory


def test_a_signed_report_counts_as_regulatory(graph):
    assert BOARD in sinks.restatement_impact(graph, GOLD).regulatory


def test_an_unpublished_dataset_reports_nothing_published(graph):
    impact = sinks.restatement_impact(graph, SIDE)
    assert not impact.is_published
    assert "nothing published downstream" in impact.summary()


def test_the_summary_calls_out_the_amendment(graph):
    text = sinks.restatement_impact(graph, SILVER).summary()
    assert "filings or signed reports" in text
    assert "amendment, not a refresh" in text


def test_regulatory_exposure_is_its_own_question(graph):
    assert sinks.has_regulatory_exposure(graph, SILVER)
    assert not sinks.has_regulatory_exposure(graph, SIDE)


def test_publication_paths_gives_the_route(graph):
    routes = sinks.publication_paths(graph, RAW, EXEC)
    assert routes and routes[0][0] == RAW and routes[0][-1] == EXEC
    assert SILVER in routes[0]


# -- inventory -----------------------------------------------------------------


def test_unpublished_finds_the_dead_end(graph):
    assert SIDE in sinks.unpublished(graph)
    assert RAW not in sinks.unpublished(graph)


def test_a_sink_is_never_listed_as_unpublished(graph):
    assert not any(sinks.is_sink(ds) for ds in sinks.unpublished(graph))


def test_published_datasets_are_the_complement(graph):
    assert set(sinks.published_datasets(graph)) == {RAW, SILVER, GOLD}


def test_sources_of_answers_the_audit_question(graph):
    """This filing said X, so where did X come from."""
    assert sinks.sources_of(graph, TENK) == sorted([RAW, SILVER], key=str)


def test_sinks_in_lists_every_artefact(graph):
    assert set(sinks.sinks_in(graph)) == {EXEC, BOARD, TENK}


def test_by_kind_groups_the_inventory(graph):
    grouped = sinks.by_kind(graph)
    assert grouped[sinks.SinkKind.FILING] == [TENK]


def test_riskiest_ranks_by_how_far_a_mistake_reaches(graph):
    ranked = sinks.riskiest(graph)
    assert ranked[0] == (RAW, 3)
    assert (SIDE, 0) not in ranked


def test_coverage_is_the_published_fraction(graph):
    assert sinks.coverage(graph) == pytest.approx(3 / 4)


def test_coverage_of_an_empty_graph_is_zero():
    assert sinks.coverage(Graph()) == 0.0


# -- the notice ----------------------------------------------------------------


def test_the_notice_names_every_artefact_and_its_route(graph):
    text = sinks.notice_text(graph, RAW, reason="fx rates were wrong since March")
    assert "fx rates were wrong since March" in text
    assert "dashboard revenue/exec on looker" in text
    assert "via" in text


def test_the_notice_flags_the_filings_separately(graph):
    text = sinks.notice_text(graph, SILVER)
    assert "formal amendment" in text
    assert "filing 10-K/2026 on sec" in text


def test_the_notice_refuses_to_judge_materiality(graph):
    """The graph knows what is downstream, not what was material."""
    text = sinks.notice_text(graph, RAW)
    assert "not the same as what is material" in text


def test_a_notice_with_nothing_published_says_so(graph):
    text = sinks.notice_text(graph, SIDE)
    assert "nothing has been told to anyone" in text
