"""Properties of sinks, lifetime, and history.

Three claims made in prose, checked against generated inputs:

- a restatement cone is exactly the descendants, split into artefacts and tables,
  and never claims regulatory exposure it does not have
- lifetime cost is order-independent, which matters because runs arrive out of order
  from every orchestrator that retries
- a graph digest depends on content and not on the order edges were inserted
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fathom.core.partitions import PartitionMapping
from fathom.core.types import UNPARTITIONED, DatasetId
from fathom.graph import Edge, Graph, history, sinks
from fathom.graph.plan import lifetime
from fathom.graph.plan.cost import CostModel
from fathom.graph.query import descendants

MODEL = CostModel(price_per_partition=1.0, price_per_tb_scanned=2.0)
MARCH = datetime(2026, 3, 1, tzinfo=UTC)

table_names = st.sampled_from(["a", "b", "c", "d"])
sink_kinds = st.sampled_from(list(sinks.SinkKind))


def table(name: str) -> DatasetId:
    return DatasetId("duckdb", f"t.{name}")


# -- sinks ---------------------------------------------------------------------


@given(
    edges=st.lists(st.tuples(table_names, table_names), max_size=8),
    published=st.lists(st.tuples(sink_kinds, table_names), max_size=4),
)
@settings(max_examples=200)
def test_the_restatement_cone_is_exactly_the_descendants(edges, published):
    """Artefacts plus tables must account for every descendant, with no overlap."""
    graph = Graph()
    for src, dst in edges:
        if src == dst:
            continue
        graph.add_edge(
            Edge(table(src), table(dst), PartitionMapping.unknown(UNPARTITIONED), evidence="x")
        )
    for index, (kind, source) in enumerate(published):
        graph.add_dataset(table(source), UNPARTITIONED)
        sink = sinks.of_kind(kind, f"artefact-{index}")
        if graph.out_edges(sink):
            continue
        sinks.record_publication(graph, sink, [table(source)])

    for node in list(graph.datasets):
        if sinks.is_sink(node):
            continue
        impact = sinks.restatement_impact(graph, node)
        reachable = set(descendants(graph, node))
        assert set(impact.sinks) | set(impact.tables) == reachable
        assert set(impact.sinks).isdisjoint(impact.tables)
        assert all(sinks.is_sink(s) for s in impact.sinks)
        assert not any(sinks.is_sink(t) for t in impact.tables)


# Matches the constructor's own contract: a name that is only whitespace or slashes
# is refused. Python counts the separator controls \x1c-\x1f as whitespace, which is
# how this filter and the source disagreed on the first run of this property.
@given(
    kind=sink_kinds,
    name=st.text(min_size=1, max_size=12).filter(lambda s: s.strip().strip("/")),
)
@settings(max_examples=200)
def test_a_sink_identity_round_trips_through_its_kind(kind, name):
    made = sinks.of_kind(kind, name)
    assert sinks.kind_of(made) is kind
    assert sinks.is_sink(made)


@given(
    edges=st.lists(st.tuples(table_names, table_names), max_size=8),
    published=st.lists(st.tuples(sink_kinds, table_names), max_size=4),
)
@settings(max_examples=150)
def test_regulatory_exposure_implies_a_regulatory_kind_downstream(edges, published):
    graph = Graph()
    for src, dst in edges:
        if src != dst:
            graph.add_edge(
                Edge(table(src), table(dst), PartitionMapping.unknown(UNPARTITIONED), evidence="x")
            )
    for index, (kind, source) in enumerate(published):
        graph.add_dataset(table(source), UNPARTITIONED)
        sink = sinks.of_kind(kind, f"artefact-{index}")
        if not graph.out_edges(sink):
            sinks.record_publication(graph, sink, [table(source)])

    for node in list(graph.datasets):
        if sinks.is_sink(node):
            continue
        exposed = sinks.has_regulatory_exposure(graph, node)
        downstream_kinds = {sinks.kind_of(d) for d in descendants(graph, node) if sinks.is_sink(d)}
        assert exposed == bool(downstream_kinds & sinks.REGULATORY)


@given(published=st.lists(st.tuples(sink_kinds, table_names), min_size=1, max_size=4))
@settings(max_examples=150)
def test_a_sink_is_never_reported_as_unpublished(published):
    graph = Graph()
    for index, (kind, source) in enumerate(published):
        graph.add_dataset(table(source), UNPARTITIONED)
        sink = sinks.of_kind(kind, f"artefact-{index}")
        if not graph.out_edges(sink):
            sinks.record_publication(graph, sink, [table(source)])
    assert not any(sinks.is_sink(ds) for ds in sinks.unpublished(graph))
    assert set(sinks.unpublished(graph)).isdisjoint(sinks.published_datasets(graph))


# -- lifetime ------------------------------------------------------------------

runs = st.lists(
    st.tuples(
        table_names,
        st.integers(min_value=0, max_value=30),
        st.integers(min_value=0, max_value=1000),
    ),
    max_size=20,
)


def to_records(raw) -> list[lifetime.RunRecord]:
    return [
        lifetime.RunRecord(table(name), MARCH + timedelta(days=day), partitions=partitions)
        for name, day, partitions in raw
    ]


@given(raw=runs, shuffle=st.randoms(use_true_random=False))
@settings(max_examples=200)
def test_accumulating_is_independent_of_the_order_runs_arrive(raw, shuffle):
    """Every orchestrator that retries delivers runs out of order."""
    records = to_records(raw)
    reordered = list(records)
    shuffle.shuffle(reordered)

    first = lifetime.accumulate(records, MODEL)
    second = lifetime.accumulate(reordered, MODEL)

    assert set(first) == set(second)
    for dataset, total in first.items():
        other = second[dataset]
        assert (total.runs, total.partitions) == (other.runs, other.partitions)
        assert abs(total.spend - other.spend) < 1e-9
        assert (total.first_run, total.last_run) == (other.first_run, other.last_run)


@given(raw=runs)
@settings(max_examples=200)
def test_total_spend_is_the_sum_of_the_parts(raw):
    totals = lifetime.accumulate(to_records(raw), MODEL)
    assert abs(lifetime.total_spend(totals) - sum(t.spend for t in totals.values())) < 1e-9


@given(raw=runs)
@settings(max_examples=200)
def test_only_datasets_with_runs_appear_and_all_are_measured(raw):
    """Absent is not zero: an unmeasured dataset must never look like a cheap one."""
    totals = lifetime.accumulate(to_records(raw), MODEL)
    assert set(totals) == {table(name) for name, _, _ in raw}
    assert all(total.is_measured for total in totals.values())


@given(raw=runs, threshold=st.floats(min_value=0.0, max_value=5000.0))
@settings(max_examples=200)
def test_a_read_dataset_is_never_flagged_for_review(raw, threshold):
    """Whatever it costs, something being read is not a retirement question."""
    totals = lifetime.accumulate(to_records(raw), MODEL)
    reads = {ds: 1 for ds in totals}
    for finding in lifetime.value(totals, reads, threshold=threshold):
        if finding.verdict is not lifetime.Verdict.UNMEASURED:
            assert finding.verdict is lifetime.Verdict.EARNING
            assert not finding.is_actionable


@given(raw=runs, threshold=st.floats(min_value=0.1, max_value=5000.0))
@settings(max_examples=200)
def test_actionable_findings_are_exactly_unread_and_over_the_threshold(raw, threshold):
    totals = lifetime.accumulate(to_records(raw), MODEL)
    findings = lifetime.value(totals, {}, threshold=threshold)
    for finding in findings:
        expected = finding.spend is not None and finding.spend >= threshold
        assert finding.is_actionable == expected


# -- history -------------------------------------------------------------------


@given(
    edges=st.lists(st.tuples(table_names, table_names), min_size=1, max_size=8, unique=True),
    shuffle=st.randoms(use_true_random=False),
)
@settings(max_examples=200)
def test_a_graph_digest_ignores_edge_insertion_order(edges, shuffle):
    """Insertion order is an artifact of ingest, not a property of the graph."""
    pairs = [(src, dst) for src, dst in edges if src != dst]
    assume(pairs)

    def build(order):
        g = Graph()
        for src, dst in order:
            g.add_edge(
                Edge(table(src), table(dst), PartitionMapping.unknown(UNPARTITIONED), evidence="x")
            )
        return g

    reordered = list(pairs)
    shuffle.shuffle(reordered)
    assert history.graph_digest(build(pairs)) == history.graph_digest(build(reordered))


@given(edges=st.lists(st.tuples(table_names, table_names), min_size=1, max_size=6, unique=True))
@settings(max_examples=150)
def test_a_different_graph_gets_a_different_digest(edges):
    pairs = [(src, dst) for src, dst in edges if src != dst]
    assume(pairs)

    def build(order, evidence="x"):
        g = Graph()
        for src, dst in order:
            g.add_edge(
                Edge(
                    table(src),
                    table(dst),
                    PartitionMapping.unknown(UNPARTITIONED),
                    evidence=evidence,
                )
            )
        return g

    assert history.graph_digest(build(pairs)) != history.graph_digest(build(pairs, "y"))
