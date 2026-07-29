"""dbt-style selection strings resolved against a graph."""

from __future__ import annotations

import pytest

from fathom import selectors
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


# -- selectors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "up", "down", "at"),
    [
        ("model", 0, 0, False),
        ("+model", selectors.UNLIMITED, 0, False),
        ("model+", 0, selectors.UNLIMITED, False),
        ("+model+", selectors.UNLIMITED, selectors.UNLIMITED, False),
        ("2+model", 2, 0, False),
        ("model+3", 0, 3, False),
        ("@model", 0, 0, True),
    ],
)
def test_selector_term_parsing(text, up, down, at):
    term = selectors.parse_term(text)
    assert (term.upstream, term.downstream, term.build_scope) == (up, down, at)


def test_selector_resolves_upstream_and_downstream(graph):
    assert selectors.resolve(graph, "silver.events") == [SILVER]
    assert selectors.resolve(graph, "+gold.monthly") == [GOLD, RAW, SILVER]
    assert selectors.resolve(graph, "raw.events+") == [GOLD, SIDE, RAW, SILVER]
    assert selectors.resolve(graph, "1+gold.monthly") == [GOLD, SILVER]


def test_selector_union_and_intersection(graph):
    assert selectors.resolve(graph, "raw.events gold.monthly") == [GOLD, RAW]
    assert selectors.resolve(graph, "ns:duckdb,name:gold.*") == [GOLD, SIDE]


def test_selector_build_scope_pulls_in_sibling_inputs(graph):
    # @silver must include gold and side, and everything they need to rebuild.
    assert selectors.resolve(graph, "@silver.events") == [GOLD, SIDE, RAW, SILVER]


def test_selector_exclusion_and_empty_behaviour(graph):
    assert selectors.resolve(graph, "raw.events+", exclude="gold.monthly") == [SIDE, RAW, SILVER]
    with pytest.raises(selectors.SelectorError):
        selectors.resolve(graph, "nothing.matches.this")
    assert selectors.select_datasets(graph, "nothing.matches.this") == []


def test_selector_explains_itself(graph):
    text = selectors.explain(graph, "+gold.monthly")
    assert "1 direct match" in text
    assert "3 dataset(s)" in text


# -- the rest of the public surface --------------------------------------------


def test_validate_reports_problems_without_needing_a_graph():
    assert selectors.validate("+gold.monthly+") == []
    assert selectors.validate("+") != []


def test_select_subgraph_returns_a_planable_graph(graph):
    sub = selectors.select_subgraph(graph, "+gold.monthly")
    assert GOLD in sub.datasets
    assert all(e.src in sub.datasets and e.dst in sub.datasets for e in sub.edges)


def test_select_edges_renders_only_edges_inside_the_selection(graph):
    edges = selectors.select_edges(graph, "+gold.monthly")
    assert edges and all(isinstance(e, str) for e in edges)
    # Nothing outside the selection may appear; `side` is not upstream of gold.
    assert not any("gold.side" in e for e in edges)


def test_difference_removes_one_selection_from_another(graph):
    everything = set(graph.datasets)
    without_orphan = selectors.difference(everything, {ORPHAN})
    assert ORPHAN not in without_orphan
    assert RAW in without_orphan


def test_selector_for_round_trips_through_resolve(graph):
    expression = selectors.selector_for([RAW, GOLD])
    assert set(selectors.resolve(graph, expression)) == {RAW, GOLD}
