"""Traversal over the dependency graph."""

from __future__ import annotations

import pytest

from fathom import query
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping, Passthrough, TimeWindow
from fathom.core.types import ColumnRef, DatasetId, PartitionField, PartitionSpec
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


# -- traversal -----------------------------------------------------------------


def test_parents_and_children(graph):
    assert query.children(graph, SILVER) == [GOLD, SIDE]
    assert query.parents(graph, GOLD) == [SILVER]
    assert query.neighbors(graph, SILVER) == [GOLD, SIDE, RAW]


def test_descendants_and_ancestors(graph):
    assert query.descendants(graph, RAW) == [GOLD, SIDE, SILVER]
    assert query.ancestors(graph, GOLD) == [RAW, SILVER]
    assert query.ancestors(graph, RAW) == []


def test_reachability_predicates(graph):
    assert query.is_upstream_of(graph, RAW, GOLD)
    assert not query.is_upstream_of(graph, GOLD, RAW)
    assert query.has_path(graph, RAW, GOLD)
    assert query.distance(graph, RAW, GOLD) == 2
    assert query.distance(graph, GOLD, RAW) is None
    assert query.distance(graph, RAW, RAW) == 0


def test_blast_radius_counts_only_downstream(graph):
    assert query.blast_radius(graph, RAW) == 3
    assert query.blast_radius(graph, GOLD) == 0


def test_paths_and_shortest_path(graph):
    assert query.shortest_path(graph, RAW, GOLD) == [RAW, SILVER, GOLD]
    assert query.paths_between(graph, RAW, GOLD) == [[RAW, SILVER, GOLD]]
    assert query.shortest_path(graph, GOLD, RAW) is None


def test_between_returns_the_corridor(graph):
    assert query.between(graph, RAW, GOLD) == [GOLD, RAW, SILVER]


def test_effective_mapping_composes_the_whole_path(graph):
    mapping = query.effective_mapping(graph, [RAW, SILVER, GOLD])
    assert isinstance(mapping.get("dt"), TimeWindow)
    assert mapping.get("dt").out_grain is Grain.MONTH
    assert isinstance(mapping.get("region"), Passthrough)


def test_effective_mapping_widens_when_a_hop_is_missing(graph):
    mapping = query.effective_mapping(graph, [RAW, GOLD])
    assert mapping.is_unbounded


def test_roots_leaves_and_isolated(graph):
    assert query.roots(graph) == [RAW, ORPHAN]
    assert query.leaves(graph) == [GOLD, SIDE, ORPHAN]
    assert query.isolated(graph) == [ORPHAN]


def test_levels_and_topological_order(graph):
    levels = query.levels(graph)
    assert levels[0] == [RAW, ORPHAN]
    assert levels[1] == [SILVER]
    order = query.topological_order(graph)
    assert order.index(RAW) < order.index(SILVER) < order.index(GOLD)


def test_connected_components_finds_the_orphan(graph):
    components = query.connected_components(graph)
    assert len(components) == 2
    assert components[0] == [GOLD, SIDE, RAW, SILVER]
    assert components[1] == [ORPHAN]


def test_cycles_are_detected_and_bounded():
    g = Graph()
    g.add_edge(Edge(RAW, SILVER, PartitionMapping()))
    g.add_edge(Edge(SILVER, RAW, PartitionMapping()))
    assert query.has_cycle(g)
    assert query.cycles(g) == [[RAW, SILVER]]
    # A self-loop is a cycle too, and must not hang the walk.
    g.add_edge(Edge(GOLD, GOLD, PartitionMapping()))
    assert [GOLD] in query.cycles(g)
    assert query.descendants(g, GOLD) == [GOLD]


def test_common_and_lowest_common_ancestors(graph):
    assert query.common_ancestors(graph, GOLD, SIDE) == [RAW, SILVER]
    assert query.lowest_common_ancestors(graph, GOLD, SIDE) == [SILVER]


def test_column_level_walks(graph):
    assert query.columns_of(graph, SILVER) == ["amount"]
    assert query.column_descendants(graph, ColumnRef(RAW, "amount")) == [
        ColumnRef(GOLD, "revenue"),
        ColumnRef(SILVER, "amount"),
    ]
    assert query.column_ancestors(graph, ColumnRef(GOLD, "revenue")) == [
        ColumnRef(RAW, "amount"),
        ColumnRef(SILVER, "amount"),
    ]


def test_subgraph_keeps_only_internal_edges(graph):
    sub = query.upstream_subgraph(graph, GOLD)
    assert sub.datasets == [GOLD, RAW, SILVER]
    assert SIDE not in sub.datasets
    assert len(sub.edges) == 2
    assert sub.spec(GOLD) == MONTH


def test_reverse_widens_mappings(graph):
    flipped = query.reverse(graph)
    assert query.descendants(flipped, GOLD) == [RAW, SILVER]
    assert all(e.mapping.is_unbounded for e in flipped.edges)


def test_merge_graphs_prefers_a_real_spec(graph):
    other = Graph()
    other.add_dataset(GOLD)  # unpartitioned; must not overwrite the real spec
    merged = query.merge_graphs(graph, other)
    assert merged.spec(GOLD) == MONTH


def test_find_and_namespaces(graph):
    assert query.find(graph, "*gold*") == [GOLD, SIDE]
    assert query.namespaces(graph) == ["duckdb", "s3://lake"]
    assert query.unpartitioned_datasets(graph) == [ORPHAN]


# -- shared helpers ------------------------------------------------------------


def test_closure_includes_the_dataset_itself(graph):
    """The idiom this replaced forgot the dataset half the time it was written out."""
    assert query.closure(graph, GOLD) == [RAW, SILVER, GOLD]
    assert query.closure(graph, RAW) == [RAW]
    assert query.closure(graph, ORPHAN) == [ORPHAN]


def test_fold_downstream_combines_at_joins(graph):
    resolved = query.fold_downstream(graph, {RAW: {"a"}, SILVER: {"b"}}, combine=lambda x, y: x | y)
    assert resolved[RAW] == {"a"}
    assert resolved[SILVER] == {"a", "b"}
    assert resolved[GOLD] == {"a", "b"}


def test_fold_downstream_default_decides_the_unreached_case(graph):
    """The permissive-versus-closed choice is the interesting part, so it is explicit."""
    without = query.fold_downstream(graph, {}, combine=lambda x, y: x | y)
    assert ORPHAN not in without

    closed = query.fold_downstream(graph, {}, combine=lambda x, y: x | y, default=frozenset())
    assert closed[ORPHAN] == frozenset()
    assert closed[GOLD] == frozenset()


def test_fold_downstream_respects_depth_order(graph):
    """Gold must see raw's contribution through silver, not just silver's own seed."""
    resolved = query.fold_downstream(graph, {RAW: 1}, combine=lambda x, y: x + y)
    assert resolved[GOLD] == 1


def test_topological_order_survives_a_dependency_learned_twice():
    """Parallel edges are normal: a dbt manifest and a query log both report a link.

    Counting indegree per edge while decrementing per distinct child leaves the
    consumer's count permanently above zero, so it is misreported as cyclic and
    name-sorted to the end — ahead of the very dataset that feeds it.
    """
    start = DatasetId("duckdb", "m.start")
    mid = DatasetId("duckdb", "z.mid")
    final = DatasetId("duckdb", "a.final")
    g = Graph()
    g.add_edge(Edge(start, mid, PartitionMapping(), evidence="sql"))
    g.add_edge(Edge(start, mid, PartitionMapping(), evidence="dbt"))
    g.add_edge(Edge(mid, final, PartitionMapping(), evidence="sql"))

    order = query.topological_order(g)
    assert order.index(start) < order.index(mid) < order.index(final)


def test_depth_is_exact_on_a_graph_with_more_paths_than_any_cap():
    """Depth was derived by enumerating root-to-node paths under a cap of 32.

    A six-wide, six-deep layered graph has vastly more than 32 paths, so the cap
    silently truncated and the reported depth came back too small.
    """
    g = Graph()
    previous = [DatasetId("duckdb", "raw.root")]
    for layer in range(6):
        current = [DatasetId("duckdb", f"l{layer}.n{i}") for i in range(6)]
        for child in current:
            for parent in previous:
                g.add_edge(Edge(parent, child, PartitionMapping()))
        previous = current

    assert query.depth_of(g, previous[0]) == 6
    assert max(query.levels(g)) == 6
    assert query.height_of(g, DatasetId("duckdb", "raw.root")) == 6


# -- neighbourhood measures ----------------------------------------------------


def test_degree_measures_count_distinct_neighbours(graph):
    assert query.in_degree(graph, SILVER) == 1
    assert query.out_degree(graph, SILVER) == 2
    assert query.degree(graph, SILVER) == 3
    assert query.fan_out(graph, SILVER) == 2
    assert query.in_degree(graph, RAW) == 0
    assert query.degree(graph, ORPHAN) == 0


def test_siblings_share_a_parent(graph):
    assert query.siblings(graph, GOLD) == [SIDE]
    assert query.siblings(graph, SIDE) == [GOLD]
    assert query.siblings(graph, RAW) == []


def test_relatives_are_ancestors_and_descendants_without_self(graph):
    assert query.relatives(graph, SILVER) == [GOLD, SIDE, RAW]
    assert query.relatives(graph, ORPHAN) == []


def test_reachable_unions_several_seeds_and_includes_them(graph):
    assert query.reachable(graph, [GOLD, SIDE]) == [GOLD, SIDE]
    assert query.reachable(graph, [RAW]) == [GOLD, SIDE, RAW, SILVER]
    assert query.reachable(graph, [GOLD], downstream=False) == [GOLD, RAW, SILVER]


def test_is_isolated_only_for_datasets_with_no_edges(graph):
    assert query.is_isolated(graph, ORPHAN)
    assert not query.is_isolated(graph, RAW)


def test_common_descendants(graph):
    assert query.common_descendants(graph, RAW, SILVER) == [GOLD, SIDE]
    assert query.common_descendants(graph, GOLD, SIDE) == []


def test_column_paths_walk_upstream(graph):
    paths = query.column_paths(graph, ColumnRef(GOLD, "revenue"))
    assert [str(step) for step in paths[0]] == [
        f"{GOLD}#revenue",
        f"{SILVER}#amount",
    ]


# -- selection and reshaping ---------------------------------------------------


def test_select_and_namespace_filters(graph):
    assert query.select(graph, lambda ds: ds.name.startswith("gold.")) == [GOLD, SIDE]
    assert query.in_namespace(graph, "s3://lake") == [ORPHAN]
    assert query.namespaces(graph) == ["duckdb", "s3://lake"]


def test_partitioned_and_unpartitioned_split_the_graph(graph):
    assert query.partitioned_datasets(graph) == [GOLD, SIDE, RAW, SILVER]
    assert query.unpartitioned_datasets(graph) == [ORPHAN]


def test_downstream_subgraph_keeps_only_reachable_edges(graph):
    sub = query.downstream_subgraph(graph, SILVER)
    assert set(sub.datasets) == {SILVER, GOLD, SIDE}
    assert all(e.src in sub.datasets and e.dst in sub.datasets for e in sub.edges)
    assert RAW not in sub.datasets


def test_prune_and_without_are_complementary(graph):
    kept = query.prune(graph, [RAW, SILVER])
    assert set(kept.datasets) == {RAW, SILVER}
    assert len(kept.edges) == 1

    dropped = query.without(graph, [ORPHAN, SIDE])
    assert ORPHAN not in dropped.datasets
    assert SIDE not in dropped.datasets
    assert RAW in dropped.datasets


def test_copy_graph_preserves_specs_and_edges(graph):
    clone = query.copy_graph(graph)
    assert clone.datasets == graph.datasets
    assert len(clone.edges) == len(graph.edges)
    assert clone.spec(GOLD) == graph.spec(GOLD)
    # Independent: adding to the copy must not touch the original.
    clone.add_dataset(DatasetId("duckdb", "extra"))
    assert DatasetId("duckdb", "extra") not in graph.datasets


def test_dataset_index_maps_rendered_name_back_to_identity(graph):
    index = query.dataset_index(graph)
    assert index[str(GOLD)] == GOLD
    assert len(index) == len(graph.datasets)


def test_is_downstream_of_is_the_mirror_of_is_upstream_of(graph):
    assert query.is_downstream_of(graph, GOLD, RAW)
    assert not query.is_downstream_of(graph, RAW, GOLD)
    assert query.is_downstream_of(graph, GOLD, RAW) == query.is_upstream_of(graph, RAW, GOLD)


def test_edges_between_returns_every_parallel_edge(graph):
    """The same dependency learned twice is two claims, not one."""
    graph.add_edge(Edge(RAW, SILVER, PartitionMapping.unknown(DAY), evidence="dbt:1"))

    found = query.edges_between(graph, RAW, SILVER)
    assert [e.evidence for e in found] == ["dbt:1", "sql:1"]
    assert query.edges_between(graph, RAW, GOLD) == []


def test_edge_between_joins_parallel_claims_rather_than_picking_one(graph):
    """Joining widens; picking could narrow, which is the unsafe direction."""
    graph.add_edge(Edge(RAW, SILVER, PartitionMapping.unknown(DAY), evidence="dbt:1"))
    merged = query.edge_between(graph, RAW, SILVER)

    assert merged is not None
    assert merged.is_unbounded  # the wider of the two claims covers both
    assert query.edge_between(graph, RAW, GOLD) is None
