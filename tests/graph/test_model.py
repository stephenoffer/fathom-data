"""Invalidation planner behaviour."""

from __future__ import annotations

from datetime import datetime

from fathom.core.grains import Grain
from fathom.core.ids import normalize_table
from fathom.core.partitions import UNBOUNDED, PartitionMapping, TimeWindow
from fathom.core.types import ANY, ColumnRef, KeyPredicate, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MONTH = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))


def tbl(name: str):
    return normalize_table(name, system="duckdb")


def daily_chain() -> tuple[Graph, tuple]:
    """raw -> silver (daily, identity) -> gold (monthly rollup)."""
    raw, silver, gold = tbl("raw.events"), tbl("silver.events"), tbl("gold.monthly")
    g = Graph()
    g.add_dataset(raw, DAY)
    g.add_dataset(silver, DAY)
    g.add_dataset(gold, MONTH)
    g.add_edge(
        Edge(
            raw,
            silver,
            PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)),
            columns=(("dt", "dt"), ("amount", "amount")),
        )
    )
    g.add_edge(
        Edge(
            silver,
            gold,
            PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.MONTH)),
            columns=(("amount", "revenue"),),
        )
    )
    return g, (raw, silver, gold)


def test_one_dirty_day_propagates_to_its_month():
    g, (raw, silver, gold) = daily_chain()
    plan = g.invalidate({raw: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})

    assert plan.partitions(silver) == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 14))})
    assert plan.partitions(gold) == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 1))})
    assert gold not in plan.widened


def test_rebuild_order_is_topological():
    g, (raw, silver, gold) = daily_chain()
    plan = g.invalidate({raw: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
    assert plan.order.index(raw) < plan.order.index(silver) < plan.order.index(gold)


def test_unbounded_edge_widens_downstream():
    raw, mart = tbl("raw.events"), tbl("mart.summary")
    g = Graph()
    g.add_dataset(raw, DAY)
    g.add_dataset(mart, DAY)
    g.add_edge(Edge(raw, mart, PartitionMapping.of(dt=UNBOUNDED), evidence="opaque udf"))

    plan = g.invalidate({raw: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
    assert plan.partitions(mart) == frozenset({KeyPredicate.of(dt=ANY)})
    assert mart in plan.widened


def test_reconverging_paths_take_the_union():
    """Two routes to the same table must produce the union, not whichever ran last."""
    raw, a, b, sink = tbl("raw.e"), tbl("s.a"), tbl("s.b"), tbl("s.sink")
    g = Graph()
    for ds in (raw, a, b, sink):
        g.add_dataset(ds, DAY)
    ident = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY))
    g.add_edge(Edge(raw, a, ident))
    g.add_edge(Edge(raw, b, PartitionMapping.of(dt=TimeWindow("dt", 0, 2, Grain.DAY, Grain.DAY))))
    g.add_edge(Edge(a, sink, ident))
    g.add_edge(Edge(b, sink, ident))

    plan = g.invalidate({raw: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
    days = {k.get("dt").day for k in plan.partitions(sink)}
    assert days == {14, 15, 16}


def test_self_referencing_model_terminates_by_widening():
    """A model reading its own history is a cycle whose window grows every pass."""
    raw, roll = tbl("raw.e"), tbl("mart.rolling")
    g = Graph()
    g.add_dataset(raw, DAY)
    g.add_dataset(roll, DAY)
    g.add_edge(
        Edge(raw, roll, PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)))
    )
    g.add_edge(
        Edge(roll, roll, PartitionMapping.of(dt=TimeWindow("dt", 0, 1, Grain.DAY, Grain.DAY)))
    )

    plan = g.invalidate({raw: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
    assert roll in plan.widened
    assert roll in plan.cyclic
    assert plan.partitions(roll) == frozenset({KeyPredicate.of(dt=ANY)})


def test_untouched_datasets_stay_out_of_the_plan():
    g, (raw, silver, gold) = daily_chain()
    unrelated = tbl("other.thing")
    g.add_dataset(unrelated, DAY)
    plan = g.invalidate({raw: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
    assert unrelated not in plan.dirty


def test_empty_seed_produces_empty_plan():
    g, _ = daily_chain()
    assert g.invalidate({}).is_empty


def test_upstream_attribution_finds_the_source_column():
    """The move that turns a drift alert into a diagnosis."""
    g, (raw, silver, gold) = daily_chain()
    paths = g.upstream_columns(ColumnRef(gold, "revenue"))
    reached = {str(step) for path in paths for step in path}
    assert f"{silver}#amount" in reached
    assert f"{raw}#amount" in reached


def test_plan_records_why_each_dataset_is_dirty():
    g, (raw, silver, gold) = daily_chain()
    plan = g.invalidate({raw: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
    assert "seed" in plan.reasons[raw][0]
    assert str(silver) in plan.reasons[gold][0]


# -- link ----------------------------------------------------------------------


def test_link_registers_both_endpoints_and_defaults_to_unbounded():
    from fathom.graph import link

    graph = Graph()
    edge = link(graph, tbl("raw.events"), tbl("model.scorer"), evidence="training", dst_spec=DAY)

    assert set(graph.datasets) == {tbl("raw.events"), tbl("model.scorer")}
    assert edge.evidence == "training"
    assert edge.mapping.is_unbounded  # nothing proved a partition relationship
    assert graph.spec(tbl("model.scorer")) == DAY


def test_link_never_overwrites_a_known_spec():
    """A recorder's convention must not replace a spec a catalog established."""
    from fathom.graph import link

    graph = Graph()
    graph.add_dataset(tbl("raw.events"), MONTH)
    link(graph, tbl("raw.events"), tbl("gold.x"), evidence="sql", src_spec=DAY)
    assert graph.spec(tbl("raw.events")) == MONTH


def test_link_carries_an_explicit_mapping_and_columns():
    from fathom.graph import link

    graph = Graph()
    graph.add_dataset(tbl("a"), DAY)
    graph.add_dataset(tbl("b"), DAY)
    edge = link(
        graph,
        tbl("a"),
        tbl("b"),
        evidence="sql",
        mapping=PartitionMapping.identity(DAY),
        columns=(("x", "y"),),
    )
    assert not edge.mapping.is_unbounded
    assert edge.columns == (("x", "y"),)


def test_widening_a_cycle_reaches_everything_downstream():
    """The planner may over-invalidate. It must never under-invalidate.

    When a non-converging cycle forces a dataset to the top of the lattice, every
    consumer of that dataset has already been handed a narrower key set. Unless the
    widening is propagated, those consumers keep it and the plan serves stale data.
    """
    raw, roll, mart = tbl("raw.e"), tbl("mart.rolling"), tbl("mart.downstream")
    g = Graph()
    for ds in (raw, roll, mart):
        g.add_dataset(ds, DAY)
    g.add_edge(
        Edge(raw, roll, PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)))
    )
    g.add_edge(
        Edge(roll, roll, PartitionMapping.of(dt=TimeWindow("dt", 0, 1, Grain.DAY, Grain.DAY)))
    )
    g.add_edge(
        Edge(roll, mart, PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)))
    )

    plan = g.invalidate({raw: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})

    assert plan.partitions(roll) == frozenset({KeyPredicate.of(dt=ANY)})
    assert plan.partitions(mart) == frozenset({KeyPredicate.of(dt=ANY)}), (
        "the consumer of a widened dataset kept its narrow key set: under-invalidation"
    )


def test_a_wide_join_is_not_mistaken_for_a_cycle():
    """Fan-in is not non-convergence.

    A hub table legitimately has its dirty set enlarged once per dirty parent. If
    that is charged against the cycle budget, every join wider than the budget is
    force-widened to a full rebuild and reported as cyclic — precisely the tables
    where precision is worth the most.
    """
    hub = tbl("gold.hub")
    g = Graph()
    g.add_dataset(hub, DAY)
    parents = [tbl(f"raw.source_{i:02d}") for i in range(20)]
    for parent in parents:
        g.add_dataset(parent, DAY)
        g.add_edge(
            Edge(parent, hub, PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)))
        )

    seeds = {p: [KeyPredicate.of(dt=datetime(2026, 3, 1 + i))] for i, p in enumerate(parents)}
    plan = g.invalidate(seeds)

    assert hub not in plan.widened
    assert hub not in plan.cyclic
    assert plan.partitions(hub) == frozenset(
        KeyPredicate.of(dt=datetime(2026, 3, 1 + i)) for i in range(20)
    )
