"""Structural graph diffing, and the merge gate it drives."""

from __future__ import annotations

from datetime import datetime

import pytest

from fathom import diff
from fathom.core.grains import Grain
from fathom.core.partitions import UNBOUNDED, PartitionMapping, TimeWindow
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


# -- diff ----------------------------------------------------------------------


def test_graph_diff_detects_widening_and_removal(graph):
    # Same shape, but the raw -> silver mapping lost its precision: a widening.
    after = Graph()
    for ds in graph.datasets:
        after.add_dataset(ds, graph.spec(ds))
    for edge in graph.edges:
        widened = edge.src == RAW and edge.dst == SILVER
        after.add_edge(
            Edge(
                edge.src,
                edge.dst,
                PartitionMapping.unknown(DAY) if widened else edge.mapping,
                columns=edge.columns,
                evidence=edge.evidence,
            )
        )

    result = diff.diff_graphs(graph, after)
    assert len(result.changed_edges) == 1
    assert result.widenings and not result.narrowings
    assert result.is_safe
    assert "1 mapping change" in result.summary()


def test_graph_diff_flags_a_narrowing_as_unsafe(graph):
    before = Graph()
    before.add_dataset(RAW, DAY)
    before.add_dataset(SILVER, DAY)
    before.add_edge(Edge(RAW, SILVER, PartitionMapping.unknown(DAY), evidence="sql:1"))

    after = Graph()
    after.add_dataset(RAW, DAY)
    after.add_dataset(SILVER, DAY)
    after.add_edge(Edge(RAW, SILVER, identity(), evidence="sql:1"))

    result = diff.diff_graphs(before, after)
    assert result.narrowings
    assert not result.is_safe
    assert "UNSAFE" in result.summary()


def test_mapping_direction_helpers():
    narrow = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY))
    wide = PartitionMapping.of(dt=UNBOUNDED)
    assert diff.mapping_widened(narrow, wide)
    assert diff.mapping_narrowed(wide, narrow)
    assert not diff.mapping_widened(narrow, narrow)


def test_review_comment_leads_with_the_dangerous_change(graph):
    empty = Graph()
    result = diff.diff_graphs(graph, empty)
    comment = diff.review_comment(result)
    body = comment.split("\n\n", 1)[1]
    assert body.startswith("#### Needs review")
    assert "removed `duckdb/raw.events` \u2192 `duckdb/silver.events`" in body


# -- the rest of the diff surface ----------------------------------------------


def test_spec_change_names_added_removed_and_regrained_fields():
    """A re-grained partition column invalidates every mapping across the dataset."""
    from fathom.core.grains import Grain
    from fathom.core.types import PartitionField, PartitionSpec

    before = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
    after = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH), PartitionField.value("tenant"))
    change = diff.diff_specs(before, after)

    assert [f.name for f in change.added_fields] == ["tenant"]
    assert [f.name for f in change.removed_fields] == ["region"]
    assert change.regrained == ("dt",)


def test_identical_specs_produce_no_change():
    from fathom.core.grains import Grain
    from fathom.core.types import PartitionField, PartitionSpec

    spec = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
    change = diff.diff_specs(spec, spec)
    assert change.added_fields == () and change.removed_fields == ()
    assert change.regrained == ()


def test_diff_plans_is_empty_for_the_same_seed_and_grows_with_a_wider_one(graph):
    from datetime import datetime

    from fathom.core.types import KeyPredicate

    one_day = {RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")]}
    before = graph.invalidate(one_day)
    assert diff.diff_plans(before, graph.invalidate(one_day)).is_empty
    assert "identical" in diff.diff_plans(before, graph.invalidate(one_day)).summary()

    wider = graph.invalidate(
        {
            RAW: [
                KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu"),
                KeyPredicate.of(dt=datetime(2026, 3, 20), region="eu"),
            ]
        }
    )
    grown = diff.diff_plans(before, wider)
    assert not grown.is_empty
    assert grown.partition_delta > 0


def test_changed_datasets_and_any_narrowing_over_a_graph_diff(graph):
    """A mapping that loses precision is the thing a review gate must catch."""
    from fathom.core.partitions import PartitionMapping

    widened = Graph()
    for ds in graph.datasets:
        widened.add_dataset(ds, graph.spec(ds))
    for edge in graph.edges:
        widened.add_edge(
            Edge(
                edge.src,
                edge.dst,
                PartitionMapping.unknown(graph.spec(edge.dst)),
                columns=edge.columns,
                evidence=edge.evidence,
            )
        )

    d = diff.diff_graphs(graph, widened)
    changed = diff.changed_datasets(d)
    assert changed
    assert SILVER in changed
    assert diff.any_narrowing([d]) is False  # widening, not narrowing


def test_edge_key_treats_evidence_as_part_of_the_identity():
    """A dbt manifest and a query log reporting one dependency are two claims."""
    from fathom.graph.diff import edge_key

    sql = Edge(RAW, SILVER, PartitionMapping(), evidence="sql:1")
    dbt = Edge(RAW, SILVER, PartitionMapping(), evidence="dbt:1")

    assert edge_key(sql) != edge_key(dbt)
    assert edge_key(sql) == edge_key(Edge(RAW, SILVER, PartitionMapping.unknown(DAY), "x", "sql:1"))


def test_spec_change_classifies_what_moved():
    from fathom.core.grains import Grain
    from fathom.graph.diff import SpecChange

    before = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
    after = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH), PartitionField.value("bucket"))
    change = SpecChange(RAW, before, after)

    assert [f.name for f in change.added_fields] == ["bucket"]
    assert [f.name for f in change.removed_fields] == ["region"]
    assert change.regrained == ("dt",)


def test_plan_diff_reports_the_direction_of_the_change(graph):
    from fathom.core.types import KeyPredicate
    from fathom.graph.diff import diff_plans

    narrow = graph.invalidate({RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")]})
    wide = graph.invalidate(
        {
            RAW: [
                KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu"),
                KeyPredicate.of(dt=datetime(2026, 3, 15), region="eu"),
            ]
        }
    )

    grew = diff_plans(narrow, wide)
    assert not grew.is_empty
    assert grew.partition_delta > 0
    assert "more partition" in grew.summary()
    assert diff_plans(narrow, narrow).is_empty
    assert "identical" in diff_plans(narrow, narrow).summary()
