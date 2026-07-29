"""Outbound payloads: OpenLineage, DataHub, Atlas, OpenMetadata."""

from __future__ import annotations

from datetime import datetime

import pytest

from fathom import emit
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph
from fathom.observe.profile import ColumnProfile, Profile

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


# -- emit ----------------------------------------------------------------------


def test_openlineage_events_skip_sources(graph):
    events = emit.openlineage_events(graph)
    jobs = {event["job"]["name"] for event in events}
    assert str(RAW) not in jobs
    assert str(SILVER) in jobs and str(GOLD) in jobs
    assert {event["eventType"] for event in events} == {"START", "COMPLETE"}


def test_run_ids_are_deterministic():
    assert emit.run_id_for("a") == emit.run_id_for("a")
    assert emit.run_id_for("a") != emit.run_id_for("b")


def test_partition_facet_carries_the_mapping(graph):
    edge = graph.in_edges(GOLD)[0]
    facet = emit.partition_facet(edge)
    assert facet["mapping"]["dt"].startswith("dt")
    assert facet["unbounded"] is False
    assert facet["evidence"] == "sql:2"


def test_column_lineage_facet_is_standard_shaped():
    g = Graph()
    g.add_edge(Edge(RAW, SILVER, PartitionMapping(), columns=(("amount", "total"),)))
    event = emit.openlineage_complete(g, SILVER)
    facet = event["outputs"][0]["facets"]["columnLineage"]
    assert facet["fields"]["total"]["inputFields"][0]["name"] == "raw.events"


def test_profile_becomes_a_quality_facet(graph):
    profile = Profile(
        dataset=GOLD,
        row_count=42,
        columns=(ColumnProfile("revenue", "double", row_count=42, null_count=1),),
    )
    event = emit.openlineage_complete(graph, GOLD, profile=profile)
    metrics = event["outputs"][0]["facets"]["dataQualityMetrics"]
    assert metrics["rowCount"] == 42
    assert metrics["columnMetrics"]["revenue"]["nullCount"] == 1


def test_plan_events_carry_the_partitions(graph, plan):
    events = emit.plan_events(graph, plan)
    facet = events[0]["outputs"][0]["facets"]["fathom_plan"]
    assert facet["partitions"]
    assert facet["widened"] is False


def test_json_lines_are_parseable(graph):
    import json

    text = emit.to_json_lines(emit.openlineage_events(graph))
    assert all(json.loads(line) for line in text.splitlines())


def test_datahub_and_atlas_payloads(graph):
    datahub = emit.to_datahub(graph, platform="duckdb")
    assert all(entry["aspectName"] == "upstreamLineage" for entry in datahub)
    atlas = emit.to_atlas(graph)
    kinds = {entity["typeName"] for entity in atlas["entities"]}
    assert kinds == {"DataSet", "Process"}
    assert len(emit.to_openmetadata(graph)) == len(graph.edges)


def test_emit_summary_describes_the_run(graph):
    events = emit.openlineage_events(graph)
    assert "event(s)" in str(emit.summarize(graph, events))


def test_start_events_name_the_inputs_without_the_quality_facets(graph):
    event = emit.openlineage_start(graph, GOLD)

    assert event["eventType"] == "START"
    assert [i["name"] for i in event["inputs"]] == ["silver.events"]
    assert event["outputs"][0]["name"] == "gold.monthly"
    assert "dataQualityMetrics" not in event["outputs"][0]["facets"]


def test_dataset_facets_carry_the_partition_spec(graph):
    facets = emit.dataset_facets(graph, GOLD)
    fields = facets["fathom_partitionSpec"]["fields"]

    assert [f["name"] for f in fields] == ["dt"]
    assert fields[0]["grain"] == "month"
    assert emit.dataset_facets(graph, DatasetId("duckdb", "unspecced")) == {}


def test_marquez_namespaces_are_registered_before_events(graph):
    assert emit.to_marquez_namespaces(graph) == [{"name": "duckdb", "ownerName": "fathom"}]


def test_partition_payload_renders_bindings_as_plain_dicts():
    keys = [KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")]
    assert emit.partition_payload(keys) == [{"dt": "2026-03-14 00:00:00", "region": "eu"}]
