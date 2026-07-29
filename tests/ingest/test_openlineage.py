"""OpenLineage ingest.

These events are shaped the way real producers emit them, including the parts
producers disagree about: file format, which facets are present, and how many
events a single run generates.
"""

from __future__ import annotations

import json

import pytest

from fathom.core.grains import Grain
from fathom.core.ids import AliasRegistry
from fathom.core.partitions import TimeWindow
from fathom.core.types import DatasetId, PartitionField, PartitionSpec
from fathom.ingest import ingest_openlineage, load_events, parse_event, read_events

SILVER = DatasetId("s3://lake", "silver/events")
GOLD = DatasetId("s3://lake", "gold/monthly")


def event(
    *,
    run_id: str = "r1",
    event_type: str = "COMPLETE",
    inputs=((SILVER.namespace, SILVER.name),),
    outputs=((GOLD.namespace, GOLD.name),),
    column_lineage: bool = True,
    symlinks=(),
    producer: str = "spark",
) -> dict:
    facets: dict = {}
    if column_lineage:
        facets["columnLineage"] = {
            "fields": {
                "revenue": {
                    "inputFields": [
                        {"namespace": SILVER.namespace, "name": SILVER.name, "field": "amount"}
                    ]
                }
            }
        }
    if symlinks:
        facets["symlinks"] = {
            "identifiers": [{"namespace": ns, "name": name} for ns, name in symlinks]
        }
    return {
        "eventType": event_type,
        "eventTime": "2026-03-14T00:00:00Z",
        "producer": producer,
        "run": {"runId": run_id},
        "job": {"namespace": "spark", "name": "gold_monthly"},
        "inputs": [{"namespace": ns, "name": name} for ns, name in inputs],
        "outputs": [{"namespace": ns, "name": name, "facets": facets} for ns, name in outputs],
    }


# -- parsing -------------------------------------------------------------------


def test_parses_a_complete_event():
    parsed = parse_event(event())
    assert parsed is not None
    assert parsed.inputs == [SILVER]
    assert parsed.outputs == [GOLD]
    assert parsed.column_edges[(SILVER, GOLD)] == [("amount", "revenue")]


def test_events_without_lineage_are_dropped():
    assert parse_event(event(inputs=())) is None
    assert parse_event({"run": {"runId": "r"}}) is None


def test_events_without_a_run_id_are_dropped():
    assert parse_event(event() | {"run": {}}) is None


@pytest.mark.parametrize(
    "body",
    [
        json.dumps([event()]),  # array
        json.dumps(event()),  # single object
        "\n".join([json.dumps(event()), json.dumps(event(run_id="r2"))]),  # ndjson
    ],
)
def test_all_three_file_formats_parse(body):
    """Producers disagree, and a parse failure reads to users as "unsupported tool"."""
    assert list(read_events(body))


def test_a_truncated_final_line_is_skipped_not_fatal():
    """Reading a live event log means catching it mid-write."""
    body = json.dumps(event()) + "\n" + '{"eventType": "COMP'
    assert len(list(read_events(body))) == 1


def test_empty_input_is_empty_output():
    assert list(read_events("   ")) == []


# -- graph building ------------------------------------------------------------


def test_builds_an_edge_with_column_detail():
    result = ingest_openlineage([event()])
    edge = result.graph.edges[0]
    assert (edge.src, edge.dst) == (SILVER, GOLD)
    assert edge.columns == (("amount", "revenue"),)
    assert edge.evidence == "openlineage:spark/gold_monthly"


def test_missing_column_facet_yields_a_dataset_edge_and_says_so():
    result = ingest_openlineage([event(column_lineage=False)])
    assert result.graph.edges[0].columns == ()
    assert any("no columnLineage facet" in n for n in result.notes)


def test_declared_specs_produce_a_real_partition_mapping():
    """OpenLineage carries no partition information, so specs must come from us."""
    specs = {
        SILVER: PartitionSpec.of(PartitionField.time("dt", Grain.DAY)),
        GOLD: PartitionSpec.of(PartitionField.time("dt", Grain.MONTH)),
    }
    result = ingest_openlineage([event()], specs=specs)
    mapping = result.graph.edges[0].mapping.get("dt")
    assert isinstance(mapping, TimeWindow)
    assert (mapping.in_grain, mapping.out_grain) == (Grain.DAY, Grain.MONTH)


def test_without_specs_mappings_are_unbounded():
    result = ingest_openlineage([event()])
    assert result.graph.edges[0].mapping.is_unbounded or not result.graph.edges[0].mapping.fields


def test_one_run_emitting_three_events_produces_one_edge():
    """START, RUNNING, and COMPLETE describe the same work."""
    events = [
        event(event_type="START", column_lineage=False),
        event(event_type="RUNNING", column_lineage=False),
        event(event_type="COMPLETE"),
    ]
    result = ingest_openlineage(events)
    assert len(result.graph.edges) == 1
    # The COMPLETE event wins, so its column detail survives.
    assert result.graph.edges[0].columns == (("amount", "revenue"),)


def test_failed_runs_are_skipped_by_default():
    """A job that died halfway wrote something, but not a dependency to plan against."""
    result = ingest_openlineage([event(event_type="FAIL")])
    assert result.graph.edges == []
    assert any("skipped 1 failed run" in n for n in result.notes)


def test_failed_runs_can_be_included_deliberately():
    result = ingest_openlineage([event(event_type="FAIL")], include_failed=True)
    assert len(result.graph.edges) == 1


def test_self_referencing_output_is_not_an_edge_to_itself():
    events = [event(inputs=((GOLD.namespace, GOLD.name),), column_lineage=False)]
    assert ingest_openlineage(events).graph.edges == []


def test_symlinks_become_aliases():
    """The external-Hive-table-on-S3 case ADR 3 otherwise leaves to a declaration."""
    registry = AliasRegistry()
    ingest_openlineage([event(symlinks=(("hive://cluster", "gold.monthly"),))], aliases=registry)
    assert registry.resolve(DatasetId("hive://cluster", "gold.monthly")) == GOLD


def test_multiple_runs_accumulate_into_one_graph():
    other = DatasetId("s3://lake", "platinum/yearly")
    events = [
        event(),
        event(
            run_id="r2",
            inputs=((GOLD.namespace, GOLD.name),),
            outputs=((other.namespace, other.name),),
            column_lineage=False,
        ),
    ]
    graph = ingest_openlineage(events).graph
    assert len(graph.edges) == 2
    assert other in graph.datasets


def test_trailing_slashes_do_not_split_a_dataset_in_two():
    events = [event(), event(run_id="r2", outputs=(("s3://lake/", "/gold/monthly"),))]
    graph = ingest_openlineage(events).graph
    assert GOLD in graph.datasets
    assert len([d for d in graph.datasets if "gold" in d.name]) == 1


# -- loading -------------------------------------------------------------------


def test_loads_a_directory_of_event_files(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps(event()))
    (tmp_path / "b.json").write_text(json.dumps([event(run_id="r2")]))
    (tmp_path / "ignore.txt").write_text("not events")

    found = load_events(str(tmp_path))
    assert len(found) == 2


def test_loads_a_single_file(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event()))
    assert len(load_events(str(path))) == 1
