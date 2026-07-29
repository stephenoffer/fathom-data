"""Rendering the graph out to Mermaid, DOT, JSON, and Markdown."""

from __future__ import annotations

from datetime import datetime

import pytest

from fathom import render
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


# -- render --------------------------------------------------------------------


def test_mermaid_escapes_and_highlights(graph):
    text = render.graph_to_mermaid(graph, highlight=[GOLD])
    assert "flowchart LR" in text
    assert "class n_duckdb_gold_monthly dirty;" in text
    assert "-->" in text


def test_graph_json_round_trips(graph):
    restored = render.graph_from_json(render.graph_to_json(graph))
    assert restored.datasets == graph.datasets
    assert restored.spec(GOLD) == MONTH
    assert {str(e) for e in restored.edges} == {str(e) for e in graph.edges}


def test_dot_and_d2_are_syntactically_plausible(graph):
    assert render.graph_to_dot(graph).startswith("digraph lineage {")
    assert render.graph_to_dot(graph).rstrip().endswith("}")
    assert "->" in render.graph_to_d2(graph)


def test_cytoscape_shape(graph):
    blob = render.graph_to_cytoscape(graph)
    assert len(blob["elements"]["nodes"]) == 5
    assert len(blob["elements"]["edges"]) == 3


def test_tree_marks_repeats_rather_than_expanding(graph):
    text = render.tree(graph, RAW)
    assert text.splitlines()[0] == str(RAW)
    assert "└──" in text or "├──" in text


def test_write_graph_rejects_an_unknown_format(tmp_path, graph):
    with pytest.raises(ValueError, match="unknown format"):
        render.write_graph(tmp_path / "out.txt", graph, format="xml")
    written = render.write_graph(tmp_path / "g.json", graph, format="json")
    assert written.exists()


# -- the rest of the rendering surface -----------------------------------------


def test_every_graph_format_names_every_dataset(graph):
    for text in (
        render.graph_to_plantuml(graph),
        render.graph_to_dot(graph),
        render.graph_to_mermaid(graph),
        render.graph_to_d2(graph),
    ):
        assert "silver" in text and "gold" in text


def test_graph_to_markdown_is_a_table(graph):
    text = render.graph_to_markdown(graph)
    assert "|" in text and "silver.events" in text


def test_plan_renders_to_text_markdown_and_mermaid(graph):
    from datetime import datetime

    from fathom.core.types import KeyPredicate

    plan = graph.invalidate({RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")]})
    text = render.plan_to_text(plan)
    assert "silver.events" in text

    markdown = render.plan_to_markdown(plan)
    assert "|" in markdown and "silver.events" in markdown

    mermaid = render.plan_to_mermaid(graph, plan)
    assert mermaid.startswith("flowchart")
    # The dirty datasets are the point of the diagram.
    assert "classDef dirty" in mermaid


def test_empty_plan_renders_without_pretending_there_is_work(graph):
    empty = graph.invalidate({})
    assert "nothing" in render.plan_to_text(empty).lower()


def test_profile_renders_to_markdown_and_round_trips_through_json():
    import json

    from fathom.observe.profile import ColumnProfile, Profile

    profile = Profile(
        dataset=RAW,
        row_count=1000,
        columns=(ColumnProfile("amount", "double", 1000, 10, 1.0, 99.0),),
    )
    markdown = render.profile_to_markdown(profile)
    assert "amount" in markdown and "1,000" in markdown or "1000" in markdown

    body = json.loads(render.profile_to_json(profile))
    assert body["row_count"] == 1000
    assert body["columns"][0]["name"] == "amount"


def test_findings_and_violations_render_to_markdown():
    from fathom.govern.policy import Violation
    from fathom.observe.profile import Finding, Severity

    findings = [Finding("amount", "null_rate", Severity.WARN, "null rate 1% -> 9%")]
    text = render.findings_to_markdown(findings)
    assert "amount" in text and "null rate" in text

    violations = [Violation(GOLD, "email", "pii", "forbidden label", 0.9, "not cleared")]
    text = render.violations_to_markdown(violations)
    assert "pii" in text and "email" in text


def test_labels_to_markdown_respects_the_confidence_floor():
    from fathom.core.types import ColumnRef
    from fathom.govern.policy import Label

    labels = {
        ColumnRef(RAW, "email"): {Label("pii", 0.9, "inferred")},
        ColumnRef(RAW, "note"): {Label("pii", 0.1, "inferred")},
    }
    text = render.labels_to_markdown(labels, min_confidence=0.5)
    assert "email" in text
    assert "note" not in text


def test_partition_table_renders_keys_in_order():
    from datetime import datetime

    from fathom.core.types import KeyPredicate

    keys = [
        KeyPredicate.of(dt=datetime(2026, 3, 15), region="eu"),
        KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu"),
    ]
    text = render.partition_table(keys)
    assert text.index("2026-03-14") < text.index("2026-03-15")


def test_write_json_and_write_text_create_files(tmp_path):
    target = tmp_path / "out.json"
    render.write_json(target, {"a": 1})
    assert target.exists()
    import json

    assert json.loads(target.read_text())["a"] == 1

    text_target = tmp_path / "out.txt"
    render.write_text(text_target, "hello")
    assert text_target.read_text() == "hello"


def test_shadow_report_renders_its_verdict():
    from fathom.observe.shadow import ShadowReport, compare

    report = ShadowReport(results=[compare(GOLD, planned=[], actual=[], total=0)])
    text = render.shadow_to_markdown(report)
    assert "gold.monthly" in text


def test_plan_to_json_is_consumable_as_a_task_list(graph):
    import json

    from fathom.core.types import KeyPredicate

    plan = graph.invalidate({RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")]})
    blob = json.loads(render.plan_to_json(plan))

    assert blob["order"] == [str(ds) for ds in plan.order]
    entry = blob["datasets"][str(GOLD)]
    assert entry["rendered"]  # human-readable alongside the typed form
    assert entry["partitions"]  # typed, so a consumer can compare keys
    assert entry["widened"] is False


def test_an_empty_plan_renders_as_nothing_to_do(graph):
    plan = graph.invalidate({})
    assert "nothing to rebuild" in render.plan_to_markdown(plan)
    assert "nothing to rebuild" in render.plan_to_text(plan)
