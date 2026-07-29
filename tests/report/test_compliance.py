"""Compliance records generated from lineage rather than maintained."""

from __future__ import annotations

import pytest

from fathom.ai import assets, training
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import ColumnRef, DatasetId, PartitionField, PartitionSpec
from fathom.govern import consent, licenses
from fathom.govern.policy import Label
from fathom.graph import Edge, Graph
from fathom.report import compliance

SCRAPED = DatasetId("s3://eu-west-1-lake", "corpus.scraped")
INTERNAL = DatasetId("s3://eu-west-1-lake", "raw.users")
JOINED = DatasetId("s3://us-east-1-lake", "gold.combined")
MODEL = assets.model("assistant", registry="internal")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))


@pytest.fixture
def graph() -> Graph:
    """Two sources joining into one derived table that then trains a model."""
    g = Graph()
    g.add_dataset(SCRAPED, DAY)
    g.add_dataset(INTERNAL, DAY)
    g.add_dataset(JOINED, DAY)
    g.add_edge(Edge(SCRAPED, JOINED, PartitionMapping.identity(DAY), evidence="sql:1"))
    g.add_edge(Edge(INTERNAL, JOINED, PartitionMapping.identity(DAY), evidence="sql:1"))
    run = training.TrainingRun(model=MODEL, code_version="abc")
    run.add_input(JOINED, snapshot="v1")
    training.record_training_run(g, run)
    return g


# -- compliance ----------------------------------------------------------------


def test_processing_record_states_its_gaps(graph):
    record = compliance.processing_record(graph, JOINED)
    assert not record.is_complete
    assert any("no consent scope" in note for note in record.unknowns)
    assert record.region == "us-east-1"
    assert SCRAPED in record.sources


def test_personal_data_inventory_groups_by_label(graph):
    labels = {
        ColumnRef(INTERNAL, "email"): {Label("pii"), Label("email")},
        ColumnRef(JOINED, "email"): {Label("pii")},
    }
    inventory = compliance.personal_data_inventory(graph, labels)
    assert inventory["pii"] == sorted([INTERNAL, JOINED], key=str)
    assert inventory["email"] == [INTERNAL]


def test_subject_access_report_mentions_the_model(graph):
    text = compliance.subject_access_report(graph, INTERNAL, subject_digest="abc123def456")
    assert "Automated processing" in text
    assert str(MODEL) in text
    assert "not removed by deleting the source records" in text


def test_ai_act_record_reports_undetermined_licences(graph):
    text = compliance.ai_act_record(
        graph,
        MODEL,
        intended_use="fraud triage",
        licenses={SCRAPED: licenses.parse_license("custom")},
    )
    assert "fraud triage" in text
    assert "cannot assert that training was permitted" in text


def test_cross_border_summary_reads_the_identities(graph):
    flows = compliance.cross_border_summary(graph)
    assert "eu-west-1 -> us-east-1" in flows


def test_readiness_is_hard_to_pass(graph):
    empty = compliance.readiness(graph)
    assert not empty.is_ready
    assert any("no labels recorded" in note for note in empty.blockers)
    assert any("consent scopes" in note for note in empty.blockers)

    better = compliance.readiness(
        graph,
        labels={ColumnRef(INTERNAL, "email"): {Label("pii")}},
        consent={INTERNAL: consent.ConsentScope(INTERNAL)},
        licenses={INTERNAL: licenses.parse_license("MIT")},
    )
    assert better.score > empty.score


def test_readiness_blocks_on_a_model_with_no_provenance():
    g = Graph()
    g.add_dataset(MODEL, assets.spec_for(assets.AssetKind.MODEL))
    check = compliance.readiness(g)
    assert any("no recorded training inputs" in note for note in check.blockers)


def test_audit_bundle_is_serializable(graph):
    import json

    bundle = compliance.audit_bundle(graph, labels={ColumnRef(INTERNAL, "email"): {Label("pii")}})
    assert json.loads(json.dumps(bundle))["readiness"]["ready"] is False
    assert str(MODEL) in bundle["models"]


def test_records_for_renders_markdown(graph):
    text = compliance.records_for(graph, [JOINED])
    assert "# Record of processing activities" in text
    assert str(JOINED) in text
