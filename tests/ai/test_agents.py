"""What an autonomous program read, wrote, and could leak."""

from __future__ import annotations

import pytest

from fathom.ai import (
    agents,
    assets,
)
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import ColumnRef, DatasetId, PartitionField, PartitionSpec
from fathom.govern.policy import Label
from fathom.graph import Edge, Graph

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.training_set")
DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))

MODEL = assets.model("fraud.scorer", registry="internal")
BASE = assets.model("base.llm", registry="internal")
INDEX = assets.vector_index("docs", store="pgvector")
SPACE = assets.embedding_space("text-embed-3", provider="openai")
CORPUS = assets.corpus("handbook", store="s3")
EVAL = assets.eval_set("fraud.holdout", suite="internal")


# -- agents --------------------------------------------------------------------


@pytest.fixture
def agent_run() -> tuple[Graph, agents.AgentRun]:
    graph = Graph()
    graph.add_dataset(RAW, DAY)
    graph.add_edge(Edge(RAW, GOLD, PartitionMapping.identity(DAY), evidence="sql:1"))

    agent = assets.agent("triage", runtime="lambda")
    run = agents.AgentRun(agent=agent, run_id="run-1")
    run.call(assets.tool("sql"), reads=[GOLD])
    run.call(assets.tool("webhook"), egress=True)
    agents.record_agent_run(graph, run)
    return graph, run


def test_agent_reads_writes_and_reach(agent_run):
    graph, run = agent_run
    assert agents.datasets_read(run) == [GOLD]
    assert agents.datasets_written(run) == []
    assert RAW in agents.reach(graph, run)
    assert len(agents.egress_points(run)) == 1


def test_exfiltration_is_reported_as_possibility_not_fact(agent_run):
    graph, run = agent_run
    labels = {ColumnRef(RAW, "email"): {Label("pii", confidence=0.9)}}
    notes = agents.exfiltration_paths(graph, run, labels)
    assert notes and "in the same run" in notes[0]


def test_first_time_access_and_least_privilege(agent_run):
    graph, run = agent_run
    prior = agents.AgentRun(agent=run.agent, run_id="run-0")
    prior.call(assets.tool("sql"), reads=[RAW])
    assert agents.first_time_access(run, [prior]) == [GOLD]
    assert agents.least_privilege_gap([GOLD, RAW, MODEL], [run, prior]) == [MODEL]


def test_risk_report_assembles_everything(agent_run):
    graph, run = agent_run
    labels = {ColumnRef(RAW, "email"): {Label("pii", confidence=0.9)}}
    report = agents.risk_report(graph, run, labels=labels)
    assert report.egress_calls == 1
    assert report.sensitive_labels["pii"] == [RAW]
    assert not report.is_clean


def test_labels_touched_groups_by_label_over_the_whole_reach(agent_run):
    graph, run = agent_run
    labels = {
        ColumnRef(RAW, "email"): {Label("pii", confidence=0.9), Label("email")},
        ColumnRef(GOLD, "amount"): {Label("monetary_amount")},
    }
    found = agents.labels_touched(graph, run, labels)

    assert found["pii"] == [RAW]  # reached transitively, through gold
    assert found["monetary_amount"] == [GOLD]


def test_tool_call_records_what_it_touched():
    call = agents.ToolCall(tool=assets.tool("sql"), reads=(RAW,), writes=(GOLD,), egress=True)
    assert "1r/1w" in str(call)
    assert "EGRESS" in str(call)
    assert "EGRESS" not in str(agents.ToolCall(tool=assets.tool("sql")))


def test_a_clean_run_says_so(agent_run):
    graph, run = agent_run
    quiet = agents.AgentRun(agent=run.agent, run_id="quiet")
    quiet.call(assets.tool("sql"), reads=[GOLD])
    agents.record_agent_run(graph, quiet)

    report = agents.risk_report(graph, quiet, history=[run, quiet])
    assert report.is_clean
    assert "nothing anomalous" in report.summary()
