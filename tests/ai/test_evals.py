"""Eval sets, and whether their scores can be believed."""

from __future__ import annotations

import pytest

from fathom.ai import (
    assets,
    evals,
)
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, PartitionField, PartitionSpec
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


# -- evals ---------------------------------------------------------------------


def test_contamination_is_clean_when_lineage_is_separate():
    graph = Graph()
    graph.add_dataset(GOLD, DAY)
    graph.add_edge(Edge(GOLD, MODEL, PartitionMapping(), evidence="training"))
    evals.record_eval(graph, evals.EvalResult(model=MODEL, eval_set=EVAL))
    report = evals.contamination(graph, MODEL, EVAL)
    assert report.severity == "clean"
    assert not report.is_contaminated


def test_contamination_detects_a_shared_lineage_path():
    graph = Graph()
    graph.add_edge(Edge(GOLD, MODEL, PartitionMapping(), evidence="training"))
    graph.add_edge(Edge(GOLD, EVAL, PartitionMapping(), evidence="etl"))
    evals.record_eval(graph, evals.EvalResult(model=MODEL, eval_set=EVAL))
    report = evals.contamination(graph, MODEL, EVAL)
    assert report.severity == "contaminated"
    assert report.paths
    assert evals.is_contaminated(graph, MODEL, EVAL)
    assert (MODEL, EVAL) in evals.contaminated_models(graph)


def test_identifier_overlap_is_measured_against_the_eval_set():
    count, ratio, examples = evals.identifier_overlap(["a", "b", "c"], ["b", "c"])
    assert count == 2
    assert ratio == pytest.approx(1.0)
    assert examples == ["b", "c"]


def test_eval_regressions_respect_metric_direction():
    before = evals.EvalResult(model=MODEL, eval_set=EVAL, metrics={"accuracy": 0.9, "loss": 0.1})
    after = evals.EvalResult(model=MODEL, eval_set=EVAL, metrics={"accuracy": 0.8, "loss": 0.3})
    found = evals.regressions(before, after)
    assert any("accuracy" in line for line in found)
    assert any("loss" in line for line in found)

    improved = evals.EvalResult(
        model=MODEL, eval_set=EVAL, metrics={"accuracy": 0.95, "loss": 0.05}
    )
    assert evals.regressions(before, improved) == []


def test_holdout_integrity_flags_a_model_generated_eval():
    graph = Graph()
    graph.add_edge(Edge(MODEL, EVAL, PartitionMapping(), evidence="synthetic"))
    findings = evals.holdout_integrity(graph, EVAL)
    assert any("cannot measure that model" in note for note in findings)


def test_overlap_ratio_measures_contaminated_eval_rows():
    """Dividing a distinct-id count by a row count mixes units and understates.

    Four of five eval rows being the same leaked record is an 80%-compromised eval
    set. Reporting 20% is the one direction a contamination check must not round.
    """
    count, ratio, _ = evals.identifier_overlap(["a", "b"], ["a", "a", "a", "a", "z"])
    assert count == 1
    assert ratio == pytest.approx(0.8)


def test_changing_the_metric_set_is_not_a_regression():
    """Filling a missing metric with 0.0 invents both a regression and an improvement."""
    model, eval_set = DatasetId("mlflow", "m"), DatasetId("mlflow", "e")
    before = evals.EvalResult(model, eval_set, {"accuracy": 0.91, "f1": 0.88})
    after = evals.EvalResult(model, eval_set, {"accuracy": 0.92})

    assert evals.regressions(before, after) == []
    assert evals.removed_metrics(before, after) == ["f1"]
    assert "f1" not in evals.compare_results(before, after)

    grew = evals.EvalResult(model, eval_set, {"accuracy": 0.92, "f1": 0.88})
    assert evals.added_metrics(after, grew) == ["f1"]


def test_eval_sets_for_finds_what_a_model_was_graded_against():
    graph = Graph()
    evals.record_eval(graph, evals.EvalResult(model=MODEL, eval_set=EVAL))
    assert evals.eval_sets_for(graph, MODEL) == [EVAL]
    assert evals.models_evaluated_by(graph, EVAL) == [MODEL]


def test_shared_ancestry_is_the_weak_signal():
    """Suggestive, common, and frequently benign — hence its own function."""
    graph = Graph()
    graph.add_edge(Edge(GOLD, MODEL, PartitionMapping(), evidence="training"))
    graph.add_edge(Edge(GOLD, EVAL, PartitionMapping(), evidence="etl"))
    assert evals.shared_ancestry(graph, MODEL, EVAL) == [GOLD]

    separate = Graph()
    separate.add_dataset(MODEL)
    separate.add_dataset(EVAL)
    assert evals.shared_ancestry(separate, MODEL, EVAL) == []


def test_leakage_paths_report_either_direction():
    forward = Graph()
    forward.add_edge(Edge(GOLD, EVAL, PartitionMapping(), evidence="etl"))
    assert evals.leakage_paths(forward, GOLD, EVAL) == [[GOLD, EVAL]]

    backward = Graph()
    backward.add_edge(Edge(EVAL, GOLD, PartitionMapping(), evidence="etl"))
    assert evals.leakage_paths(backward, GOLD, EVAL) == [[EVAL, GOLD]]

    assert evals.leakage_paths(Graph(), GOLD, EVAL) == []


def test_audit_evals_covers_every_set_a_model_was_graded_against():
    graph = Graph()
    graph.add_edge(Edge(GOLD, MODEL, PartitionMapping(), evidence="training"))
    graph.add_edge(Edge(GOLD, EVAL, PartitionMapping(), evidence="etl"))
    evals.record_eval(graph, evals.EvalResult(model=MODEL, eval_set=EVAL))

    reports = evals.audit_evals(graph, MODEL)
    assert [r.eval_set for r in reports] == [EVAL]
    assert reports[0].is_contaminated
    assert "CONTAMINATED" in reports[0].summary()
