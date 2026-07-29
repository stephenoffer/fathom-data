"""Training runs as edges, and what follows from that."""

from __future__ import annotations

from datetime import datetime

import pytest

from fathom.ai import (
    assets,
    training,
)
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, KeyPredicate, PartitionField, PartitionSpec
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


# -- training ------------------------------------------------------------------


@pytest.fixture
def trained() -> tuple[Graph, training.TrainingRun]:
    graph = Graph()
    graph.add_dataset(RAW, DAY)
    graph.add_dataset(GOLD, DAY)
    graph.add_edge(Edge(RAW, GOLD, PartitionMapping.identity(DAY), evidence="sql:1"))

    run = training.TrainingRun(model=MODEL, version="v3", code_version="abc123", run_id="r1")
    run.add_input(GOLD, partitions=[KeyPredicate.of(dt=datetime(2026, 3, 14))], columns=["amount"])
    run.base_model = BASE
    training.record_training_run(graph, run)
    return graph, run


def test_training_run_becomes_edges(trained):
    graph, run = trained
    assert MODEL in graph.datasets
    assert training.training_inputs(graph, MODEL) == sorted([GOLD, BASE], key=str)
    assert training.models_trained_on(graph, RAW) == [MODEL]
    # A model edge is unbounded: no slice of a model comes from one slice of data.
    assert all(e.mapping.is_unbounded for e in graph.in_edges(MODEL))


def test_transitive_inputs_reach_the_raw_source(trained):
    graph, _ = trained
    assert RAW in training.training_inputs(graph, MODEL, transitive=True)


def test_run_digest_ignores_time_and_tracks_data(trained):
    _, run = trained
    same = training.TrainingRun(model=MODEL, version="v3", code_version="abc123", run_id="other")
    same.add_input(GOLD, partitions=[KeyPredicate.of(dt=datetime(2026, 3, 14))], columns=["amount"])
    same.base_model = BASE
    assert training.run_digest(run) == training.run_digest(same)

    same.code_version = "def456"
    assert training.run_digest(run) != training.run_digest(same)


def test_reproducibility_requires_pinned_inputs():
    run = training.TrainingRun(model=MODEL, code_version="abc")
    run.add_input(GOLD)  # no snapshot and no partitions
    assert training.unpinned_inputs(run) == [GOLD]
    assert not training.is_reproducible(run)

    run.inputs.clear()
    run.add_input(GOLD, snapshot="v42")
    assert training.is_reproducible(run)


def test_retraining_plan_reaches_the_model(trained):
    graph, _ = trained
    dirty = {RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14))]}
    assert training.stale_models(graph, dirty) == [MODEL]


def test_bill_of_materials_states_its_own_gaps(trained):
    graph, _ = trained
    bom = training.data_bill_of_materials(graph, MODEL)
    assert GOLD in bom.direct
    assert RAW in bom.transitive
    assert bom.base_models == [BASE]
    assert not bom.is_complete  # the base-model edge carries no column detail


def test_training_data_summary_is_generated(trained):
    graph, _ = trained
    text = training.training_data_summary(graph, MODEL)
    assert "# Training data summary" in text
    assert "duckdb" in text
    assert "Known gaps" in text


def test_compare_runs_isolates_what_changed(trained):
    _, before = trained
    after = training.TrainingRun(model=MODEL, version="v4", code_version="def456")
    after.add_input(GOLD, partitions=[KeyPredicate.of(dt=datetime(2026, 3, 15))])
    result = training.compare_runs(before, after)
    assert result["code_changed"]
    assert result["inputs_changed"] == [str(GOLD)]
    assert not result["same_digest"]


def test_untracked_models_are_named():
    graph = Graph()
    graph.add_dataset(MODEL, assets.spec_for(assets.AssetKind.MODEL))
    assert training.untracked_models(graph) == [MODEL]


def test_input_digest_tracks_the_slice_not_the_dataset():
    """Two runs on different days of the same table are not the same input."""

    def pin(day: int) -> training.InputPin:
        return training.InputPin(
            dataset=GOLD, partitions=frozenset({KeyPredicate.of(dt=datetime(2026, 3, day))})
        )

    assert training.input_digest(pin(14)) == training.input_digest(pin(14))
    assert training.input_digest(pin(14)) != training.input_digest(pin(15))


def test_a_snapshot_is_a_pin_and_a_bare_dataset_is_not():
    assert training.InputPin(dataset=GOLD, snapshot="v42").is_pinned
    assert training.InputPin(dataset=GOLD, partitions=frozenset({KeyPredicate()})).is_pinned
    assert not training.InputPin(dataset=GOLD).is_pinned
    assert "unpinned" in str(training.InputPin(dataset=GOLD))
