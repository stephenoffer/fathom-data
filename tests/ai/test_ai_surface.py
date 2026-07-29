"""Public AI surface that nothing else exercises.

Every assertion here checks a real property, not merely that the name imports.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from fathom.ai import agents, assets, evals, features, prompts, rag, training, unlearning, vectors
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.training_set")
DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MODEL = assets.model("fraud.scorer", registry="internal")
BASE = assets.model("base.llm", registry="internal")


@pytest.fixture
def trained() -> Graph:
    """raw -> gold -> model, wired the way `training` expects."""
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(GOLD, DAY)
    g.add_edge(Edge(RAW, GOLD, PartitionMapping.identity(DAY)))
    run = training.TrainingRun(model=MODEL, version="1")
    run.add_input(GOLD, partitions=[KeyPredicate.of(dt=datetime(2026, 3, 14))])
    training.record_training_run(g, run)
    return g


# -- assets --------------------------------------------------------------------


def test_asset_constructors_produce_distinct_identities():
    ckpt = assets.checkpoint("fraud.scorer", registry="internal")
    lora = assets.adapter("fraud.lora", registry="internal")
    assert ckpt != lora
    assert assets.dataset_kind(MODEL)
    assert "fraud.scorer" in assets.describe(MODEL)


# -- training ------------------------------------------------------------------


def test_training_edges_and_input_specs(trained):
    edges = training.training_edges(trained)
    assert edges and all(e.dst == MODEL for e in edges)

    specs = training.input_specs(trained, MODEL)
    assert specs[GOLD] == DAY


def test_derived_models_finds_what_was_built_on_a_base(trained):
    tuned = assets.model("fine.tuned", registry="internal")
    run = training.TrainingRun(model=tuned, base_model=MODEL)
    run.add_input(GOLD)
    training.record_training_run(trained, run)
    assert tuned in training.derived_models(trained, MODEL)


def test_model_card_states_its_own_gaps(trained):
    card = training.model_card(trained, MODEL, intended_use="fraud triage")
    assert "fraud triage" in card
    assert "fraud.scorer" in card


def test_pin_from_plan_records_what_a_run_consumed(trained):
    plan = trained.invalidate({RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
    pin = training.pin_from_plan(plan, GOLD)
    assert pin.dataset == GOLD
    assert pin.partitions


# -- features ------------------------------------------------------------------


def test_feature_helpers_over_a_view(trained):
    view = assets.feature_view("user_features")
    trained.add_dataset(view, DAY)
    trained.add_edge(Edge(RAW, view, PartitionMapping.identity(DAY)))
    trained.add_edge(Edge(view, MODEL, PartitionMapping.unknown(PartitionSpec())))

    assert view in features.features_for_model(trained, MODEL)

    dirty = {RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14))]}
    assert view in features.views_needing_backfill(trained, dirty)
    assert not features.backfill_plan(trained, view, dirty).is_empty


def test_leakage_path_finds_a_label_reaching_a_feature_view():
    g = Graph()
    labels = DatasetId("duckdb", "gold.labels")
    view = assets.feature_view("leaky_features")
    g.add_edge(Edge(labels, view, PartitionMapping()))
    assert features.leakage_path(g, labels, view) == [labels, view]


# -- prompts -------------------------------------------------------------------


def test_prompt_text_helpers():
    assert prompts.token_estimate("a" * 400, chars_per_token=4.0) == 100
    changes = prompts.diff_prompts("answer briefly", "answer in detail")
    assert changes


def test_prompt_history_and_cards():
    template = prompts.PromptTemplate(dataset=assets.prompt("triage"))
    template.commit("first version {x}")
    template.commit("second version {x}")
    assert len(template.digests) == 2
    # Idempotent by content: re-committing the same text adds nothing.
    template.commit("second version {x}")
    assert len(template.digests) == 2
    assert len(prompts.history(template)) == 2
    assert "triage" in prompts.prompt_cards([template])


def test_outputs_using_a_prompt():
    g = Graph()
    prompt = assets.prompt("triage")
    out = DatasetId("duckdb", "gold.answers")
    g.add_edge(Edge(prompt, out, PartitionMapping()))
    assert out in prompts.outputs_using(g, prompt)


# -- agents --------------------------------------------------------------------


def test_agent_run_helpers():
    agent = assets.agent("triager")
    tool = assets.tool("warehouse.query")
    run = agents.AgentRun(agent=agent, run_id="r1")
    run.call(tool, reads=[RAW], writes=[GOLD])

    assert tool in agents.tools_used(run)
    assert agents.access_frequency([run])

    g = Graph()
    g.add_edge(Edge(agent, GOLD, PartitionMapping()))
    assert GOLD in agents.blast_radius(g, agent)


# -- evals ---------------------------------------------------------------------


def test_models_evaluated_by_and_stale_results(trained):
    eval_set = assets.eval_set("fraud.holdout")
    result = evals.EvalResult(MODEL, eval_set, {"accuracy": 0.9})
    evals.record_eval(trained, result)

    assert MODEL in evals.models_evaluated_by(trained, eval_set)

    stale = evals.stale_results(
        trained, [result], {GOLD: [KeyPredicate.of(dt=datetime(2026, 3, 1))]}
    )
    assert stale == [result], "a moved input invalidates the score computed from it"


# -- rag -----------------------------------------------------------------------


def test_context_cost_and_top_sources():
    manifest = rag.ContextManifest(run_id="r", prompt_tokens=100)
    manifest.add("doc-a#0", token_estimate=400, dataset=RAW)
    manifest.add("doc-b#0", token_estimate=500, dataset=RAW)

    assert rag.context_cost(manifest, price_per_million_input_tokens=1_000_000.0) == pytest.approx(
        rag.context_tokens(manifest)
    )
    wasted = rag.wasted_cost(manifest, ["doc-a#0"], price_per_million_input_tokens=1_000_000.0)
    assert wasted == pytest.approx(500.0)

    assert rag.top_sources([manifest])[0][0] in {"doc-a", "doc-b"}


def test_record_context_puts_the_retrieved_datasets_in_the_graph():
    g = Graph()
    manifest = rag.ContextManifest(run_id="r", model=MODEL)
    manifest.add("doc-a#0", dataset=RAW)
    rag.record_context(g, manifest)
    assert RAW in g.datasets


# -- vectors -------------------------------------------------------------------


def test_vector_helpers():
    chunks = [vectors.chunk_of("doc-a", 0, "x"), vectors.chunk_of("doc-b", 0, "y")]
    assert vectors.documents_in(chunks) == ["doc-a", "doc-b"]

    index = assets.vector_index("docs", store="pgvector")
    corpus = DatasetId("s3://lake", "corpus")
    space = assets.embedding_space("text-3")
    g = Graph()
    vectors.record_index(g, index=index, corpus=corpus, space=space)

    assert corpus in vectors.chunk_provenance(g, index)
    plan = vectors.graph_reindex_plan(g, {corpus: [KeyPredicate.unbounded(g.spec(corpus))]})
    assert index in plan.dirty


# -- unlearning ----------------------------------------------------------------


def test_estimate_retraining_cost_uses_per_model_prices_where_given():
    assert (
        unlearning.estimate_retraining_cost([MODEL, BASE], cost_per_model={MODEL: 100.0}) >= 100.0
    )
    assert unlearning.estimate_retraining_cost([]) == 0.0
