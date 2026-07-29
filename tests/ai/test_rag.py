"""What entered a context window, and what that means for policy."""

from __future__ import annotations

import pytest

from fathom.ai import (
    assets,
    rag,
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


# -- rag -----------------------------------------------------------------------


@pytest.fixture
def context() -> tuple[Graph, rag.ContextManifest]:
    graph = Graph()
    graph.add_dataset(RAW, DAY)
    graph.add_dataset(CORPUS)
    graph.add_edge(Edge(RAW, CORPUS, PartitionMapping(), evidence="etl"))

    manifest = rag.ContextManifest(run_id="req-1", model=MODEL, prompt_tokens=100)
    manifest.add("handbook.md#0", score=0.9, dataset=CORPUS, token_estimate=200)
    manifest.add("handbook.md#4", score=0.7, dataset=CORPUS, token_estimate=150)
    return graph, manifest


def test_context_manifest_basics(context):
    _, manifest = context
    assert rag.sources_of(manifest) == ["handbook.md"]
    assert rag.context_tokens(manifest) == 450
    assert manifest.datasets == [CORPUS]
    assert "2 chunk(s)" in manifest.summary()


def test_context_digest_is_content_addressed(context):
    _, manifest = context
    same = rag.ContextManifest(run_id="req-2", model=MODEL)
    same.add("handbook.md#4")
    same.add("handbook.md#0")
    assert rag.context_digest(manifest) == rag.context_digest(same)


def test_provenance_reaches_the_raw_table(context):
    graph, manifest = context
    assert RAW in rag.provenance(graph, manifest)


def test_labels_reach_the_context_through_the_corpus(context):
    graph, manifest = context
    labels = {ColumnRef(RAW, "email"): {Label("pii", confidence=0.9)}}
    found = rag.labels_in_context(graph, manifest, labels)
    assert found["pii"] == [ColumnRef(RAW, "email")]


def test_context_policy_blocks_pii_reaching_a_sink(context):
    from fathom.govern.policy import SinkPolicy

    graph, manifest = context
    labels = {ColumnRef(RAW, "email"): {Label("pii", confidence=0.9)}}
    sink = DatasetId("https://api.example", "chat")
    violations = rag.enforce_context(graph, manifest, labels, SinkPolicy.no_pii(sink))
    assert len(violations) == 1
    assert violations[0].label == "pii"

    quiet = rag.enforce_context(
        graph,
        manifest,
        {ColumnRef(RAW, "email"): {Label("pii", confidence=0.1)}},
        SinkPolicy.no_pii(sink),
    )
    assert quiet == []


def test_redaction_targets_and_citation_accounting(context):
    graph, manifest = context
    labels = {ColumnRef(RAW, "email"): {Label("pii", confidence=0.9)}}
    assert len(rag.redaction_targets(graph, manifest, labels)) == 2
    assert [r.chunk_key for r in rag.unused_context(manifest, ["handbook.md#0"])] == [
        "handbook.md#4"
    ]
    assert rag.citation_coverage(manifest, ["handbook.md#0"]) == pytest.approx(0.5)
    assert rag.uncited_claims(["made-up#0"], manifest) == ["made-up#0"]


def test_manifest_json_round_trip(context):
    _, manifest = context
    restored = rag.manifest_from_json(rag.manifest_to_json(manifest))
    assert rag.context_digest(restored) == rag.context_digest(manifest)
    assert restored.run_id == "req-1"


def test_retrieval_index_and_never_retrieved(context):
    _, manifest = context
    assert rag.retrieval_index([manifest]) == {"handbook.md": ["req-1"]}
    assert rag.never_retrieved([manifest], ["handbook.md", "unused.md"]) == ["unused.md"]


def test_a_chunk_with_no_dataset_is_reported_not_ignored():
    """ "I could not check" must not render as "nothing forbidden".

    `Retrieval.dataset` defaults to None, so a chunk recorded without it is invisible
    to `provenance` — and a context carrying personal data to a third-party endpoint
    returned zero violations purely because nobody said where the chunk came from.
    """
    from fathom.govern.policy import Label, SinkPolicy

    corpus = DatasetId("s3://lake", "support_tickets")
    sink = DatasetId("openai", "gpt-4o")
    graph = Graph()
    graph.add_dataset(corpus)
    labels = {ColumnRef(corpus, "email"): {Label("pii", 0.95, "inferred")}}

    manifest = rag.ContextManifest(run_id="r", sink=sink)
    manifest.add("ticket-1#0", token_estimate=200)  # no dataset recorded

    assert rag.unattributed_retrievals(manifest)
    violations = rag.enforce_context(graph, manifest, labels, SinkPolicy.no_pii(sink))
    assert [v.label for v in violations] == ["unattributed"]
