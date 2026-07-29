"""Embeddings and indexes: re-embedding only what changed."""

from __future__ import annotations

import pytest

from fathom.ai import (
    assets,
    vectors,
)
from fathom.core.grains import Grain
from fathom.core.types import ANY, DatasetId, PartitionField, PartitionSpec
from fathom.graph import Graph

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.training_set")
DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))

MODEL = assets.model("fraud.scorer", registry="internal")
BASE = assets.model("base.llm", registry="internal")
INDEX = assets.vector_index("docs", store="pgvector")
SPACE = assets.embedding_space("text-embed-3", provider="openai")
CORPUS = assets.corpus("handbook", store="s3")
EVAL = assets.eval_set("fraud.holdout", suite="internal")


# -- vectors -------------------------------------------------------------------


def test_chunking_and_digests():
    chunk = vectors_chunk("handbook.md", 0, "the quick brown fox")
    assert chunk.key == "handbook.md#0"
    assert chunk.token_estimate >= 1
    same = vectors_chunk("handbook.md", 0, "the quick brown fox")
    assert chunk.content_digest == same.content_digest


def vectors_chunk(document, ordinal, text, shard=""):
    from fathom.ai import vectors

    return vectors.chunk_of(document, ordinal, text, shard=shard)


def test_reindex_plan_only_touches_changed_content():
    from fathom.ai import vectors

    current = [
        vectors_chunk("a.md", 0, "unchanged text", shard="s0"),
        vectors_chunk("b.md", 0, "new text", shard="s1"),
    ]
    indexed = {"a.md#0": current[0].content_digest}
    plan = vectors.reindex_plan(INDEX, indexed=indexed, current=current)
    assert [c.key for c in plan.chunks] == ["b.md#0"]
    assert plan.total_chunks == 2
    assert plan.savings == pytest.approx(0.5)
    assert not plan.full_reindex


def test_an_embedding_model_change_forces_a_full_reindex():
    from fathom.ai import vectors

    current = [vectors_chunk("a.md", 0, "text")]
    indexed = {"a.md#0": current[0].content_digest}
    plan = vectors.reindex_plan(
        INDEX, indexed=indexed, current=current, index_version="v1", space_version="v2"
    )
    assert plan.full_reindex
    assert len(plan.chunks) == 1
    assert "not comparable" in plan.reason
    assert vectors.requires_full_reindex("v1", "v2")
    assert not vectors.requires_full_reindex("v1", "v1")


def test_orphan_vectors_survive_a_corpus_deletion():
    from fathom.ai import vectors

    current = [vectors_chunk("a.md", 0, "text")]
    assert vectors.orphan_vectors(["a.md#0", "gone.md#0"], current) == ["gone.md#0"]
    assert vectors.retrievable_after_erasure(["gone.md#0", "a.md#0"], ["gone.md"]) == ["gone.md#0"]


def test_embedding_cost_and_savings():
    from fathom.ai import vectors

    current = [vectors_chunk("a.md", i, "x" * 400) for i in range(10)]
    plan = vectors.reindex_plan(INDEX, indexed={}, current=current)
    numbers = vectors.estimate_savings(plan, price_per_million_tokens=0.13)
    assert numbers["chunks_reembedded"] == 10
    assert numbers["cost_incremental"] > 0
    assert numbers["savings_ratio"] == 0.0  # nothing was skipped


def test_vector_profile_drift():
    from fathom.ai import vectors

    before = vectors.vector_profile(
        INDEX, count=1000, dimensions=1536, mean_norm=1.0, centroid=[0.0, 0.0]
    )
    after = vectors.vector_profile(
        INDEX, count=1000, dimensions=768, mean_norm=1.4, centroid=[1.0, 0.0]
    )
    findings = vectors.embedding_drift(before, after)
    assert any("dimensionality changed" in f for f in findings)
    assert any("centroid moved" in f for f in findings)
    assert vectors.dimension_change(before, after) == (1536, 768)
    assert vectors.centroid_shift(before, after) == pytest.approx(1.0)


def test_record_index_wires_corpus_and_space():
    from fathom.ai import vectors

    graph = Graph()
    vectors.record_index(graph, index=INDEX, corpus=CORPUS, space=SPACE)
    assert vectors.space_of(graph, INDEX) == SPACE
    assert vectors.indexes_for_space(graph, SPACE) == [INDEX]


def test_stale_shards_widen_when_shard_is_unknown():
    from fathom.ai import vectors

    keys = vectors.stale_shards([vectors_chunk("a.md", 0, "x")])
    assert keys[0].get("shard") is ANY


def test_an_unknown_index_version_forces_a_full_reindex():
    """Unknown is unproven, and the safe direction is to re-embed.

    An index built before anyone recorded a version, now facing a named embedding
    model, is the commonest real case. Requiring both versions to be known before
    reporting a mismatch waved it through and left the index holding two embedding
    spaces at once — meaningless neighbours that look entirely healthy.
    """
    assert vectors.requires_full_reindex("", "v2") is True
    assert vectors.requires_full_reindex("v1", "") is True
    assert vectors.requires_full_reindex("v1", "v2") is True
    # No versioning at all is not evidence of a change.
    assert vectors.requires_full_reindex("", "") is False
    assert vectors.requires_full_reindex("v1", "v1") is False


def test_a_reindex_plan_reports_vectors_left_behind_by_deleted_documents():
    """Re-embedding is half a reindex.

    Vectors for documents that no longer exist keep answering questions about them,
    which is how an erasure that stops at the corpus stays visible through retrieval.
    """
    index = DatasetId("pinecone", "docs")
    chunks = [vectors.chunk_of("doc-a", i, f"text {i}") for i in range(4)]
    indexed = {c.key: c.content_digest for c in chunks}

    survivors = chunks[:1]
    plan = vectors.reindex_plan(index, indexed=indexed, current=survivors)

    assert plan.chunks == []  # nothing changed among the survivors
    assert plan.orphans == sorted(c.key for c in chunks[1:])
    assert "orphaned vector" in plan.summary()


def test_chunk_digest_is_content_addressed():
    from fathom.ai import vectors

    assert vectors.chunk_digest("same text") == vectors.chunk_digest("same text")
    assert vectors.chunk_digest("a") != vectors.chunk_digest("b")


def test_chunk_carries_its_key_and_shard():
    from fathom.ai import vectors

    chunk = vectors.Chunk(document="a.md", ordinal=3, shard="s1")
    assert chunk.key == "a.md#3"
    assert str(chunk) == "a.md#3"
    assert chunk.shard == "s1"


def test_stale_chunks_compares_content_not_timestamps():
    """A pipeline that rewrites every document nightly must not pay for all of them."""
    from fathom.ai import vectors

    current = [
        vectors.chunk_of("a.md", 0, "unchanged"),
        vectors.chunk_of("b.md", 0, "edited"),
        vectors.chunk_of("c.md", 0, "brand new"),
    ]
    indexed = {"a.md#0": current[0].content_digest, "b.md#0": "an older digest"}

    assert [c.key for c in vectors.stale_chunks(indexed, current)] == ["b.md#0", "c.md#0"]
    assert vectors.stale_chunks({c.key: c.content_digest for c in current}, current) == []


def test_one_unknown_version_counts_as_a_mismatch():
    """Unknown means unproven, and the safe direction for an index is to re-embed.

    The commonest real case is an index built before anyone recorded a version, now
    facing a named model. Waiving that through leaves the index holding two embedding
    spaces at once, which looks entirely healthy from the outside.
    """
    from fathom.ai import vectors

    assert vectors.version_mismatch("v1", "v2")
    assert vectors.version_mismatch("", "v2")
    assert vectors.version_mismatch("v1", "")
    assert not vectors.version_mismatch("v1", "v1")
    # Two unknowns are a project not using versioning, not evidence of a change.
    assert not vectors.version_mismatch("", "")


def test_embedding_cost_prices_tokens():
    from fathom.ai import vectors

    cost = vectors.estimate_cost(
        [vectors.chunk_of("a.md", i, "x" * 4000) for i in range(10)],
        price_per_million_tokens=0.13,
    )
    assert cost.chunks == 10
    assert cost.tokens == 10_000
    assert cost.cost == pytest.approx(0.0013)
    assert "chunk(s)" in str(cost)


def test_an_empty_cost_is_free_not_an_error():
    from fathom.ai import vectors

    assert vectors.estimate_cost([], price_per_million_tokens=1.0).cost == 0.0


def test_reindex_plan_reports_what_it_skipped():
    from fathom.ai import vectors

    current = [vectors.chunk_of("a.md", i, "text") for i in range(4)]
    plan = vectors.reindex_plan(
        INDEX, indexed={c.key: c.content_digest for c in current[:3]}, current=current
    )
    assert isinstance(plan, vectors.ReindexPlan)
    assert plan.skipped == 3
    assert plan.tokens == plan.chunks[0].token_estimate
