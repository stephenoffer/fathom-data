"""Erasure that reaches the model, and where it stops."""

from __future__ import annotations

import pytest

from fathom.ai import (
    assets,
    unlearning,
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


# -- unlearning ----------------------------------------------------------------


@pytest.fixture
def exposed() -> Graph:
    graph = Graph()
    graph.add_dataset(RAW, DAY)
    graph.add_edge(Edge(RAW, GOLD, PartitionMapping.identity(DAY), evidence="sql:1"))
    graph.add_edge(Edge(GOLD, MODEL, PartitionMapping(), evidence="training:r1"))
    graph.add_edge(Edge(GOLD, INDEX, PartitionMapping(), evidence="embedding:corpus"))
    return graph


def test_exposures_name_the_route(exposed):
    found = {e.asset: e.route for e in unlearning.exposures(exposed, RAW)}
    assert found[MODEL] is unlearning.ExposureRoute.TRAINING
    assert found[INDEX] is unlearning.ExposureRoute.RETRIEVAL
    assert unlearning.models_exposed_to(exposed, RAW) == [MODEL]


def test_deletion_is_not_sufficient_once_a_model_was_trained(exposed):
    assert not unlearning.is_deletion_sufficient(exposed, RAW)
    assert unlearning.retraining_required(exposed, RAW) == [MODEL]


def test_crypto_shredding_changes_the_obligation(exposed):
    plain = {o.asset: o.remediation for o in unlearning.obligations(exposed, RAW)}
    assert plain[MODEL] is unlearning.Remediation.RETRAIN

    shredded = {
        o.asset: o.remediation
        for o in unlearning.obligations(exposed, RAW, encrypted_per_subject=[MODEL])
    }
    assert shredded[MODEL] is unlearning.Remediation.CRYPTO_SHRED


def test_approximate_unlearning_is_never_reported_as_complete(exposed):
    found = {
        o.asset: o
        for o in unlearning.obligations(exposed, RAW, supports_approximate_unlearning=[MODEL])
    }
    assert found[MODEL].remediation is unlearning.Remediation.APPROXIMATE_UNLEARN
    assert not found[MODEL].is_complete_if_done
    assert "does not prove removal" in found[MODEL].note


def test_completeness_statement_leads_with_the_bad_news(exposed):
    text = unlearning.completeness_statement(exposed, RAW, subject_digest="deadbeefcafe")
    assert "**This erasure is not complete.**" in text
    assert "parameters" in text
    assert "backups and snapshots" in text


def test_extend_plan_blocks_the_model_target(exposed):
    from fathom.govern.erasure import ErasureRequest, plan_erasure

    request = ErasureRequest(subject="u1", key_column="user_id", origin=RAW)
    plan = plan_erasure(exposed, request)
    assert "no adapter configured" in dict((t.dataset, t.blocked) for t in plan.targets)[MODEL]

    unlearning.extend_plan(exposed, plan)
    reasons = {t.dataset: t.blocked for t in plan.targets}
    assert "parameters" in reasons[MODEL]
    assert "searchable" in reasons[INDEX]
    assert not plan.is_complete


def test_exposure_summary_counts_routes(exposed):
    assert unlearning.exposure_summary(exposed, RAW) == {"retrieval": 1, "training": 1}


def test_unreachable_copies_says_the_uncomfortable_part_every_time(exposed):
    notes = unlearning.unreachable_copies(exposed, RAW)

    assert any("backups" in note for note in notes)
    assert any("replicas" in note for note in notes)
    assert any("parameters retain information" in note for note in notes)


def test_an_obligation_knows_whether_doing_it_would_be_enough(exposed):
    found = {o.asset: o for o in unlearning.obligations(exposed, RAW)}

    assert found[MODEL].is_complete_if_done  # a retrain genuinely removes them
    assert found[INDEX].is_complete_if_done  # deleting vectors does too
    assert "outstanding" in str(found[MODEL])


def test_approximate_unlearning_is_the_one_that_is_never_enough(exposed):
    found = {
        o.asset: o
        for o in unlearning.obligations(exposed, RAW, supports_approximate_unlearning=[MODEL])
    }
    assert not found[MODEL].is_complete_if_done
