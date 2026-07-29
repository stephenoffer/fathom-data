"""Identities and partition specs for AI assets."""

from __future__ import annotations

import pytest

from fathom.ai import (
    assets,
)
from fathom.core.grains import Grain
from fathom.core.types import DatasetId, PartitionField, PartitionSpec

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.training_set")
DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))

MODEL = assets.model("fraud.scorer", registry="internal")
BASE = assets.model("base.llm", registry="internal")
INDEX = assets.vector_index("docs", store="pgvector")
SPACE = assets.embedding_space("text-embed-3", provider="openai")
CORPUS = assets.corpus("handbook", store="s3")
EVAL = assets.eval_set("fraud.holdout", suite="internal")


# -- assets --------------------------------------------------------------------


def test_asset_identities_and_kinds():
    assert str(MODEL) == "model://internal/fraud.scorer"
    assert assets.kind_of(MODEL) is assets.AssetKind.MODEL
    assert assets.kind_of(RAW) is assets.AssetKind.TABLE
    assert assets.is_model(MODEL) and not assets.is_model(INDEX)
    assert assets.is_vector_index(INDEX) and assets.is_vector_index(SPACE)
    assert assets.is_ai_asset(INDEX) and not assets.is_ai_asset(RAW)
    assert assets.instance_of(MODEL) == "internal"


def test_asset_specs_make_slices_plannable():
    assert assets.spec_for(assets.AssetKind.MODEL).names == ("version",)
    assert assets.spec_for(INDEX).names == ("shard", "dt")
    assert assets.spec_for(RAW).names == ()


def test_parse_ref_round_trips_with_a_version():
    ref = assets.parse_ref("model://internal/fraud.scorer@v3")
    assert ref.dataset == MODEL
    assert ref.version == "v3"
    assert str(ref) == "model://internal/fraud.scorer@v3"
    with pytest.raises(ValueError, match="not an asset reference"):
        assets.parse_ref("fraud.scorer")


def test_kinds_present_inventory():
    assert assets.kinds_present([MODEL, INDEX, RAW]) == {"index": 1, "model": 1, "table": 1}
