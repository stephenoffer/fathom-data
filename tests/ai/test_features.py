"""Feature views: target leakage, skew, and staleness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fathom.ai import (
    assets,
    features,
)
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import ColumnRef, DatasetId, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph
from fathom.observe.profile import ColumnProfile, Profile

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.training_set")
DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))

MODEL = assets.model("fraud.scorer", registry="internal")
BASE = assets.model("base.llm", registry="internal")
INDEX = assets.vector_index("docs", store="pgvector")
SPACE = assets.embedding_space("text-embed-3", provider="openai")
CORPUS = assets.corpus("handbook", store="s3")
EVAL = assets.eval_set("fraud.holdout", suite="internal")


# -- features ------------------------------------------------------------------


def test_feature_view_wiring_and_staleness():
    view_id = assets.feature_view("user.activity", store="feast")
    view = features.FeatureView(
        dataset=view_id,
        entity="user_id",
        features=("logins_7d",),
        sources=[GOLD],
        ttl=timedelta(hours=1),
        last_materialized=datetime(2026, 3, 14, tzinfo=UTC),
    )
    graph = Graph()
    graph.add_dataset(GOLD, DAY)
    features.record_feature_view(graph, view)

    assert GOLD in features.feature_dependencies(graph, view_id)
    now = datetime(2026, 3, 14, 5, tzinfo=UTC)
    assert features.is_stale(view, now=now)
    assert features.freshness_age(view, now=now) == timedelta(hours=5)
    assert features.stale_views([view], now=now) == [view]


def test_a_view_without_a_ttl_is_never_stale():
    view = features.FeatureView(dataset=assets.feature_view("x"), last_materialized=None)
    assert not features.is_stale(view)


def test_leaky_features_follow_column_lineage():
    view_id = assets.feature_view("user.activity")
    graph = Graph()
    graph.add_edge(
        Edge(GOLD, view_id, PartitionMapping(), columns=(("is_fraud", "fraud_rate_7d"),))
    )
    leaks = features.leaky_features(graph, view_id, ColumnRef(GOLD, "is_fraud"))
    assert leaks == [ColumnRef(view_id, "fraud_rate_7d")]
    assert features.label_reaches_features(graph, GOLD, view_id)


def test_serving_risks_escalate_when_a_model_consumes_the_view():
    view_id = assets.feature_view("user.activity")
    view = features.FeatureView(
        dataset=view_id,
        sources=[GOLD],
        ttl=timedelta(hours=1),
        last_materialized=datetime(2026, 3, 14, tzinfo=UTC),
    )
    graph = Graph()
    features.record_feature_view(graph, view)
    graph.add_edge(Edge(view_id, MODEL, PartitionMapping(), evidence="training"))
    risks = features.serving_risks(graph, [view], now=datetime(2026, 3, 20, tzinfo=UTC))
    assert any(note.startswith("[error]") for note in risks)


def test_training_serving_skew_uses_the_drift_comparison():
    offline = Profile(
        dataset=GOLD,
        row_count=2000,
        columns=(ColumnProfile("score", "double", row_count=2000, null_count=0),),
    )
    online = Profile(
        dataset=GOLD,
        row_count=2000,
        columns=(ColumnProfile("score", "double", row_count=2000, null_count=1000),),
    )
    assert features.training_serving_skew(offline, online)
    assert not features.training_serving_skew(offline, offline)


def test_point_in_time_violations_are_structural():
    view = features.FeatureView(dataset=assets.feature_view("x"))
    findings = features.point_in_time_violations(view)
    assert any("point-in-time join" in note for note in findings)
    assert any("entity key" in note for note in findings)


def test_models_using_is_the_blast_radius_of_a_view():
    view_id = assets.feature_view("user.activity")
    graph = Graph()
    graph.add_edge(Edge(view_id, MODEL, PartitionMapping(), evidence="training"))

    assert features.models_using(graph, view_id) == [MODEL]
    assert features.models_using(graph, MODEL) == []
