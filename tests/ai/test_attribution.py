"""Turning a drift alert into a ranked diagnosis."""

from __future__ import annotations

from fathom.ai import (
    assets,
    attribution,
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


# -- attribution ---------------------------------------------------------------


def test_attribution_ranks_a_drifting_upstream_first():
    graph = Graph()
    graph.add_edge(Edge(RAW, GOLD, PartitionMapping.identity(DAY), evidence="sql:1"))

    clean = Profile(
        dataset=RAW,
        row_count=5000,
        columns=(ColumnProfile("amount", "double", row_count=5000, null_count=0),),
    )
    drifted = Profile(
        dataset=RAW,
        row_count=5000,
        columns=(ColumnProfile("amount", "double", row_count=5000, null_count=2500),),
    )
    diagnosis = attribution.attribute(graph, GOLD, before={RAW: clean}, after={RAW: drifted})
    assert diagnosis.best is not None
    assert diagnosis.best.dataset == RAW
    assert diagnosis.best.has_drift
    assert not diagnosis.unchecked


def test_unprofiled_upstream_is_unchecked_not_clean():
    graph = Graph()
    graph.add_edge(Edge(RAW, GOLD, PartitionMapping.identity(DAY), evidence="sql:1"))
    diagnosis = attribution.attribute(graph, GOLD, before={}, after={})
    assert diagnosis.unchecked
    assert not diagnosis.confirmed
    assert "not profiled" in diagnosis.causes[0].explain()
    assert "incomplete" in attribution.blame_report(diagnosis)


def test_column_attribution_narrows_to_the_column():
    graph = Graph()
    graph.add_edge(
        Edge(RAW, GOLD, PartitionMapping.identity(DAY), columns=(("fx_rate", "revenue"),))
    )
    before = Profile(
        dataset=RAW,
        row_count=5000,
        columns=(ColumnProfile("fx_rate", "double", row_count=5000, null_count=0),),
    )
    after = Profile(
        dataset=RAW,
        row_count=5000,
        columns=(ColumnProfile("fx_rate", "double", row_count=5000, null_count=3000),),
    )
    diagnosis = attribution.attribute_column(
        graph, ColumnRef(GOLD, "revenue"), before={RAW: before}, after={RAW: after}
    )
    assert diagnosis.best is not None
    assert diagnosis.best.column == "fx_rate"
    assert attribution.root_causes(diagnosis)[0].dataset == RAW


def test_suspects_are_the_upstream_candidates():
    graph = Graph()
    graph.add_edge(Edge(RAW, GOLD, PartitionMapping.identity(DAY), evidence="sql:1"))

    assert attribution.suspects(graph, GOLD) == [RAW]
    assert attribution.suspects(graph, RAW) == []


def test_unchecked_names_the_blind_spots():
    graph = Graph()
    graph.add_edge(Edge(RAW, GOLD, PartitionMapping.identity(DAY), evidence="sql:1"))

    assert attribution.unchecked(graph, GOLD, {}) == [RAW]
    profiled = Profile(dataset=RAW, row_count=1, columns=())
    assert attribution.unchecked(graph, GOLD, {RAW: profiled}) == []
