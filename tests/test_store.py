"""Persistence round-trips.

The thing to protect against here is silent type coercion. A partition key holding
`datetime(2026, 3, 14)` that comes back as the string `"2026-03-14T00:00:00"` still
prints identically and compares unequal, so every plan built after a reload would
quietly miss.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fathom.grains import Grain
from fathom.graph import Edge, Graph
from fathom.ids import normalize_table
from fathom.partitions import UNBOUNDED, PartitionMapping, Passthrough, TimeWindow
from fathom.profile import ColumnProfile, Profile
from fathom.store import ShadowObservation, Store
from fathom.types import ANY, KeyPredicate, PartitionField, PartitionSpec

RAW = normalize_table("raw.events", system="duckdb")
GOLD = normalize_table("gold.monthly", system="duckdb")
DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
MONTH = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH), PartitionField.value("region"))


@pytest.fixture
def store():
    with Store() as s:
        yield s


def sample_graph() -> Graph:
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(GOLD, MONTH)
    g.add_edge(
        Edge(
            RAW,
            GOLD,
            PartitionMapping.of(
                dt=TimeWindow("dt", 0, 2, Grain.DAY, Grain.MONTH), region=Passthrough("region")
            ),
            columns=(("amount", "revenue"), ("region", "region")),
            evidence="sql:gold.sql",
        )
    )
    return g


def test_graph_round_trips_exactly(store: Store):
    store.save_graph(sample_graph())
    loaded = store.load_graph()

    assert loaded.datasets == sample_graph().datasets
    assert loaded.spec(GOLD) == MONTH
    edge = loaded.edges[0]
    mapping = edge.mapping.get("dt")
    assert isinstance(mapping, TimeWindow)
    assert (mapping.lo, mapping.hi, mapping.in_grain, mapping.out_grain) == (
        0,
        2,
        Grain.DAY,
        Grain.MONTH,
    )
    assert edge.mapping.get("region") == Passthrough("region")
    assert edge.columns == (("amount", "revenue"), ("region", "region"))


def test_reloaded_graph_plans_identically(store: Store):
    """The property that actually matters: a reloaded graph is the same planner."""
    original = sample_graph()
    store.save_graph(original)
    reloaded = store.load_graph()

    seeds = {RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")]}
    assert original.invalidate(seeds).dirty == reloaded.invalidate(seeds).dirty


def test_saving_twice_is_idempotent(store: Store):
    store.save_graph(sample_graph())
    store.save_graph(sample_graph())
    assert len(store.load_graph().edges) == 1


def test_unbounded_mapping_survives(store: Store):
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(GOLD, MONTH)
    g.add_edge(Edge(RAW, GOLD, PartitionMapping.unknown(MONTH), evidence="opaque"))
    store.save_graph(g)
    assert store.load_graph().edges[0].mapping.get("dt") is UNBOUNDED


def test_datetime_partition_values_keep_their_type(store: Store):
    key = KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
    store.save_profile(Profile(dataset=RAW, partition=key, row_count=5))
    got = store.latest_profile(RAW, key)
    assert got is not None
    assert isinstance(got.partition.get("dt"), datetime)
    assert got.partition == key


def test_any_bindings_survive(store: Store):
    key = KeyPredicate.of(dt=ANY, region="eu")
    store.save_profile(Profile(dataset=RAW, partition=key, row_count=1))
    got = store.latest_profile(RAW, key)
    assert got is not None and got.partition.get("dt") is ANY


def test_profile_history_is_newest_first(store: Store):
    key = KeyPredicate.of(dt=datetime(2026, 3, 14))
    for i, day in enumerate([1, 2, 3]):
        store.save_profile(
            Profile(dataset=RAW, partition=key, row_count=i),
            captured=datetime(2026, 4, day, tzinfo=UTC),
        )
    history = store.profile_history(RAW, key)
    assert [p.row_count for p in history] == [2, 1, 0]
    assert store.latest_profile(RAW, key).row_count == 2


def test_column_profiles_round_trip(store: Store):
    profile = Profile(
        dataset=RAW,
        row_count=10,
        columns=(ColumnProfile(name="amount", dtype="double", row_count=10, null_count=2),),
    )
    store.save_profile(profile)
    got = store.latest_profile(RAW)
    assert got is not None
    column = got.column("amount")
    assert column is not None and column.null_count == 2


def test_tokens_round_trip(store: Store):
    assert store.get_token(RAW, "delta") is None
    store.set_token(RAW, "delta", "17")
    store.set_token(RAW, "local", "2026-03-14T00:00:00")
    assert store.get_token(RAW, "delta") == "17"
    assert store.get_token(RAW, "local") == "2026-03-14T00:00:00"
    store.set_token(RAW, "delta", "18")
    assert store.get_token(RAW, "delta") == "18"


def test_confirmed_labels_survive_reinference(store: Store):
    """A human decision must not be undone by the next inference run."""
    store.set_label(RAW, "email", "pii", confidence=0.9, origin="inferred", confirmed=True)
    store.set_label(RAW, "email", "pii", confidence=0.4, origin="inferred", confirmed=False)
    labels = store.labels_for(RAW)
    assert labels["email"] == [("pii", 0.4, True)]


def test_shadow_summary_aggregates(store: Store):
    for missed in (0, 0, 1):
        store.record_shadow(
            ShadowObservation(
                dataset=GOLD,
                observed=datetime(2026, 4, 1, tzinfo=UTC),
                planned=2,
                actual=2,
                missed=missed,
                total=10,
            )
        )
    summary = store.shadow_summary()
    assert summary["runs"] == 3
    assert summary["missed"] == 1
    assert summary["savings"] == pytest.approx(1 - 6 / 30)


def test_a_newer_schema_is_refused_rather_than_corrupted(tmp_path):
    path = tmp_path / "fathom.db"
    Store(path).close()
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="newer fathom"):
        Store(path)


def test_store_persists_across_reopen(tmp_path):
    path = tmp_path / "fathom.db"
    with Store(path) as s:
        s.save_graph(sample_graph())
        s.set_token(RAW, "delta", "7")
    with Store(path) as s:
        assert len(s.load_graph().edges) == 1
        assert s.get_token(RAW, "delta") == "7"
