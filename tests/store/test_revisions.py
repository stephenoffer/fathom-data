"""A revision history that survives the process, which is the only kind that helps.

Every question a history answers is asked weeks after the change. These tests care
about the chain surviving a reopen, the idempotence a replayed ingest depends on, and
the one query an incident actually runs.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping, TimeWindow
from fathom.core.types import DatasetId, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph, history
from fathom.store.sqlite import Store

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.events")
GOLD = DatasetId("duckdb", "gold.monthly")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MARCH = datetime(2026, 3, 1, tzinfo=UTC)
APRIL = datetime(2026, 4, 1, tzinfo=UTC)


def build(*, hi: int = 6, with_gold: bool = False) -> Graph:
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(SILVER, DAY)
    g.add_edge(
        Edge(
            RAW,
            SILVER,
            PartitionMapping.of(dt=TimeWindow("dt", 0, hi, Grain.DAY, Grain.DAY)),
            evidence="sql:1",
        )
    )
    if with_gold:
        g.add_dataset(GOLD, DAY)
        g.add_edge(Edge(SILVER, GOLD, PartitionMapping.identity(DAY), evidence="sql:2"))
    return g


@pytest.fixture
def store():
    with Store(":memory:") as s:
        yield s


def narrowed_pair(log: history.History) -> None:
    wide = build(hi=6)
    history.record(log, wide, author="ana", note="initial", at=MARCH)
    history.record(log, build(hi=1), author="ben", note="perf tuning", at=APRIL, previous=wide)


# -- round trip ----------------------------------------------------------------


def test_a_revision_round_trips(store):
    log = history.History()
    revision = history.record(log, build(), author="ana", note="initial ingest", at=MARCH)
    store.record_revision(revision)

    (found,) = store.revisions()
    assert found["digest"] == revision.digest
    assert found["author"] == "ana"
    assert found["note"] == "initial ingest"
    assert found["parent"] is None
    assert found["is_safe"]


def test_revisions_come_back_oldest_first(store):
    log = history.History()
    narrowed_pair(log)
    for revision in log:
        store.record_revision(revision)
    assert [r["author"] for r in store.revisions()] == ["ana", "ben"]


def test_the_chain_survives_a_reopen(tmp_path):
    log = history.History()
    narrowed_pair(log)
    path = tmp_path / "fathom.db"
    with Store(path) as first:
        for revision in log:
            first.record_revision(revision)
    with Store(path) as second:
        assert len(second.revisions()) == 2
        assert second.head_revision()["author"] == "ben"


def test_recording_the_same_revision_twice_is_a_no_op(store):
    """A replayed ingest must not fork the chain."""
    log = history.History()
    revision = history.record(log, build(), author="ana", at=MARCH)
    store.record_revision(revision)
    store.record_revision(revision)
    assert len(store.revisions()) == 1


def test_the_parent_links_the_chain(store):
    log = history.History()
    narrowed_pair(log)
    for revision in log:
        store.record_revision(revision)
    first, second = store.revisions()
    assert second["parent"] == first["digest"]


def test_the_size_of_the_graph_is_recorded(store):
    log = history.History()
    store.record_revision(history.record(log, build(with_gold=True), at=MARCH))
    (found,) = store.revisions()
    assert (found["datasets"], found["edges"]) == (3, 2)


def test_head_of_an_empty_history_is_none(store):
    assert store.head_revision() is None
    assert store.revisions() == []


# -- the incident question -----------------------------------------------------


def test_a_narrowing_is_stored_as_unsafe(store):
    log = history.History()
    narrowed_pair(log)
    for revision in log:
        store.record_revision(revision)
    assert [r["author"] for r in store.unsafe_revisions()] == ["ben"]


def test_the_initial_revision_is_never_reported_unsafe(store):
    log = history.History()
    store.record_revision(history.record(log, build(), author="ana", at=MARCH))
    assert store.unsafe_revisions() == []


def test_edge_changes_answer_who_narrowed_it_and_when(store):
    """Six days stopped being invalidated: when did that window shrink, and who."""
    log = history.History()
    narrowed_pair(log)
    for revision in log:
        store.record_revision(revision)

    (change,) = store.edge_changes(RAW, SILVER, verb="narrowed")
    assert change["author"] == "ben"
    assert change["note"] == "perf tuning"
    assert change["at"] == APRIL


def test_edge_changes_without_a_verb_returns_everything_touching_it(store):
    log = history.History()
    narrowed_pair(log)
    for revision in log:
        store.record_revision(revision)
    verbs = {c["verb"] for c in store.edge_changes(RAW, SILVER)}
    assert "narrowed" in verbs


def test_an_added_edge_is_recorded_as_added(store):
    log = history.History()
    first = build()
    history.record(log, first, author="ana", at=MARCH)
    history.record(log, build(with_gold=True), author="ben", at=APRIL, previous=first)
    for revision in log:
        store.record_revision(revision)

    (change,) = store.edge_changes(SILVER, GOLD)
    assert change["verb"] == "added"
    assert change["author"] == "ben"


def test_a_removed_edge_is_recorded_as_removed(store):
    log = history.History()
    first = build(with_gold=True)
    history.record(log, first, author="ana", at=MARCH)
    history.record(log, build(), author="ben", at=APRIL, previous=first)
    for revision in log:
        store.record_revision(revision)

    (change,) = store.edge_changes(SILVER, GOLD)
    assert change["verb"] == "removed"


def test_a_widening_is_recorded_and_stays_safe(store):
    log = history.History()
    narrow = build(hi=1)
    history.record(log, narrow, author="ana", at=MARCH)
    history.record(log, build(hi=9), author="ben", at=APRIL, previous=narrow)
    for revision in log:
        store.record_revision(revision)

    (change,) = store.edge_changes(RAW, SILVER, verb="widened")
    assert change["author"] == "ben"
    assert store.unsafe_revisions() == []


def test_an_untouched_edge_has_no_changes(store):
    log = history.History()
    store.record_revision(history.record(log, build(with_gold=True), at=MARCH))
    assert store.edge_changes(SILVER, GOLD) == []


def test_the_digest_lets_a_reader_verify_a_graph_they_still_hold(store):
    """What the store keeps instead of a snapshot, and what an audit needs."""
    log = history.History()
    graph = build()
    revision = history.record(log, graph, author="ana", at=MARCH)
    store.record_revision(revision)

    assert store.revisions()[0]["digest"] == history.graph_digest(graph)
    assert store.revisions()[0]["digest"] != history.graph_digest(build(hi=1))
