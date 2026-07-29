"""When an edge narrowed, and who narrowed it."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping, TimeWindow
from fathom.core.types import DatasetId, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph, history

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.events")
GOLD = DatasetId("duckdb", "gold.monthly")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))

MARCH = datetime(2026, 3, 1, tzinfo=UTC)
APRIL = datetime(2026, 4, 1, tzinfo=UTC)
MAY = datetime(2026, 5, 1, tzinfo=UTC)


def window(lo: int, hi: int) -> PartitionMapping:
    return PartitionMapping.of(dt=TimeWindow("dt", lo, hi, Grain.DAY, Grain.DAY))


def build(*, lo: int = 0, hi: int = 6, with_gold: bool = False) -> Graph:
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(SILVER, DAY)
    g.add_edge(Edge(RAW, SILVER, window(lo, hi), evidence="sql:1"))
    if with_gold:
        g.add_dataset(GOLD, DAY)
        g.add_edge(Edge(SILVER, GOLD, PartitionMapping.identity(DAY), evidence="sql:2"))
    return g


# -- digests -------------------------------------------------------------------


def test_the_same_graph_digests_the_same():
    assert history.graph_digest(build()) == history.graph_digest(build())


def test_a_changed_mapping_changes_the_digest():
    assert history.graph_digest(build(hi=6)) != history.graph_digest(build(hi=1))


def test_edge_insertion_order_does_not_change_the_digest():
    """Insertion order is an artifact of ingest, not a property of the graph."""
    a = Graph()
    a.add_edge(Edge(RAW, SILVER, window(0, 6), evidence="sql:1"))
    a.add_edge(Edge(SILVER, GOLD, window(0, 0), evidence="sql:2"))
    b = Graph()
    b.add_edge(Edge(SILVER, GOLD, window(0, 0), evidence="sql:2"))
    b.add_edge(Edge(RAW, SILVER, window(0, 6), evidence="sql:1"))
    assert history.graph_digest(a) == history.graph_digest(b)


def test_a_changed_spec_changes_the_digest():
    hourly = PartitionSpec.of(PartitionField.time("dt", Grain.HOUR))
    a, b = build(), build()
    b.add_dataset(DatasetId("duckdb", "x"), hourly)
    assert history.graph_digest(a) != history.graph_digest(b)


# -- recording -----------------------------------------------------------------


def test_the_first_revision_has_no_parent():
    log = history.History()
    first = history.record(log, build(), author="ana", note="initial ingest", at=MARCH)
    assert first.is_initial and first.parent is None
    assert first.is_safe


def test_a_second_revision_chains_to_the_first():
    log = history.History()
    before = build()
    history.record(log, before, author="ana", at=MARCH)
    after = build(hi=10)
    second = history.record(log, after, author="ben", at=APRIL, previous=before)
    assert second.parent == log.revisions[0].digest
    assert len(log) == 2


def test_recording_an_unchanged_graph_is_a_no_op():
    """A scheduled ingest that found nothing new must not fill the history with noise."""
    log = history.History()
    graph = build()
    first = history.record(log, graph, author="ana", at=MARCH)
    again = history.record(log, build(), author="ana", at=APRIL, previous=graph)
    assert again is first
    assert len(log) == 1


def test_recording_without_previous_refuses():
    log = history.History()
    history.record(log, build(), author="ana", at=MARCH)
    with pytest.raises(ValueError, match="needs `previous`"):
        history.record(log, build(hi=10), author="ben", at=APRIL)


def test_recording_against_a_stale_graph_refuses():
    """Otherwise one person's change is attributed to the next person to commit."""
    log = history.History()
    first = build()
    history.record(log, first, author="ana", at=MARCH)
    second = build(hi=10)
    history.record(log, second, author="ben", at=APRIL, previous=first)
    with pytest.raises(ValueError, match="misattribute"):
        history.record(log, build(hi=20), author="cal", at=MAY, previous=first)


def test_a_revision_records_its_size():
    log = history.History()
    revision = history.record(log, build(with_gold=True), at=MARCH)
    assert revision.datasets == 3
    assert revision.edges == 2


# -- the incident question -----------------------------------------------------


def test_a_narrowing_is_marked_unsafe():
    log = history.History()
    wide = build(hi=6)
    history.record(log, wide, author="ana", at=MARCH)
    narrow = build(hi=1)
    revision = history.record(
        log, narrow, author="ben", note="perf tuning", at=APRIL, previous=wide
    )
    assert not revision.is_safe
    assert "UNSAFE" in str(revision)


def test_a_widening_stays_safe():
    log = history.History()
    narrow = build(hi=1)
    history.record(log, narrow, author="ana", at=MARCH)
    wide = build(hi=6)
    revision = history.record(log, wide, author="ben", at=APRIL, previous=narrow)
    assert revision.is_safe


def test_narrowings_of_names_the_revision_and_the_author():
    """Six days stopped being invalidated: when did that window shrink, and who shrank it."""
    log = history.History()
    wide = build(hi=6)
    history.record(log, wide, author="ana", at=MARCH)
    narrow = build(hi=1)
    history.record(log, narrow, author="ben", note="perf tuning", at=APRIL, previous=wide)

    (revision, change) = history.narrowings_of(log, RAW, SILVER)[0]
    assert revision.author == "ben"
    assert revision.note == "perf tuning"
    assert revision.at == APRIL
    assert change.narrowed


def test_an_edge_that_never_narrowed_has_no_narrowings():
    log = history.History()
    history.record(log, build(), author="ana", at=MARCH)
    assert history.narrowings_of(log, RAW, SILVER) == []


def test_unsafe_revisions_collects_them_oldest_first():
    log = history.History()
    a = build(hi=6)
    history.record(log, a, author="ana", at=MARCH)
    b = build(hi=3)
    history.record(log, b, author="ben", at=APRIL, previous=a)
    c = build(hi=1)
    history.record(log, c, author="cal", at=MAY, previous=b)
    assert [r.author for r in history.unsafe_revisions(log)] == ["ben", "cal"]


def test_the_initial_revision_is_never_reported_unsafe():
    log = history.History()
    history.record(log, build(), author="ana", at=MARCH)
    assert history.unsafe_revisions(log) == []


def test_revisions_touching_reports_what_each_did():
    log = history.History()
    a = build(hi=6)
    history.record(log, a, author="ana", at=MARCH)
    b = build(hi=1)
    history.record(log, b, author="ben", at=APRIL, previous=a)
    verbs = [verb for _, verb in history.revisions_touching(log, RAW, SILVER)]
    assert "narrowed" in verbs


def test_an_added_edge_is_reported_as_added():
    log = history.History()
    a = build()
    history.record(log, a, author="ana", at=MARCH)
    b = build(with_gold=True)
    history.record(log, b, author="ben", at=APRIL, previous=a)
    assert tuple(v for _, v in history.revisions_touching(log, SILVER, GOLD)) == ("added",)


def test_authors_of_lists_everyone_who_touched_an_edge_in_order():
    log = history.History()
    a = build(hi=6)
    history.record(log, a, author="ana", at=MARCH)
    b = build(hi=3)
    history.record(log, b, author="ben", at=APRIL, previous=a)
    c = build(hi=9)
    history.record(log, c, author="cal", at=MAY, previous=b)
    assert history.authors_of(log, RAW, SILVER) == ["ben", "cal"]


# -- reading the history -------------------------------------------------------


def test_digest_at_returns_the_revision_in_force():
    log = history.History()
    a = build(hi=6)
    first = history.record(log, a, author="ana", at=MARCH)
    b = build(hi=1)
    second = history.record(log, b, author="ben", at=MAY, previous=a)
    assert log.digest_at(APRIL) == first.digest
    assert log.digest_at(MAY) == second.digest


def test_digest_at_before_any_revision_is_none():
    log = history.History()
    history.record(log, build(), at=APRIL)
    assert log.digest_at(MARCH) is None


def test_get_finds_a_revision_by_digest():
    log = history.History()
    revision = history.record(log, build(), at=MARCH)
    assert log.get(revision.digest) is revision
    assert log.get("nope") is None


def test_an_empty_history_summarizes_as_empty():
    assert history.History().summary() == "no revisions recorded"
    assert history.timeline(history.History()) == "no revisions recorded"


def test_the_summary_surfaces_the_unsafe_revisions():
    log = history.History()
    a = build(hi=6)
    history.record(log, a, author="ana", at=MARCH)
    b = build(hi=1)
    history.record(log, b, author="ben", at=APRIL, previous=a)
    assert "1 narrowed or removed an edge" in log.summary()


def test_since_returns_only_later_revisions():
    log = history.History()
    a = build(hi=6)
    history.record(log, a, author="ana", at=MARCH)
    b = build(hi=1)
    history.record(log, b, author="ben", at=MAY, previous=a)
    assert [r.author for r in history.since(log, APRIL)] == ["ben"]


def test_timeline_is_newest_first():
    log = history.History()
    a = build(hi=6)
    history.record(log, a, author="ana", at=MARCH)
    b = build(hi=1)
    history.record(log, b, author="ben", at=APRIL, previous=a)
    assert history.timeline(log).splitlines()[0].endswith("[UNSAFE]")


def test_replay_folds_a_run_into_one_diff():
    log = history.History()
    a = build()
    history.record(log, a, author="ana", at=MARCH)
    b = build(with_gold=True)
    history.record(log, b, author="ben", at=APRIL, previous=a)
    combined = history.replay(log.revisions)
    assert GOLD in combined.added_datasets


def test_replay_cancels_a_dataset_added_then_removed():
    log = history.History()
    a = build()
    history.record(log, a, author="ana", at=MARCH)
    b = build(with_gold=True)
    history.record(log, b, author="ben", at=APRIL, previous=a)
    c = build()
    history.record(log, c, author="cal", at=MAY, previous=b)
    combined = history.replay(log.revisions)
    assert GOLD not in combined.added_datasets
    assert GOLD not in combined.removed_datasets
