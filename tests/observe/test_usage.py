"""Observed reads, and the refusal to turn their absence into a conclusion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.partitions import PartitionMapping
from fathom.core.types import UNPARTITIONED, DatasetId
from fathom.graph import Edge, Graph
from fathom.observe import usage

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.events")
GOLD = DatasetId("duckdb", "gold.monthly")
DEAD = DatasetId("duckdb", "gold.abandoned")

MARCH = datetime(2026, 3, 14, tzinfo=UTC)
WINDOW = timedelta(days=90)


@pytest.fixture
def graph() -> Graph:
    g = Graph()
    for ds in (RAW, SILVER, GOLD, DEAD):
        g.add_dataset(ds, UNPARTITIONED)
    mapping = PartitionMapping.identity(UNPARTITIONED)
    g.add_edge(Edge(RAW, SILVER, mapping, evidence="sql:1"))
    g.add_edge(Edge(SILVER, GOLD, mapping, evidence="sql:2"))
    g.add_edge(Edge(SILVER, DEAD, mapping, evidence="sql:3"))
    return g


def read(ds: DatasetId, principal: str, *, days: int = 0, kind: str = "query") -> usage.ReadEvent:
    return usage.ReadEvent(ds, principal, MARCH - timedelta(days=days), kind=kind)


# -- aggregation ---------------------------------------------------------------


def test_reads_are_counted_per_dataset():
    stats = usage.summarize([read(GOLD, "ana"), read(GOLD, "ben"), read(RAW, "ana")])
    assert stats[GOLD].reads == 2
    assert stats[GOLD].principals == {"ana", "ben"}
    assert stats[RAW].reads == 1


def test_first_and_last_read_bracket_the_events():
    stats = usage.summarize([read(GOLD, "ana", days=10), read(GOLD, "ana", days=2)])
    assert stats[GOLD].first_read == MARCH - timedelta(days=10)
    assert stats[GOLD].last_read == MARCH - timedelta(days=2)


def test_kinds_are_counted_separately():
    stats = usage.summarize([read(GOLD, "ana"), read(GOLD, "bi", kind="dashboard")])
    assert stats[GOLD].kinds == {"query": 1, "dashboard": 1}


def test_the_window_is_carried_not_inferred_from_the_events():
    """How far back reads happened is not how far back the log reached."""
    stats = usage.summarize([read(GOLD, "ana", days=3)], window=WINDOW)
    assert stats[GOLD].window == WINDOW


def test_age_measures_from_the_last_read():
    stats = usage.summarize([read(GOLD, "ana", days=5)])
    assert stats[GOLD].age(at=MARCH) == timedelta(days=5)


def test_age_without_a_read_is_none():
    assert usage.UsageStats(dataset=GOLD).age(at=MARCH) is None


def test_naive_and_aware_timestamps_mix_without_raising():
    events = [usage.ReadEvent(GOLD, "ana", datetime(2026, 3, 1)), read(GOLD, "ben", days=1)]
    assert usage.summarize(events)[GOLD].reads == 2


# -- what nobody read ----------------------------------------------------------


def test_never_observed_covers_every_dataset_in_the_graph(graph):
    stats = usage.summarize([read(GOLD, "ana")])
    unseen = {s.dataset for s in usage.never_observed(graph, stats)}
    assert unseen == {RAW, SILVER, DEAD}


def test_never_observed_carries_the_window_onto_datasets_with_no_events(graph):
    """A caller cannot report 'unused' without also being handed 'over what period'."""
    found = usage.never_observed(graph, {}, window=WINDOW)
    assert all(s.window == WINDOW for s in found)


def test_a_dataset_with_no_reads_says_so_with_the_window():
    stats = usage.UsageStats(dataset=DEAD, window=WINDOW)
    assert "no reads observed" in stats.summary()
    assert "90 day(s)" in stats.summary()


def test_unread_since_ranks_the_stalest_first():
    stats = usage.summarize([read(GOLD, "ana", days=40), read(RAW, "ana", days=100)])
    found = usage.unread_since(stats, since=MARCH - timedelta(days=30))
    assert [s.dataset for s in found] == [RAW, GOLD]


def test_a_recent_read_is_not_unread_since():
    stats = usage.summarize([read(GOLD, "ana", days=2)])
    assert usage.unread_since(stats, since=MARCH - timedelta(days=30)) == []


# -- retirement candidates -----------------------------------------------------


def test_a_dead_leaf_is_a_candidate(graph):
    stats = usage.summarize([read(GOLD, "ana")], window=WINDOW)
    candidates = {c.dataset for c in usage.retirement_candidates(graph, stats, window=WINDOW)}
    assert DEAD in candidates


def test_a_dataset_feeding_something_read_is_not_a_candidate(graph):
    """Not unused — one hop away from something that is used."""
    stats = usage.summarize([read(GOLD, "ana")], window=WINDOW)
    candidates = {c.dataset for c in usage.retirement_candidates(graph, stats, window=WINDOW)}
    assert RAW not in candidates and SILVER not in candidates


def test_a_read_by_a_scheduler_alone_still_leaves_a_candidate(graph):
    stats = usage.summarize([read(DEAD, "airflow_worker")], window=WINDOW)
    candidates = {c.dataset for c in usage.retirement_candidates(graph, stats, window=WINDOW)}
    assert DEAD in candidates


def test_counting_scheduled_reads_removes_the_candidate(graph):
    stats = usage.summarize([read(DEAD, "airflow_worker")], window=WINDOW)
    candidates = {
        c.dataset
        for c in usage.retirement_candidates(graph, stats, window=WINDOW, ignore_scheduled=False)
    }
    assert DEAD not in candidates


def test_a_candidate_states_the_caveat_in_its_own_text(graph):
    stats = usage.summarize([read(GOLD, "ana")], window=WINDOW)
    candidate = next(
        c for c in usage.retirement_candidates(graph, stats, window=WINDOW) if c.dataset == DEAD
    )
    assert "not\nevidence" in str(candidate) or "not evidence" in str(candidate)
    assert "90 day(s)" in str(candidate)


def test_a_candidate_records_how_many_descendants_were_checked(graph):
    stats = usage.summarize([], window=WINDOW)
    by_dataset = {c.dataset: c for c in usage.retirement_candidates(graph, stats, window=WINDOW)}
    assert by_dataset[RAW].descendants_checked == 3
    assert by_dataset[DEAD].descendants_checked == 0


def test_a_leaf_with_no_descendants_gets_its_own_reason(graph):
    stats = usage.summarize([], window=WINDOW)
    by_dataset = {c.dataset: c for c in usage.retirement_candidates(graph, stats, window=WINDOW)}
    assert "nothing derives from it" in by_dataset[DEAD].reason
    assert "across its downstream" in by_dataset[RAW].reason


# -- human principals ----------------------------------------------------------


def test_scheduled_principals_are_separated_from_people():
    stats = usage.summarize([read(GOLD, "dbt_cloud"), read(GOLD, "ana")])
    assert stats[GOLD].human_principals == {"ana"}


def test_a_dataset_read_only_by_jobs_has_no_human_principals():
    stats = usage.summarize([read(GOLD, "svc_ingest"), read(GOLD, "nightly_job")])
    assert stats[GOLD].human_principals == set()


# -- rollups -------------------------------------------------------------------


def test_busiest_ranks_by_read_count():
    stats = usage.summarize([read(GOLD, "a"), read(GOLD, "b"), read(RAW, "a")])
    assert usage.busiest(stats) == [(GOLD, 2), (RAW, 1)]


def test_busiest_respects_the_limit():
    stats = usage.summarize([read(GOLD, "a"), read(RAW, "a")])
    assert len(usage.busiest(stats, limit=1)) == 1


def test_principals_of_lists_readers_sorted():
    stats = usage.summarize([read(GOLD, "zoe"), read(GOLD, "ana")])
    assert usage.principals_of(stats, GOLD) == ["ana", "zoe"]


def test_principals_of_an_unread_dataset_is_empty():
    assert usage.principals_of({}, GOLD) == []


def test_read_ratio_is_the_covered_fraction(graph):
    stats = usage.summarize([read(GOLD, "ana"), read(RAW, "ana")])
    assert usage.read_ratio(graph, stats) == pytest.approx(0.5)


def test_read_ratio_of_an_empty_graph_is_zero():
    assert usage.read_ratio(Graph(), {}) == 0.0


def test_events_from_builds_events_out_of_log_rows():
    events = usage.events_from([(GOLD, "ana", MARCH)], kind="dashboard")
    assert events[0].dataset == GOLD and events[0].kind == "dashboard"


# -- read counts for the value question ----------------------------------------


def test_read_counts_returns_a_plain_mapping():
    stats = usage.summarize([read(GOLD, "ana"), read(GOLD, "ben"), read(RAW, "ana")])
    assert usage.read_counts(stats) == {GOLD: 2, RAW: 1}


def test_people_only_zeroes_a_dataset_only_a_scheduler_touches():
    """Otherwise an intermediate table looks read by the job that maintains it."""
    stats = usage.summarize([read(GOLD, "airflow_worker"), read(RAW, "ana")])
    assert usage.read_counts(stats, people_only=True) == {GOLD: 0, RAW: 1}


def test_people_only_keeps_a_dataset_with_any_human_reader():
    stats = usage.summarize([read(GOLD, "airflow_worker"), read(GOLD, "ana")])
    assert usage.read_counts(stats, people_only=True)[GOLD] == 2


def test_read_counts_of_nothing_is_empty():
    assert usage.read_counts({}) == {}
