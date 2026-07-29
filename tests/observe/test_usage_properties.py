"""Properties of usage, where the failure is a conclusion the data cannot support.

Everything here is one-directional. Reporting a dataset as read when it is not costs
a wasted look; reporting one as unread when it is not is how a table read once a year
for a filing gets switched off. So:

- a retirement candidate always has an unread downstream cone, with no exceptions
- aggregation is order-independent and its counts are exact
- the observation window is carried, never inferred from the events
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from fathom.core.partitions import PartitionMapping
from fathom.core.types import UNPARTITIONED, DatasetId
from fathom.graph import Edge, Graph
from fathom.graph.query import descendants
from fathom.observe import usage

MARCH = datetime(2026, 3, 14, tzinfo=UTC)
WINDOW = timedelta(days=90)

names = st.sampled_from(["a", "b", "c", "d"])
principals = st.sampled_from(["ana", "ben", "airflow_worker", "svc_ingest", "dbt_cloud"])
events = st.lists(
    st.tuples(names, principals, st.integers(min_value=0, max_value=200)), max_size=25
)
edges = st.lists(st.tuples(names, names), max_size=8)


def table(name: str) -> DatasetId:
    return DatasetId("duckdb", f"t.{name}")


def to_events(raw) -> list[usage.ReadEvent]:
    return [
        usage.ReadEvent(table(name), principal, MARCH - timedelta(days=days))
        for name, principal, days in raw
    ]


def build(pairs) -> Graph:
    graph = Graph()
    for name in ["a", "b", "c", "d"]:
        graph.add_dataset(table(name), UNPARTITIONED)
    for src, dst in pairs:
        if src != dst:
            graph.add_edge(
                Edge(table(src), table(dst), PartitionMapping.unknown(UNPARTITIONED), evidence="x")
            )
    return graph


# -- aggregation ---------------------------------------------------------------


@given(raw=events)
@settings(max_examples=250)
def test_counts_are_exact_and_principals_are_the_distinct_readers(raw):
    stats = usage.summarize(to_events(raw))
    for name in {n for n, _, _ in raw}:
        found = stats[table(name)]
        assert found.reads == sum(1 for n, _, _ in raw if n == name)
        assert found.principals == {p for n, p, _ in raw if n == name}


@given(raw=events)
@settings(max_examples=200)
def test_aggregation_does_not_depend_on_arrival_order(raw):
    made = to_events(raw)
    reordered = list(made)
    reordered.reverse()

    first, second = usage.summarize(made), usage.summarize(reordered)
    assert first.keys() == second.keys()
    for dataset, stats in first.items():
        other = second[dataset]
        assert (stats.reads, stats.principals) == (other.reads, other.principals)
        assert (stats.first_read, stats.last_read) == (other.first_read, other.last_read)


@given(raw=events.filter(bool))
@settings(max_examples=250)
def test_first_read_never_follows_last_read(raw):
    for stats in usage.summarize(to_events(raw)).values():
        assert stats.first_read is not None and stats.last_read is not None
        assert stats.first_read <= stats.last_read


@given(raw=events, days=st.integers(min_value=1, max_value=365))
@settings(max_examples=200)
def test_the_window_is_carried_onto_every_result(raw, days):
    """A caller must never hold counts without also holding the period they cover."""
    span = timedelta(days=days)
    for stats in usage.summarize(to_events(raw), window=span).values():
        assert stats.window == span


@given(raw=events)
@settings(max_examples=250)
def test_human_principals_are_a_subset_of_all_principals(raw):
    for stats in usage.summarize(to_events(raw)).values():
        assert stats.human_principals <= stats.principals


# -- retirement, which must never over-claim -----------------------------------


@given(raw=events, pairs=edges)
@settings(max_examples=250)
def test_a_candidate_always_has_an_entirely_unread_downstream(raw, pairs):
    """The claim the module makes; anything else retires a table one hop from a reader."""
    graph = build(pairs)
    stats = usage.summarize(to_events(raw), window=WINDOW)

    def read_by_a_person(ds: DatasetId) -> bool:
        found = stats.get(ds)
        return bool(found and found.human_principals)

    for candidate in usage.retirement_candidates(graph, stats, window=WINDOW):
        assert not read_by_a_person(candidate.dataset)
        assert not any(read_by_a_person(child) for child in descendants(graph, candidate.dataset))


@given(raw=events, pairs=edges)
@settings(max_examples=250)
def test_a_dataset_read_by_a_person_is_never_a_candidate(raw, pairs):
    graph = build(pairs)
    stats = usage.summarize(to_events(raw), window=WINDOW)
    candidates = {c.dataset for c in usage.retirement_candidates(graph, stats, window=WINDOW)}

    for dataset, found in stats.items():
        if found.human_principals:
            assert dataset not in candidates


@given(raw=events, pairs=edges)
@settings(max_examples=200)
def test_every_candidate_carries_the_window_it_was_judged_over(raw, pairs):
    graph = build(pairs)
    stats = usage.summarize(to_events(raw), window=WINDOW)
    for candidate in usage.retirement_candidates(graph, stats, window=WINDOW):
        assert candidate.window == WINDOW
        assert "not evidence" in str(candidate)


@given(raw=events, pairs=edges)
@settings(max_examples=200)
def test_candidates_and_read_datasets_partition_the_graph_under_strict_counting(raw, pairs):
    """With scheduled reads counted, a dataset is a candidate exactly when its cone is unread."""
    graph = build(pairs)
    stats = usage.summarize(to_events(raw), window=WINDOW)
    candidates = {
        c.dataset
        for c in usage.retirement_candidates(graph, stats, window=WINDOW, ignore_scheduled=False)
    }
    for dataset in graph.datasets:
        cone = [dataset, *descendants(graph, dataset)]
        unread = all(stats.get(d, usage.UsageStats(dataset=d)).reads == 0 for d in cone)
        assert (dataset in candidates) == unread


# -- derived views -------------------------------------------------------------


@given(raw=events)
@settings(max_examples=250)
def test_read_counts_never_exceeds_the_raw_counts(raw):
    stats = usage.summarize(to_events(raw))
    strict = usage.read_counts(stats, people_only=True)
    loose = usage.read_counts(stats)
    assert strict.keys() == loose.keys()
    assert all(strict[ds] <= loose[ds] for ds in loose)


@given(raw=events)
@settings(max_examples=200)
def test_busiest_is_sorted_and_bounded(raw):
    stats = usage.summarize(to_events(raw))
    ranked = usage.busiest(stats, limit=3)
    assert len(ranked) <= 3
    assert [count for _, count in ranked] == sorted((c for _, c in ranked), reverse=True)


@given(raw=events, pairs=edges)
@settings(max_examples=200)
def test_read_ratio_is_a_fraction(raw, pairs):
    graph = build(pairs)
    ratio = usage.read_ratio(graph, usage.summarize(to_events(raw)))
    assert 0.0 <= ratio <= 1.0
