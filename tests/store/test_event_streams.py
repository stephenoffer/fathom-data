"""The three append-only logs: arrivals, reads, and runs.

Each exists because a module above answers a question about history and previously
had nowhere to get it. These tests care about round-tripping and about the two
properties the modules depend on: arrivals come back oldest-first so duplicate groups
read in order, and a usage window travels with the stats rather than being inferred.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.grains import Grain
from fathom.core.types import DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.graph.plan import lifetime
from fathom.graph.plan.cost import CostModel
from fathom.observe import completeness, usage
from fathom.store.sqlite import Store

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.monthly")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MARCH = datetime(2026, 3, 14, tzinfo=UTC)


@pytest.fixture
def store():
    with Store(":memory:") as s:
        yield s


def key(day: int) -> KeyPredicate:
    return KeyPredicate.of(dt=datetime(2026, 3, day))


# -- arrivals ------------------------------------------------------------------


def test_an_arrival_round_trips(store):
    store.record_arrival(completeness.Arrival(RAW, key(14), MARCH, "abc", 100))
    (found,) = store.arrivals()
    assert found.dataset == RAW
    assert found.key == key(14)
    assert found.digest == "abc"
    assert found.row_count == 100


def test_a_partition_value_keeps_its_type_through_the_store(store):
    """A datetime that came back as a string would never match a planner's key."""
    store.record_arrival(completeness.Arrival(RAW, key(14), MARCH))
    assert store.arrivals()[0].key.get("dt") == datetime(2026, 3, 14)


def test_arrivals_come_back_oldest_first(store):
    """Duplicate groups are read in order, so a restatement's direction is legible."""
    store.record_arrivals(
        [
            completeness.Arrival(RAW, key(14), MARCH + timedelta(days=2), "b"),
            completeness.Arrival(RAW, key(14), MARCH, "a"),
        ]
    )
    assert [a.digest for a in store.arrivals()] == ["a", "b"]


def test_record_arrivals_reports_how_many_it_wrote(store):
    written = store.record_arrivals([completeness.Arrival(RAW, key(d), MARCH) for d in (1, 2, 3)])
    assert written == 3


def test_arrivals_filter_by_dataset(store):
    store.record_arrival(completeness.Arrival(RAW, key(14), MARCH))
    store.record_arrival(completeness.Arrival(GOLD, key(14), MARCH))
    assert [a.dataset for a in store.arrivals(RAW)] == [RAW]


def test_arrivals_filter_by_time(store):
    store.record_arrivals(
        [
            completeness.Arrival(RAW, key(1), MARCH - timedelta(days=10)),
            completeness.Arrival(RAW, key(2), MARCH),
        ]
    )
    assert len(store.arrivals(since=MARCH - timedelta(days=1))) == 1


def test_a_naive_arrival_timestamp_is_stored_as_utc(store):
    store.record_arrival(completeness.Arrival(RAW, key(14), datetime(2026, 3, 14, 12)))
    assert store.arrivals()[0].observed.tzinfo is not None


def test_present_partitions_dedupes_repeat_arrivals(store):
    store.record_arrivals(
        [
            completeness.Arrival(RAW, key(14), MARCH, "a"),
            completeness.Arrival(RAW, key(14), MARCH + timedelta(days=1), "b"),
            completeness.Arrival(RAW, key(15), MARCH),
        ]
    )
    assert len(store.present_partitions(RAW)) == 2


def test_present_partitions_still_answers_after_a_deletion(store):
    """A listing cannot; an arrival log can, which is why completeness reads this."""
    store.record_arrival(completeness.Arrival(RAW, key(14), MARCH))
    assert key(14) in store.present_partitions(RAW)


def test_stored_arrivals_drive_a_completeness_report(store):
    for day in (1, 2, 5):
        store.record_arrival(completeness.Arrival(RAW, key(day), MARCH))
    result = completeness.report(
        RAW,
        DAY,
        store.present_partitions(RAW),
        start=datetime(2026, 3, 1),
        end=datetime(2026, 3, 5),
    )
    assert len(result.absent) == 2
    assert result.runs[0].count == 2


def test_stored_arrivals_classify_a_restatement(store):
    store.record_arrivals(
        [
            completeness.Arrival(RAW, key(14), MARCH, "a"),
            completeness.Arrival(RAW, key(14), MARCH + timedelta(days=1), "b"),
        ]
    )
    assert len(completeness.restatements(store.arrivals())) == 1


# -- reads ---------------------------------------------------------------------


def test_a_read_round_trips(store):
    store.record_read(usage.ReadEvent(GOLD, "ana", MARCH, kind="dashboard", query_id="q1"))
    (found,) = store.reads()
    assert (found.principal, found.kind, found.query_id) == ("ana", "dashboard", "q1")


def test_reads_come_back_newest_first(store):
    store.record_reads(
        [
            usage.ReadEvent(GOLD, "old", MARCH - timedelta(days=5)),
            usage.ReadEvent(GOLD, "new", MARCH),
        ]
    )
    assert [r.principal for r in store.reads()] == ["new", "old"]


def test_record_reads_reports_its_count(store):
    assert store.record_reads([usage.ReadEvent(GOLD, "a", MARCH)] * 3) == 3


def test_reads_filter_by_dataset_and_time(store):
    store.record_reads(
        [
            usage.ReadEvent(GOLD, "a", MARCH),
            usage.ReadEvent(RAW, "b", MARCH),
            usage.ReadEvent(GOLD, "c", MARCH - timedelta(days=90)),
        ]
    )
    assert len(store.reads(GOLD)) == 2
    assert len(store.reads(GOLD, since=MARCH - timedelta(days=1))) == 1


def test_usage_aggregates_the_stored_reads(store):
    store.record_reads([usage.ReadEvent(GOLD, "ana", MARCH), usage.ReadEvent(GOLD, "ben", MARCH)])
    stats = store.usage()
    assert stats[GOLD].reads == 2
    assert stats[GOLD].principals == {"ana", "ben"}


def test_the_usage_window_travels_with_the_stats(store):
    """A caller cannot report 'unused' without also holding 'over what period'."""
    store.record_read(usage.ReadEvent(GOLD, "ana", datetime.now(UTC)))
    stats = store.usage(window=timedelta(days=30))
    assert stats[GOLD].window == timedelta(days=30)


def test_a_windowed_usage_query_excludes_older_reads(store):
    store.record_reads(
        [
            usage.ReadEvent(GOLD, "recent", datetime.now(UTC)),
            usage.ReadEvent(RAW, "ancient", datetime(2020, 1, 1, tzinfo=UTC)),
        ]
    )
    assert set(store.usage(window=timedelta(days=30))) == {GOLD}


# -- runs ----------------------------------------------------------------------


def test_a_run_round_trips(store):
    store.record_run(lifetime.RunRecord(GOLD, MARCH, partitions=5, bytes_scanned=99, seconds=1.5))
    (found,) = store.runs()
    assert (found.partitions, found.bytes_scanned) == (5, 99)
    assert found.seconds == pytest.approx(1.5)


def test_runs_come_back_oldest_first_for_accumulation(store):
    store.record_run(lifetime.RunRecord(GOLD, MARCH, partitions=1))
    store.record_run(lifetime.RunRecord(GOLD, MARCH - timedelta(days=1), partitions=2))
    assert [r.partitions for r in store.runs()] == [2, 1]


def test_runs_filter_by_dataset_and_time(store):
    store.record_run(lifetime.RunRecord(GOLD, MARCH))
    store.record_run(lifetime.RunRecord(RAW, MARCH - timedelta(days=90)))
    assert len(store.runs(GOLD)) == 1
    assert len(store.runs(since=MARCH - timedelta(days=1))) == 1


def test_stored_runs_accumulate_into_a_lifetime_cost(store):
    store.record_run(lifetime.RunRecord(GOLD, MARCH - timedelta(days=1), partitions=10))
    store.record_run(lifetime.RunRecord(GOLD, MARCH, partitions=10))
    totals = lifetime.accumulate(store.runs(), CostModel(price_per_partition=2.0))
    assert totals[GOLD].spend == pytest.approx(40.0)
    assert totals[GOLD].runs == 2


def test_the_whole_value_question_answers_from_one_store(store):
    """Cost from runs, usage from reads — the two halves that were never divided."""
    store.record_run(lifetime.RunRecord(GOLD, MARCH, partitions=100))
    store.record_run(lifetime.RunRecord(RAW, MARCH, partitions=100))
    store.record_read(usage.ReadEvent(RAW, "ana", datetime.now(UTC)))

    totals = lifetime.accumulate(store.runs(), CostModel(price_per_partition=1.0))
    reads = {ds: s.reads for ds, s in store.usage(window=timedelta(days=30)).items()}
    findings = lifetime.value(totals, reads, threshold=50.0)

    by_dataset = {f.dataset: f.verdict for f in findings}
    assert by_dataset[GOLD] is lifetime.Verdict.REVIEW
    assert by_dataset[RAW] is lifetime.Verdict.EARNING


# -- schema --------------------------------------------------------------------


def test_the_new_tables_survive_a_reopen(tmp_path):
    path = tmp_path / "fathom.db"
    with Store(path) as first:
        first.record_read(usage.ReadEvent(GOLD, "ana", MARCH))
    with Store(path) as second:
        assert len(second.reads()) == 1


def test_an_empty_store_returns_empty_streams(store):
    assert store.arrivals() == []
    assert store.reads() == []
    assert store.runs() == []
    assert store.usage() == {}
