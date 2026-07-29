""""""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fathom import cost, schedule
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import ANY, DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.graph import Edge, Graph
from fathom.store import Store

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.events")
GOLD = DatasetId("duckdb", "gold.monthly")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MONTH = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))


@pytest.fixture
def graph() -> Graph:
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(SILVER, DAY)
    g.add_dataset(GOLD, MONTH)
    g.add_edge(Edge(RAW, SILVER, PartitionMapping.identity(DAY), evidence="sql:1"))
    g.add_edge(Edge(SILVER, GOLD, PartitionMapping.rollup(DAY, MONTH), evidence="sql:2"))
    return g


@pytest.fixture
def plan(graph):
    days = [KeyPredicate.of(dt=datetime(2026, 3, d)) for d in range(1, 8)]
    return graph.invalidate({RAW: days})


# -- cost ----------------------------------------------------------------------


def test_plan_cost_scales_with_partitions(plan):
    model = cost.CostModel(price_per_partition=0.5)
    estimate = cost.estimate_plan(plan, model)
    assert estimate.partitions_planned == cost.total_partitions(plan)
    assert estimate.planned == pytest.approx(0.5 * estimate.partitions_planned)


def test_savings_compares_against_a_full_rebuild(graph, plan):
    model = cost.CostModel(price_per_partition=1.0)
    counts = {RAW: 365, SILVER: 365, GOLD: 12}
    estimate = cost.savings(graph, plan, model, partition_counts=counts)
    assert estimate.full == 742.0
    assert estimate.avoided > 0
    assert 0.9 < estimate.savings_ratio < 1.0
    assert "% avoided" in estimate.summary()


def test_byte_and_token_bases_are_summed():
    model = cost.CostModel(price_per_tb_scanned=5.0, price_per_million_tokens=0.13)
    item = cost.measure(RAW, bytes_scanned=1_099_511_627_776, tokens=1_000_000)
    assert model.cost_of(item) == pytest.approx(5.13)


def test_cost_per_dataset_ranks_by_spend(plan):
    unit = {SILVER: cost.measure(SILVER, seconds=3600)}
    model = cost.CostModel(price_per_compute_hour=10.0)
    ranked = cost.cost_per_dataset(plan, model, unit_costs=unit)
    assert ranked[0][0] == SILVER
    assert ranked[0][1] > 0


def test_unused_expensive_finds_deletion_candidates(graph):
    model = cost.CostModel(price_per_partition=2.0)
    found = cost.unused_expensive(graph, model)
    assert [ds for ds, _ in found] == [GOLD]  # nothing consumes gold


def test_carbon_tracks_the_same_lever():
    estimate = cost.CostEstimate(planned=10.0, full=100.0)
    numbers = cost.carbon(estimate, cost.CostModel(), bytes_scanned=10 * 1_099_511_627_776)
    assert numbers["grams_avoided"] == pytest.approx(numbers["grams_full_rebuild"] * 0.9)


def test_budget_gate_and_annualization():
    estimate = cost.CostEstimate(planned=42.0)
    assert cost.budget_exceeded(estimate, budget=10.0)
    assert not cost.budget_exceeded(estimate, budget=100.0)
    assert cost.annualized(1.0, runs_per_day=24) == 8760.0


def test_shadow_savings_reads_recorded_evidence():
    from fathom.store import ShadowObservation

    with Store() as store:
        store.record_shadow(
            ShadowObservation(RAW, datetime.now(UTC), planned=10, actual=10, missed=0, total=100)
        )
        assert cost.shadow_savings(store, cost.CostModel(price_per_partition=0.25)) == 22.5


# -- schedule ------------------------------------------------------------------


def test_waves_respect_dependencies(graph, plan):
    groups = schedule.waves(graph, plan)
    assert groups[0] == [RAW]
    assert groups[1] == [SILVER]
    assert groups[2] == [GOLD]


def test_contiguous_days_batch_into_one_job():
    keys = [KeyPredicate.of(dt=datetime(2026, 3, d)) for d in range(1, 8)]
    batches = schedule.batch_partitions(keys)
    assert len(batches) == 1
    assert len(batches[0]) == 7


def test_a_gap_splits_the_batch():
    keys = [
        KeyPredicate.of(dt=datetime(2026, 3, 1)),
        KeyPredicate.of(dt=datetime(2026, 3, 2)),
        KeyPredicate.of(dt=datetime(2026, 3, 10)),
    ]
    assert [len(b) for b in schedule.batch_partitions(keys)] == [2, 1]


def test_max_size_caps_a_batch():
    keys = [KeyPredicate.of(dt=datetime(2026, 3, d)) for d in range(1, 11)]
    assert [len(b) for b in schedule.batch_partitions(keys, max_size=4)] == [4, 4, 2]


def test_schedule_labels_a_date_range(graph, plan):
    built = schedule.schedule(graph, plan)
    assert built.total_partitions == cost.total_partitions(plan)
    labels = [b.label for wave in built.waves for b in wave.batches]
    assert any(".." in label for label in labels)
    assert "wave(s)" in built.summary()


def test_critical_path_is_the_longest_chain(graph, plan):
    assert schedule.critical_path(graph, plan) == [RAW, SILVER, GOLD]


def test_task_list_carries_wave_dependencies(graph, plan):
    tasks = schedule.to_task_list(schedule.schedule(graph, plan))
    assert tasks[0]["depends_on"] == []
    assert tasks[-1]["depends_on"]
    assert all("partitions" in task for task in tasks)


def test_shell_export_waits_between_waves(graph, plan):
    text = schedule.to_shell(schedule.schedule(graph, plan))
    assert text.startswith("#!/usr/bin/env bash")
    assert text.count("wait") == 3


def test_unbounded_batches_are_surfaced(graph):
    wide = graph.invalidate({RAW: [KeyPredicate.of(dt=ANY)]})
    built = schedule.schedule(graph, wide)
    assert schedule.unbounded_batches(built)


def test_rebalance_keeps_the_wave_structure(graph, plan):
    built = schedule.schedule(graph, plan, max_batch=32)
    smaller = schedule.rebalance(built, max_batch=2)
    assert len(smaller.waves) == len(built.waves)
    assert smaller.total_batches > built.total_batches
    assert smaller.total_partitions == built.total_partitions


def test_duration_estimate_shrinks_with_workers(graph, plan):
    built = schedule.schedule(graph, plan)
    one = schedule.estimate_duration(built, workers=1)
    many = schedule.estimate_duration(built, workers=16)
    assert many <= one
