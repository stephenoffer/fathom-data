"""What a plan costs, and what skipping the rest of it saved.

The argument for partition-scoped invalidation is financial, and an argument nobody
can price is an argument nobody acts on. This module turns a plan into a number, and
more importantly turns the *difference* between a plan and a full rebuild into a
number, because that difference is the product.

Three cost bases, because different systems bill differently and a single "cost per
run" hides which one you are paying:

- **per partition** — a warehouse charging by the query, roughly flat per slice
- **per byte scanned** — Athena, BigQuery, most lakehouse engines
- **per token** — embedding and inference endpoints, where a re-embed of an unchanged
  corpus is the largest avoidable line item in most AI budgets

`carbon` is the same arithmetic against a grid intensity figure. It is an estimate
and labelled one; the point is that it moves with the same lever the money does, so
a team optimizing one is optimizing both.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from ...core.types import DatasetId, KeyPredicate
from ..model import Graph, InvalidationPlan
from ..query import closure, descendants

__all__ = [
    "CostEstimate",
    "CostModel",
    "PartitionCost",
    "ShadowSummary",
    "annualized",
    "attributed_cost",
    "budget_exceeded",
    "carbon",
    "compare_models",
    "measure",
    "partition_counts_from",
    "total_partitions",
    "cost_per_dataset",
    "estimate_full_rebuild",
    "estimate_plan",
    "most_expensive",
    "savings",
    "shadow_savings",
    "unused_expensive",
]

# Grams of CO2 per kilowatt-hour, world average. Deliberately a constant a caller can
# override rather than a lookup table that would need maintaining and would still be
# wrong for any specific region.
DEFAULT_GRID_INTENSITY = 400.0


@runtime_checkable
class ShadowSummary(Protocol):
    """The one thing costing needs from a store: the accumulated shadow totals."""

    def shadow_summary(self) -> Mapping[str, Any]:
        """Accumulated shadow totals, as the store reports them."""
        ...


@dataclass(frozen=True)
class PartitionCost:
    """What one partition of one dataset costs to rebuild."""

    dataset: DatasetId
    bytes_scanned: int = 0
    rows: int = 0
    tokens: int = 0
    seconds: float = 0.0
    partitions: int = 1


@dataclass(frozen=True)
class CostModel:
    """Prices for the three bases, plus an optional per-partition floor.

    A model can populate any subset. Leaving `price_per_tb_scanned` at zero on a
    per-query warehouse is correct rather than incomplete.

    **`price_per_tb_scanned` is per tebibyte (2^40 bytes), not per 10^12 bytes.**
    That matches how BigQuery and Databricks quote scan pricing, so their published
    figure can be pasted in directly. It is stated because the two units differ by
    about 10%, and a cost estimate silently 10% out is worse than no estimate — it
    gets quoted in a business case.
    """

    # 2^40. Named so the unit is legible at the call site rather than a magic number.
    BYTES_PER_TIB: ClassVar[int] = 1 << 40

    price_per_partition: float = 0.0
    price_per_tb_scanned: float = 0.0
    price_per_million_tokens: float = 0.0
    price_per_compute_hour: float = 0.0
    kwh_per_tb_scanned: float = 0.12
    grid_intensity: float = DEFAULT_GRID_INTENSITY

    def cost_of(self, item: PartitionCost) -> float:
        """What one entry costs under this model, summing every basis that applies."""
        return (
            item.partitions * self.price_per_partition
            + item.bytes_scanned / self.BYTES_PER_TIB * self.price_per_tb_scanned
            + item.tokens / 1_000_000 * self.price_per_million_tokens
            + item.seconds / 3600 * self.price_per_compute_hour
        )


@dataclass
class CostEstimate:
    """What a plan costs, against what doing everything would have."""

    planned: float = 0.0
    full: float = 0.0
    partitions_planned: int = 0
    partitions_total: int = 0
    per_dataset: dict[DatasetId, float] = field(default_factory=dict)

    @property
    def avoided(self) -> float:
        """Money the plan did not spend, floored at zero."""
        return max(0.0, self.full - self.planned)

    @property
    def savings_ratio(self) -> float:
        """Avoided cost as a fraction of a full rebuild."""
        return 0.0 if self.full <= 0 else self.avoided / self.full

    def summary(self) -> str:
        """The estimate as text: plan cost against full-rebuild cost."""
        return (
            f"plan costs {self.planned:,.2f} vs {self.full:,.2f} for a full rebuild — "
            f"{self.savings_ratio:.0%} avoided "
            f"({self.partitions_planned:,} of {self.partitions_total:,} partitions)"
        )


def estimate_plan(
    plan: InvalidationPlan,
    model: CostModel,
    *,
    unit_costs: Mapping[DatasetId, PartitionCost] | None = None,
) -> CostEstimate:
    """What executing this plan costs.

    `unit_costs` gives the per-partition cost of each dataset, typically measured from
    a previous run. Datasets absent from it fall back to the flat per-partition price,
    which is a floor rather than a guess.
    """
    costs = dict(unit_costs or {})
    estimate = CostEstimate()

    for ds, keys in plan.dirty.items():
        count = len(keys)
        unit = costs.get(ds, PartitionCost(dataset=ds))
        per_partition = model.cost_of(
            PartitionCost(
                dataset=ds,
                bytes_scanned=unit.bytes_scanned,
                rows=unit.rows,
                tokens=unit.tokens,
                seconds=unit.seconds,
                partitions=1,
            )
        )
        total = per_partition * count
        estimate.per_dataset[ds] = round(total, 6)
        estimate.planned += total
        estimate.partitions_planned += count

    estimate.planned = round(estimate.planned, 6)
    return estimate


def estimate_full_rebuild(
    graph: Graph,
    model: CostModel,
    *,
    unit_costs: Mapping[DatasetId, PartitionCost] | None = None,
    partition_counts: Mapping[DatasetId, int] | None = None,
) -> float:
    """What rebuilding everything would cost.

    `partition_counts` is how many partitions each dataset has. Without it a dataset
    counts as one partition, which under-reports the saving rather than inflating it.
    """
    costs = dict(unit_costs or {})
    counts = dict(partition_counts or {})
    total = 0.0
    for ds in graph.datasets:
        unit = costs.get(ds, PartitionCost(dataset=ds))
        per_partition = model.cost_of(
            PartitionCost(
                dataset=ds,
                bytes_scanned=unit.bytes_scanned,
                rows=unit.rows,
                tokens=unit.tokens,
                seconds=unit.seconds,
                partitions=1,
            )
        )
        total += per_partition * counts.get(ds, 1)
    return round(total, 6)


def savings(
    graph: Graph,
    plan: InvalidationPlan,
    model: CostModel,
    *,
    unit_costs: Mapping[DatasetId, PartitionCost] | None = None,
    partition_counts: Mapping[DatasetId, int] | None = None,
) -> CostEstimate:
    """The full picture: plan cost, full-rebuild cost, and the difference."""
    estimate = estimate_plan(plan, model, unit_costs=unit_costs)
    estimate.full = estimate_full_rebuild(
        graph, model, unit_costs=unit_costs, partition_counts=partition_counts
    )
    counts = dict(partition_counts or {})
    estimate.partitions_total = sum(counts.get(ds, 1) for ds in graph.datasets)
    return estimate


def cost_per_dataset(
    plan: InvalidationPlan,
    model: CostModel,
    *,
    unit_costs: Mapping[DatasetId, PartitionCost] | None = None,
) -> list[tuple[DatasetId, float]]:
    """Plan cost broken down by dataset, most expensive first."""
    estimate = estimate_plan(plan, model, unit_costs=unit_costs)
    return sorted(estimate.per_dataset.items(), key=lambda kv: (-kv[1], str(kv[0])))


def most_expensive(
    plan: InvalidationPlan,
    model: CostModel,
    *,
    unit_costs: Mapping[DatasetId, PartitionCost] | None = None,
    limit: int = 10,
) -> list[tuple[DatasetId, float]]:
    """The datasets worth optimizing first."""
    return cost_per_dataset(plan, model, unit_costs=unit_costs)[:limit]


def attributed_cost(
    graph: Graph,
    consumer: DatasetId,
    model: CostModel,
    *,
    unit_costs: Mapping[DatasetId, PartitionCost] | None = None,
) -> float:
    """What one consumer's whole upstream costs to keep fresh.

    The number that makes a dashboard nobody opens visible as an expense: the report
    costs nothing, and the fourteen tables it needs cost the budget.
    """
    costs = dict(unit_costs or {})
    total = 0.0
    for ds in closure(graph, consumer):
        unit = costs.get(ds, PartitionCost(dataset=ds))
        total += model.cost_of(unit)
    return round(total, 6)


def unused_expensive(
    graph: Graph,
    model: CostModel,
    *,
    unit_costs: Mapping[DatasetId, PartitionCost] | None = None,
    threshold: float = 0.0,
) -> list[tuple[DatasetId, float]]:
    """Datasets that cost real money and that nothing downstream consumes.

    The deletion candidates. Every warehouse has them, and they are invisible without
    lineage precisely because nothing points at them.
    """
    costs = dict(unit_costs or {})
    out: list[tuple[DatasetId, float]] = []
    for ds in graph.datasets:
        if descendants(graph, ds):
            continue
        price = model.cost_of(costs.get(ds, PartitionCost(dataset=ds)))
        if price > threshold:
            out.append((ds, round(price, 6)))
    return sorted(out, key=lambda kv: (-kv[1], str(kv[0])))


def carbon(estimate: CostEstimate, model: CostModel, *, bytes_scanned: int = 0) -> dict[str, float]:
    """Estimated grams of CO2 for a plan, and for the full rebuild it replaced.

    An estimate. It moves with exactly the same lever the money does, which is the
    only claim being made.
    """
    terabytes = bytes_scanned / CostModel.BYTES_PER_TIB
    grams_full = terabytes * model.kwh_per_tb_scanned * model.grid_intensity
    ratio = 1.0 - estimate.savings_ratio
    return {
        "grams_full_rebuild": round(grams_full, 3),
        "grams_planned": round(grams_full * ratio, 3),
        "grams_avoided": round(grams_full * estimate.savings_ratio, 3),
    }


def shadow_savings(
    store: ShadowSummary, model: CostModel, *, price_per_partition: float = 0.0
) -> float:
    """Money avoided across every shadow run recorded so far.

    The accumulated evidence figure. Shadow mode is how a team builds confidence in
    the planner; this is what that confidence was worth while they were building it.
    """
    summary = store.shadow_summary()
    skipped = max(0, int(summary["total"]) - int(summary["planned"]))
    unit = price_per_partition or model.price_per_partition
    return round(skipped * unit, 6)


def annualized(amount: float, *, runs_per_day: float = 1.0) -> float:
    """Scale a per-run figure to a yearly one. The number that gets a budget approved."""
    return round(amount * runs_per_day * 365, 2)


def budget_exceeded(estimate: CostEstimate, *, budget: float) -> bool:
    """True when a plan costs more than its allowance.

    Worth gating a scheduled job on: an unexpectedly wide plan usually means a graph
    edit widened a mapping, and catching it here is cheaper than at the invoice.
    """
    return estimate.planned > budget


def measure(
    dataset: DatasetId,
    *,
    bytes_scanned: int = 0,
    rows: int = 0,
    tokens: int = 0,
    seconds: float = 0.0,
) -> PartitionCost:
    """Record what one partition actually cost, for feeding back as a `unit_cost`.

    Estimates built from measurements beat estimates built from assumptions, and the
    measurement is available from every engine's query history.
    """
    return PartitionCost(
        dataset=dataset, bytes_scanned=bytes_scanned, rows=rows, tokens=tokens, seconds=seconds
    )


def total_partitions(plan: InvalidationPlan) -> int:
    """How many partitions a plan touches in total."""
    return sum(len(keys) for keys in plan.dirty.values())


def partition_counts_from(
    profiles: Mapping[DatasetId, Sequence[KeyPredicate]],
) -> dict[DatasetId, int]:
    """Turn observed partition lists into the counts `savings` wants."""
    return {ds: len(keys) for ds, keys in profiles.items()}


def compare_models(
    plan: InvalidationPlan, models: Iterable[tuple[str, CostModel]]
) -> dict[str, float]:
    """Price one plan under several cost models.

    Useful when choosing an engine: the same plan on a per-byte engine and a per-query
    one can differ by an order of magnitude in either direction.
    """
    return {name: estimate_plan(plan, model).planned for name, model in models}
