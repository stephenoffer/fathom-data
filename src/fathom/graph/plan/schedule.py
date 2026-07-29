"""Turning a plan into something an orchestrator can run.

A plan says what to rebuild. It does not say in how many steps, what can run at once,
or which partitions to group into one job — and those decisions are most of the
wall-clock time. A hundred single-partition jobs against a warehouse cost a hundred
query startups; the same hundred partitions in four jobs cost four.

Three things live here:

- **Waves.** Datasets grouped so everything in wave *n* can run concurrently once
  wave *n-1* is done. Derived from the plan's own order, so a cycle the planner
  widened around does not become a scheduler deadlock.
- **Batching.** Adjacent partitions of one dataset merged into single units of work,
  because a contiguous date range is one query and not thirty.
- **Export.** The same structure rendered as a task list, a DAG, or a shell script,
  so it drops into Airflow, Dagster, or a Makefile without a translation layer.

Nothing here executes anything. A scheduler that also runs jobs is a scheduler you
have to trust with credentials, and the whole point is that the plan is inspectable
before anything touches data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TypedDict

from ...core.grains import Grain, step
from ...core.types import ANY, DatasetId, KeyPredicate
from ..model import Graph, InvalidationPlan

__all__ = [
    "TaskEntry",
    "Batch",
    "Schedule",
    "Wave",
    "batch_partitions",
    "critical_path",
    "partitions_in_wave",
    "rebalance",
    "to_mermaid",
    "unbounded_batches",
    "estimate_duration",
    "schedule",
    "to_dag",
    "to_shell",
    "to_task_list",
    "waves",
]


@dataclass(frozen=True)
class Batch:
    """One unit of work: a dataset and the partitions to rebuild together."""

    dataset: DatasetId
    partitions: tuple[KeyPredicate, ...] = ()
    label: str = ""

    @property
    def size(self) -> int:
        """How many partitions this batch covers."""
        return len(self.partitions)

    def __str__(self) -> str:
        if self.label:
            return f"{self.dataset} [{self.label}]"
        return f"{self.dataset} ({self.size} partition(s))"


@dataclass
class Wave:
    """Work that can run concurrently."""

    index: int = 0
    batches: list[Batch] = field(default_factory=list)

    @property
    def datasets(self) -> list[DatasetId]:
        """Datasets appearing in this wave."""
        return sorted({b.dataset for b in self.batches}, key=str)

    @property
    def partitions(self) -> int:
        """Partitions across every batch in this wave."""
        return sum(b.size for b in self.batches)

    def __str__(self) -> str:
        return f"wave {self.index}: {len(self.batches)} batch(es), {self.partitions} partition(s)"


@dataclass
class Schedule:
    """A plan arranged into waves of batches."""

    waves: list[Wave] = field(default_factory=list)

    @property
    def total_batches(self) -> int:
        """Batches across the whole schedule — the unit of work dispatched."""
        return sum(len(w.batches) for w in self.waves)

    @property
    def total_partitions(self) -> int:
        """Partitions across the whole schedule."""
        return sum(w.partitions for w in self.waves)

    @property
    def max_parallelism(self) -> int:
        """The widest wave — how many workers this schedule can actually use."""
        return max((len(w.batches) for w in self.waves), default=0)

    def summary(self) -> str:
        """The schedule as text, one line per wave."""
        return (
            f"{len(self.waves)} wave(s), {self.total_batches} batch(es), "
            f"{self.total_partitions} partition(s), "
            f"peak parallelism {self.max_parallelism}"
        )


def waves(graph: Graph, plan: InvalidationPlan) -> list[list[DatasetId]]:
    """Group the plan's datasets into concurrent waves.

    A dataset joins the first wave after every affected dataset it depends on. Only
    dependencies *inside the plan* count — an unaffected upstream is already built and
    would otherwise push everything into an artificial extra wave.
    """
    affected = set(plan.dirty)
    remaining = dict.fromkeys(plan.order or sorted(affected, key=str))
    placed: dict[DatasetId, int] = {}

    for ds in remaining:
        parents = {
            edge.src for edge in graph.in_edges(ds) if edge.src in affected and edge.src != ds
        }
        # The planner's order already resolves cycles, so any unplaced parent here is
        # one the planner put later; treating it as wave 0 keeps this total.
        depth = max((placed.get(parent, 0) + 1 for parent in parents), default=0)
        placed[ds] = depth

    grouped: dict[int, list[DatasetId]] = {}
    for ds, depth in placed.items():
        grouped.setdefault(depth, []).append(ds)
    return [sorted(grouped[depth], key=str) for depth in sorted(grouped)]


def _time_field(key: KeyPredicate) -> tuple[str, datetime] | None:
    for name, value in key.bindings:
        if isinstance(value, datetime):
            return name, value
    return None


def batch_partitions(
    partitions: Iterable[KeyPredicate], *, max_size: int = 32, grain: Grain = Grain.DAY
) -> list[list[KeyPredicate]]:
    """Group partitions into contiguous runs, so a date range becomes one job.

    Contiguity is judged on the single time field, holding every other binding equal.
    Partitions with no time field, or with an unbounded one, are grouped by their
    remaining bindings and chunked by size.
    """
    keys = list(partitions)
    by_group: dict[tuple[tuple[str, str], ...], list[KeyPredicate]] = {}
    for key in keys:
        time = _time_field(key)
        others = tuple(
            (name, str(value)) for name, value in key.bindings if not (time and name == time[0])
        )
        by_group.setdefault(others, []).append(key)

    out: list[list[KeyPredicate]] = []
    for group in by_group.values():
        timed = [(key, _time_field(key)) for key in group]
        if all(entry is not None for _, entry in timed):
            ordered = sorted(
                ((key, entry) for key, entry in timed if entry is not None),
                key=lambda pair: pair[1][1],
            )
            run: list[KeyPredicate] = []
            previous: datetime | None = None
            for key, entry in ordered:
                _, stamp = entry
                contiguous = previous is not None and step(previous, 1, grain) == stamp
                if run and (not contiguous or len(run) >= max_size):
                    out.append(run)
                    run = []
                run.append(key)
                previous = stamp
            if run:
                out.append(run)
        else:
            for index in range(0, len(group), max_size):
                out.append(group[index : index + max_size])
    return out


def schedule(
    graph: Graph, plan: InvalidationPlan, *, max_batch: int = 32, grain: Grain = Grain.DAY
) -> Schedule:
    """Arrange a plan into waves of batched work."""
    result = Schedule()
    for index, group in enumerate(waves(graph, plan)):
        wave = Wave(index=index)
        for ds in group:
            keys = plan.partitions(ds)
            if not keys:
                continue
            runs = batch_partitions(keys, max_size=max_batch, grain=grain)
            for run in runs:
                items = tuple(run)
                wave.batches.append(Batch(dataset=ds, partitions=items, label=_range_label(items)))
        if wave.batches:
            result.waves.append(wave)
    return result


def _range_label(keys: Sequence[KeyPredicate]) -> str:
    """A short human label for a batch: a date range where there is one."""
    stamps = [entry[1] for entry in (_time_field(k) for k in keys) if entry is not None]
    if not stamps:
        return f"{len(keys)} partition(s)"
    low, high = min(stamps), max(stamps)
    if low == high:
        return low.date().isoformat()
    return f"{low.date().isoformat()}..{high.date().isoformat()}"


def critical_path(graph: Graph, plan: InvalidationPlan) -> list[DatasetId]:
    """The longest dependency chain inside the plan.

    The lower bound on wall-clock time no amount of parallelism removes.
    """
    affected = set(plan.dirty)
    best: dict[DatasetId, list[DatasetId]] = {}
    for ds in plan.order or sorted(affected, key=str):
        parents = [
            edge.src for edge in graph.in_edges(ds) if edge.src in affected and edge.src != ds
        ]
        longest: list[DatasetId] = []
        for parent in parents:
            candidate = best.get(parent, [])
            if len(candidate) > len(longest):
                longest = candidate
        best[ds] = [*longest, ds]
    return max(best.values(), key=len, default=[])


def estimate_duration(
    plan: Schedule, *, per_batch: Mapping[DatasetId, timedelta] | None = None, workers: int = 8
) -> timedelta:
    """Estimated wall clock for a schedule at a given worker count.

    Each wave costs its batches divided across the workers, rounded up, times the
    per-batch duration. Crude, and close enough to answer "will this finish before
    the morning" — which is the only question anyone asks of it.
    """
    durations = dict(per_batch or {})
    total = timedelta()
    for wave in plan.waves:
        if not wave.batches:
            continue
        slots = max(1, -(-len(wave.batches) // max(1, workers)))
        slowest = max(
            (durations.get(b.dataset, timedelta(minutes=1)) for b in wave.batches),
            default=timedelta(minutes=1),
        )
        total += slowest * slots
    return total


# -- export --------------------------------------------------------------------


# Anything an orchestrator would reject in a task id. Airflow, Dagster, and Prefect
# all use the id as a Python identifier or a URL segment somewhere, and a dataset
# named `weird."name` otherwise emits a DAG file that does not parse.
_NOT_IDENTIFIER = re.compile(r"[^0-9A-Za-z_]+")


def _identifier(name: str) -> str:
    """Reduce a dataset name to something usable as a task id.

    The wave and batch position already make the full id unique, so collapsing
    distinct names to the same suffix cannot produce a duplicate id — which is why
    this can be lossy without being wrong.
    """
    cleaned = _NOT_IDENTIFIER.sub("_", name).strip("_")
    return cleaned or "dataset"


class TaskEntry(TypedDict):
    """One entry of `to_task_list`.

    Typed rather than `dict[str, object]` because every consumer indexes it several
    times, and an untyped mapping turns a renamed key into a `KeyError` inside
    somebody's DAG parse step rather than an error here.
    """

    id: str
    wave: int
    dataset: str
    label: str
    partitions: list[str]
    depends_on: list[str]


def to_task_list(plan: Schedule) -> list[TaskEntry]:
    """A flat task list with dependencies, the shape most orchestrators accept."""
    tasks: list[TaskEntry] = []
    previous_ids: list[str] = []
    for wave in plan.waves:
        current_ids: list[str] = []
        for index, batch in enumerate(wave.batches):
            task_id = f"w{wave.index}_{index}_{_identifier(batch.dataset.name)}"
            current_ids.append(task_id)
            tasks.append(
                TaskEntry(
                    id=task_id,
                    wave=wave.index,
                    dataset=str(batch.dataset),
                    label=batch.label,
                    partitions=[str(k) for k in batch.partitions],
                    depends_on=list(previous_ids),
                )
            )
        previous_ids = current_ids
    return tasks


def to_dag(plan: Schedule, *, indent: int | None = 2) -> str:
    """The task list as JSON, for an orchestrator to consume directly."""
    return json.dumps({"tasks": to_task_list(plan)}, indent=indent, sort_keys=True)


def to_shell(plan: Schedule, *, command: str = "fathom rebuild") -> str:
    """A shell script that runs the schedule, waves separated by `wait`.

    The escape hatch. Not everyone has an orchestrator, and a script that can be read
    top to bottom is a better first step than a DAG nobody can inspect.
    """
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for wave in plan.waves:
        lines.append(f"# wave {wave.index} — {len(wave.batches)} batch(es)")
        for batch in wave.batches:
            keys = " ".join(f"'{k}'" for k in batch.partitions)
            lines.append(f"{command} '{batch.dataset}' --partitions {keys} &")
        lines.append("wait")
        lines.append("")
    return "\n".join(lines)


def to_mermaid(plan: Schedule) -> str:
    """The schedule as a Mermaid diagram, one subgraph per wave."""
    lines = ["flowchart TD"]
    for wave in plan.waves:
        lines.append(f"    subgraph wave{wave.index}[Wave {wave.index}]")
        for index, batch in enumerate(wave.batches):
            node = f"w{wave.index}b{index}"
            lines.append(f'        {node}["{batch.dataset.name}<br/>{batch.label}"]')
        lines.append("    end")
    for previous, current in zip(plan.waves, plan.waves[1:], strict=False):
        lines.append(f"    wave{previous.index} --> wave{current.index}")
    return "\n".join(lines)


def partitions_in_wave(plan: Schedule, index: int) -> list[KeyPredicate]:
    """Every partition scheduled in one wave."""
    for wave in plan.waves:
        if wave.index == index:
            return [key for batch in wave.batches for key in batch.partitions]
    return []


def rebalance(plan: Schedule, *, max_batch: int) -> Schedule:
    """Re-split batches to a new maximum size, keeping the wave structure.

    Used when the first estimate turns out wrong: the shape of the dependency graph
    has not changed, only how much work fits in one job.
    """
    out = Schedule()
    for wave in plan.waves:
        rebuilt = Wave(index=wave.index)
        for batch in wave.batches:
            keys = list(batch.partitions)
            for start in range(0, len(keys), max_batch):
                chunk = tuple(keys[start : start + max_batch])
                rebuilt.batches.append(
                    Batch(dataset=batch.dataset, partitions=chunk, label=_range_label(chunk))
                )
        out.waves.append(rebuilt)
    return out


def unbounded_batches(plan: Schedule) -> list[Batch]:
    """Batches whose partitions are unconstrained — whole-dataset rebuilds in disguise.

    Worth surfacing before a run: these are where the plan gave up on precision, and
    they will dominate the bill.
    """
    return [
        batch
        for wave in plan.waves
        for batch in wave.batches
        if any(all(v is ANY for _, v in key.bindings) for key in batch.partitions)
    ]
