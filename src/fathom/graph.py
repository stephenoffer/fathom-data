"""Dependency graph and the invalidation planner.

The graph is the durable artifact everything else rides on. Invalidation walks it
downstream composing partition mappings; drift attribution walks it upstream
following column edges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .partitions import PartitionMapping, apply
from .types import ANY, UNPARTITIONED, ColumnRef, DatasetId, KeyPredicate, PartitionSpec

__all__ = ["Edge", "Graph", "InvalidationPlan"]

# How many times one dataset may be re-widened before we give up and mark it whole.
# Self-referencing incremental models create cycles whose windows grow by a constant
# each pass; without this the worklist would never converge.
MAX_REVISITS = 8


def _subsumes(outer: KeyPredicate, inner: KeyPredicate) -> bool:
    """True when `outer` covers everything `inner` does."""
    names = {k for k, _ in outer.bindings} | {k for k, _ in inner.bindings}
    for n in names:
        ov, iv = outer.get(n), inner.get(n)
        if ov is ANY:
            continue
        if iv is ANY or ov != iv:
            return False
    return True


def _absorb(
    existing: frozenset[KeyPredicate], incoming: Iterable[KeyPredicate]
) -> tuple[frozenset[KeyPredicate], bool]:
    """Merge predicates, dropping any that another already covers.

    Returns the merged set and whether it grew, which is what drives the worklist.
    """
    merged = set(existing)
    changed = False
    for cand in incoming:
        if any(_subsumes(e, cand) for e in merged):
            continue
        merged = {e for e in merged if not _subsumes(cand, e)}
        merged.add(cand)
        changed = True
    return frozenset(merged), changed


@dataclass(frozen=True)
class Edge:
    """A proven dependency from one dataset to another."""

    src: DatasetId
    dst: DatasetId
    mapping: PartitionMapping
    columns: tuple[tuple[str, str], ...] = ()  # (source column, target column)
    evidence: str = "declared"  # how we learned this, for explainability

    def __str__(self) -> str:
        return f"{self.src} -> {self.dst} {self.mapping} [{self.evidence}]"


@dataclass
class InvalidationPlan:
    """What must be rebuilt, in what order, and why."""

    dirty: dict[DatasetId, frozenset[KeyPredicate]] = field(default_factory=dict)
    order: list[DatasetId] = field(default_factory=list)
    reasons: dict[DatasetId, list[str]] = field(default_factory=lambda: defaultdict(list))
    widened: set[DatasetId] = field(default_factory=set)
    cyclic: set[DatasetId] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return not self.dirty

    def partitions(self, ds: DatasetId) -> frozenset[KeyPredicate]:
        return self.dirty.get(ds, frozenset())

    def summary(self) -> str:
        if self.is_empty:
            return "nothing to rebuild"
        lines = []
        for ds in self.order:
            keys = sorted(str(k) for k in self.dirty[ds])
            note = "  (widened to whole dataset)" if ds in self.widened else ""
            shown = ", ".join(keys[:4]) + (f", +{len(keys) - 4} more" if len(keys) > 4 else "")
            lines.append(f"{ds}{note}\n    {shown}")
        return "\n".join(lines)


class Graph:
    """Datasets, their partition specs, and the proven edges between them."""

    def __init__(self) -> None:
        self._specs: dict[DatasetId, PartitionSpec] = {}
        self._out: dict[DatasetId, list[Edge]] = defaultdict(list)
        self._in: dict[DatasetId, list[Edge]] = defaultdict(list)

    # -- construction ----------------------------------------------------------

    def add_dataset(self, ds: DatasetId, spec: PartitionSpec = UNPARTITIONED) -> None:
        existing = self._specs.get(ds)
        if existing is not None and existing != spec and len(existing) > 0:
            raise ValueError(f"conflicting partition specs for {ds}: {existing} vs {spec}")
        self._specs[ds] = spec

    def add_edge(self, edge: Edge) -> None:
        self._specs.setdefault(edge.src, UNPARTITIONED)
        self._specs.setdefault(edge.dst, UNPARTITIONED)
        self._out[edge.src].append(edge)
        self._in[edge.dst].append(edge)

    def spec(self, ds: DatasetId) -> PartitionSpec:
        return self._specs.get(ds, UNPARTITIONED)

    @property
    def datasets(self) -> list[DatasetId]:
        return sorted(self._specs)

    @property
    def edges(self) -> list[Edge]:
        return [e for edges in self._out.values() for e in edges]

    def out_edges(self, ds: DatasetId) -> list[Edge]:
        return list(self._out.get(ds, ()))

    def in_edges(self, ds: DatasetId) -> list[Edge]:
        return list(self._in.get(ds, ()))

    # -- downstream: what do I have to rebuild ---------------------------------

    def invalidate(
        self,
        seeds: Mapping[DatasetId, Iterable[KeyPredicate]],
        *,
        max_revisits: int = MAX_REVISITS,
    ) -> InvalidationPlan:
        """Propagate dirty partitions downstream to a fixpoint.

        Over-approximates by construction: unprovable relationships widen to the whole
        dataset, and a dataset revisited too many times (a cycle) widens as well.
        """
        plan = InvalidationPlan()
        queue: deque[DatasetId] = deque()
        revisits: dict[DatasetId, int] = defaultdict(int)

        for ds, keys in seeds.items():
            merged, changed = _absorb(frozenset(), keys)
            if changed:
                plan.dirty[ds] = merged
                plan.reasons[ds].append("seed: reported changed at source")
                queue.append(ds)

        while queue:
            ds = queue.popleft()
            for edge in self._out.get(ds, ()):
                dst_spec = self.spec(edge.dst)
                produced: set[KeyPredicate] = set()
                for key in plan.dirty.get(ds, frozenset()):
                    produced |= apply(edge.mapping, key, dst_spec)
                if not produced:
                    continue

                revisits[edge.dst] += 1
                if revisits[edge.dst] > max_revisits:
                    # A cycle whose window keeps growing. Stop chasing it and widen.
                    if edge.dst not in plan.widened:
                        plan.widened.add(edge.dst)
                        plan.cyclic.add(edge.dst)
                        plan.dirty[edge.dst] = frozenset({KeyPredicate.unbounded(dst_spec)})
                        plan.reasons[edge.dst].append(
                            f"widened: revisited more than {max_revisits} times via a cycle"
                        )
                    continue

                merged, changed = _absorb(plan.dirty.get(edge.dst, frozenset()), produced)
                if changed:
                    plan.dirty[edge.dst] = merged
                    plan.reasons[edge.dst].append(
                        f"via {edge.src} {edge.mapping} [{edge.evidence}]"
                    )
                    # Only a partitioned dataset can lose precision. An unpartitioned one
                    # was always going to be rebuilt whole, so that is not a widening.
                    if dst_spec.fields and any(k.is_unbounded for k in merged):
                        plan.widened.add(edge.dst)
                    queue.append(edge.dst)

        plan.order = self._rebuild_order(set(plan.dirty), plan)
        return plan

    def _rebuild_order(self, affected: set[DatasetId], plan: InvalidationPlan) -> list[DatasetId]:
        """Kahn's algorithm over the affected subgraph; cycles fall back to name order."""
        indegree = {ds: 0 for ds in affected}
        for ds in affected:
            for edge in self._out.get(ds, ()):
                if edge.dst in affected and edge.dst != ds:
                    indegree[edge.dst] += 1

        ready = deque(sorted((ds for ds, n in indegree.items() if n == 0), key=str))
        out: list[DatasetId] = []
        while ready:
            ds = ready.popleft()
            out.append(ds)
            for edge in self._out.get(ds, ()):
                if edge.dst in affected and edge.dst != ds:
                    indegree[edge.dst] -= 1
                    if indegree[edge.dst] == 0:
                        ready.append(edge.dst)

        leftover = sorted(affected - set(out), key=str)
        if leftover:
            plan.cyclic.update(leftover)
        return out + leftover

    # -- upstream: why did this change ----------------------------------------

    def upstream_columns(self, target: ColumnRef, *, max_depth: int = 6) -> list[list[ColumnRef]]:
        """Column-level paths feeding `target`, nearest first.

        This is what turns a drift alert into a diagnosis: the profiler says a column
        moved, and these are the upstream columns that could have moved it.
        """
        paths: list[list[ColumnRef]] = []
        frontier: list[list[ColumnRef]] = [[target]]
        for _ in range(max_depth):
            nxt: list[list[ColumnRef]] = []
            for path in frontier:
                tip = path[-1]
                for edge in self._in.get(tip.dataset, ()):
                    sources = [s for s, t in edge.columns if t == tip.column]
                    if not sources and not edge.columns:
                        # Dataset-level edge: we know it contributes, not which column.
                        sources = ["*"]
                    for s in sources:
                        step = ColumnRef(edge.src, s)
                        if step in path:
                            continue
                        extended = [*path, step]
                        paths.append(extended)
                        nxt.append(extended)
            if not nxt:
                break
            frontier = nxt
        return paths
