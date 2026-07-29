"""Dependency graph and the invalidation planner.

The graph is the durable artifact everything else rides on. Invalidation walks it
downstream composing partition mappings; drift attribution walks it upstream
following column edges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..core.partitions import PartitionMapping, apply
from ..core.types import UNPARTITIONED, ColumnRef, DatasetId, KeyPredicate, PartitionSpec, subsumes

__all__ = ["Edge", "Graph", "InvalidationPlan", "link"]

# How many times a dataset *on a cycle* may have its dirty set enlarged before we
# give up and mark it whole. Self-referencing incremental models create cycles whose
# windows grow by a constant each pass; without this the worklist would never
# converge. It deliberately does not apply to acyclic datasets, which always
# converge on their own — see `invalidate`.
MAX_REVISITS = 8

# Backstop for acyclic datasets. Reaching it means the predicate space is growing in
# a way we did not anticipate, so we widen rather than spin. Set far above anything a
# real graph produces, since firing it costs a full rebuild.
_ACYCLIC_SAFETY_VALVE = 10_000


def _absorb(
    existing: frozenset[KeyPredicate], incoming: Iterable[KeyPredicate]
) -> tuple[frozenset[KeyPredicate], bool]:
    """Merge predicates, dropping any that another already covers.

    Returns the merged set and whether it grew, which is what drives the worklist.
    """
    merged = set(existing)
    changed = False
    for cand in incoming:
        if any(subsumes(e, cand) for e in merged):
            continue
        merged = {e for e in merged if not subsumes(cand, e)}
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
        """True when nothing needs rebuilding."""
        return not self.dirty

    def partitions(self, ds: DatasetId) -> frozenset[KeyPredicate]:
        """Dirty partitions for one dataset, empty when it is unaffected."""
        return self.dirty.get(ds, frozenset())

    def summary(self) -> str:
        """The plan as text: datasets in build order, with their dirty partitions."""
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
        self._cyclic_cache: frozenset[DatasetId] | None = None

    # -- construction ----------------------------------------------------------

    def add_dataset(self, ds: DatasetId, spec: PartitionSpec = UNPARTITIONED) -> None:
        """Register a dataset and its partition spec.

        Raises when a real spec would be replaced by a different real one: two
        sources disagreeing about partitioning silently corrupts every mapping
        composed across this dataset.
        """
        existing = self._specs.get(ds)
        if existing is not None and existing != spec and len(existing) > 0:
            raise ValueError(f"conflicting partition specs for {ds}: {existing} vs {spec}")
        self._specs[ds] = spec

    def add_edge(self, edge: Edge) -> None:
        """Add a proven dependency. Parallel edges from different evidence are kept."""
        self._specs.setdefault(edge.src, UNPARTITIONED)
        self._specs.setdefault(edge.dst, UNPARTITIONED)
        self._out[edge.src].append(edge)
        self._in[edge.dst].append(edge)
        self._cyclic_cache = None

    def spec(self, ds: DatasetId) -> PartitionSpec:
        """How this dataset is partitioned, or `UNPARTITIONED` when unknown."""
        return self._specs.get(ds, UNPARTITIONED)

    @property
    def datasets(self) -> list[DatasetId]:
        """Every dataset in the graph, sorted for stable output."""
        return sorted(self._specs)

    @property
    def edges(self) -> list[Edge]:
        """Every edge in the graph, parallel edges included."""
        return [e for edges in self._out.values() for e in edges]

    def out_edges(self, ds: DatasetId) -> list[Edge]:
        """Edges leaving this dataset — where its changes flow."""
        return list(self._out.get(ds, ()))

    def in_edges(self, ds: DatasetId) -> list[Edge]:
        """Edges entering this dataset — what it is built from."""
        return list(self._in.get(ds, ()))

    def _on_cycle(self) -> frozenset[DatasetId]:
        """Datasets that can reach themselves — the only ones the planner can loop on.

        Iterative Tarjan, because a warehouse chain is long enough to blow the
        recursion limit. Cached until an edge is added, since the planner asks once
        per `invalidate` call and the answer only changes when the graph does.
        """
        if self._cyclic_cache is not None:
            return self._cyclic_cache

        index: dict[DatasetId, int] = {}
        low: dict[DatasetId, int] = {}
        stack: list[DatasetId] = []
        on_stack: set[DatasetId] = set()
        found: set[DatasetId] = set()
        counter = 0

        for root in self._specs:
            if root in index:
                continue
            work: list[tuple[DatasetId, int]] = [(root, 0)]
            while work:
                node, child_index = work[-1]
                if child_index == 0:
                    index[node] = low[node] = counter
                    counter += 1
                    stack.append(node)
                    on_stack.add(node)
                kids = self._out.get(node, ())
                recursed = False
                while child_index < len(kids):
                    kid = kids[child_index].dst
                    child_index += 1
                    if kid == node:
                        found.add(node)  # self-loop
                        continue
                    if kid not in index:
                        work[-1] = (node, child_index)
                        work.append((kid, 0))
                        recursed = True
                        break
                    if kid in on_stack:
                        low[node] = min(low[node], index[kid])
                if recursed:
                    continue
                work[-1] = (node, child_index)
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    component: list[DatasetId] = []
                    while True:
                        popped = stack.pop()
                        on_stack.discard(popped)
                        component.append(popped)
                        if popped == node:
                            break
                    if len(component) > 1:
                        found.update(component)

        self._cyclic_cache = frozenset(found)
        return self._cyclic_cache

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
        enlargements: dict[DatasetId, int] = defaultdict(int)
        forced: set[DatasetId] = set()
        on_cycle = self._on_cycle()

        for ds, keys in seeds.items():
            merged, changed = _absorb(frozenset(), keys)
            if changed:
                plan.dirty[ds] = merged
                plan.reasons[ds].append("seed: reported changed at source")
                queue.append(ds)

        while queue:
            ds = queue.popleft()
            for edge in self._out.get(ds, ()):
                dst = edge.dst
                if dst in forced:
                    # Already at the top of the lattice; nothing can enlarge it further.
                    continue

                dst_spec = self.spec(dst)
                produced: set[KeyPredicate] = set()
                for key in plan.dirty.get(ds, frozenset()):
                    produced |= apply(edge.mapping, key, dst_spec)
                if not produced:
                    continue

                merged, changed = _absorb(plan.dirty.get(dst, frozenset()), produced)
                if not changed:
                    continue

                # Only a dataset that can reach itself needs a non-convergence budget:
                # over a DAG the worklist provably terminates, because every step only
                # ever enlarges a dirty set and the space of predicates is finite. A
                # wide join is enlarged once per dirty parent and a deep one once per
                # arriving key, and charging either against a cycle budget is what
                # turns ordinary hub tables into spurious full rebuilds.
                enlargements[dst] += 1
                budget = max_revisits if dst in on_cycle else _ACYCLIC_SAFETY_VALVE
                if enlargements[dst] > budget:
                    forced.add(dst)
                    plan.widened.add(dst)
                    plan.cyclic.add(dst)
                    plan.dirty[dst] = frozenset({KeyPredicate.unbounded(dst_spec)})
                    plan.reasons[dst].append(
                        f"widened: enlarged more than {budget} times without converging"
                        + (" via a cycle" if dst in on_cycle else "")
                    )
                    # The widening has to reach everything downstream. Without this the
                    # consumers keep whatever narrower key set they were handed before
                    # this dataset widened, and the plan under-invalidates.
                    queue.append(dst)
                    continue

                plan.dirty[dst] = merged
                plan.reasons[dst].append(f"via {edge.src} {edge.mapping} [{edge.evidence}]")
                # Only a partitioned dataset can lose precision. An unpartitioned one
                # was always going to be rebuilt whole, so that is not a widening.
                if dst_spec.fields and any(k.is_unbounded for k in merged):
                    plan.widened.add(dst)
                queue.append(dst)

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


def link(
    graph: Graph,
    src: DatasetId,
    dst: DatasetId,
    *,
    evidence: str,
    mapping: PartitionMapping | None = None,
    columns: Iterable[tuple[str, str]] = (),
    src_spec: PartitionSpec | None = None,
    dst_spec: PartitionSpec | None = None,
) -> Edge:
    """Register both endpoints and the edge between them, and return the edge.

    Every recorder — a training run, a retrieval, an agent tool call, a prompt
    binding — needs the same three steps, and each one that reimplements them finds a
    different way to get the default mapping wrong. The default here is
    `PartitionMapping.unknown`, which is the honest answer whenever nothing can prove
    a partition relationship, and the answer the planner degrades safely around.

    Existing specs win. Passing `dst_spec` for a dataset the graph already knows
    would otherwise let a recorder overwrite a real catalog spec with a convention.
    """
    for dataset, spec in ((src, src_spec), (dst, dst_spec)):
        known = graph.spec(dataset)
        graph.add_dataset(dataset, known if known.fields else (spec or known))

    edge = Edge(
        src=src,
        dst=dst,
        mapping=mapping if mapping is not None else PartitionMapping.unknown(graph.spec(dst)),
        columns=tuple(columns),
        evidence=evidence,
    )
    graph.add_edge(edge)
    return edge
