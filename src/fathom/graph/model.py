"""Dependency graph and the invalidation planner.

The graph is the durable artifact everything else rides on. Invalidation walks it
downstream composing partition mappings; drift attribution walks it upstream
following column edges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from ..core.partitions import PartitionMapping, apply
from ..core.types import UNPARTITIONED, ColumnRef, DatasetId, KeyPredicate, PartitionSpec, subsumes

__all__ = ["Edge", "Graph", "InvalidationPlan", "link"]


def _as_dataset(ds: DatasetId | str) -> DatasetId:
    """Accept an identity or the string form of one.

    Every constructor here takes both, because a graph written out by hand — in a
    test, a notebook, a bug report — is mostly string literals, and wrapping each
    one in `DatasetId(...)` triples its width for no added clarity.
    """
    return ds if isinstance(ds, DatasetId) else DatasetId.parse(ds)


def _as_spec(spec: PartitionSpec | str) -> PartitionSpec:
    """Accept a spec or its compact string form (``"dt:day, region"``)."""
    return spec if isinstance(spec, PartitionSpec) else PartitionSpec.parse(spec)


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
    """A proven dependency from one dataset to another.

    Three things travel on an edge, and each answers a different question:

    - **mapping** — which output partitions one dirty input partition dirties. This
      is what makes a plan narrower than "everything downstream".
    - **columns** — which source column feeds which target column. Empty means we
      know the datasets are related but not which columns, so drift attributes to
      the table rather than to a column.
    - **evidence** — how we learned this edge, carried so a surprising plan can be
      traced back to the thing that claimed the dependency.

    Example:
        >>> from fathom.core.partitions import PartitionMapping, TimeWindow
        >>> edge = Edge(
        ...     DatasetId("duckdb", "raw.events"),
        ...     DatasetId("duckdb", "silver.events"),
        ...     PartitionMapping.of(dt=TimeWindow.identity("dt", "day")),
        ...     columns=(("amount", "amount"),),
        ...     evidence="sql",
        ... )
        >>> print(edge)
        duckdb/raw.events -> duckdb/silver.events {dt: dt@day} [sql]
    """

    src: DatasetId
    dst: DatasetId
    mapping: PartitionMapping
    columns: tuple[tuple[str, str], ...] = ()  # (source column, target column)
    evidence: str = "declared"  # how we learned this, for explainability

    def __str__(self) -> str:
        return f"{self.src} -> {self.dst} {self.mapping} [{self.evidence}]"

    def explain(self) -> str:
        """This edge in sentences — what it claims, and where the claim came from.

        What `fathom lineage --explain` prints. The mapping is the part nobody can
        verify by reading the SQL, so it is the part worth spelling out.
        """
        lines = [f"{self.src} feeds {self.dst}, learned from {self.evidence}."]
        if self.columns:
            pairs = ", ".join(f"{s} -> {t}" for s, t in self.columns)
            lines.append(f"Columns: {pairs}")
        else:
            lines.append(
                "Columns: not known, so drift here attributes to the dataset "
                "rather than to a column."
            )
        lines.append("Partitions:")
        lines.extend(f"  {line}" for line in self.mapping.explain().splitlines())
        return "\n".join(lines)


@dataclass
class InvalidationPlan:
    """What must be rebuilt, in what order, and why.

    What `Graph.invalidate` returns and `fathom plan` prints. Iterating a plan walks
    its datasets in build order, so the common use reads as a loop:

        for dataset in plan:
            for key in plan.partitions(dataset):
                rebuild(dataset, key)

    Four fields carry the answer, and two carry the caveats. `widened` names the
    datasets the planner could not scope precisely, and `cyclic` the ones it had to
    stop iterating on. A plan much larger than expected is almost always explained
    by one of those two sets — start there, then read `why(dataset)`.
    """

    dirty: dict[DatasetId, frozenset[KeyPredicate]] = field(default_factory=dict)
    order: list[DatasetId] = field(default_factory=list)
    reasons: dict[DatasetId, list[str]] = field(default_factory=lambda: defaultdict(list))
    widened: set[DatasetId] = field(default_factory=set)
    cyclic: set[DatasetId] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        """True when nothing needs rebuilding."""
        return not self.dirty

    def __bool__(self) -> bool:
        """A plan is truthy when it has work in it, so `if plan:` reads correctly."""
        return not self.is_empty

    def __len__(self) -> int:
        """How many datasets are affected."""
        return len(self.dirty)

    def __iter__(self) -> Iterator[DatasetId]:
        """Affected datasets in build order — dependencies before dependents."""
        return iter(self.order)

    def __contains__(self, ds: object) -> bool:
        """True when this dataset is in the plan."""
        return ds in self.dirty

    def partitions(self, ds: DatasetId) -> frozenset[KeyPredicate]:
        """Dirty partitions for one dataset, empty when it is unaffected."""
        return self.dirty.get(ds, frozenset())

    @property
    def total_partitions(self) -> int:
        """Every dirty partition across every dataset — the size of the rebuild.

        Counts predicates, and a widened dataset contributes one predicate covering
        all of it. Compare against `graph.plan.cost` for what that actually costs.
        """
        return sum(len(keys) for keys in self.dirty.values())

    def why(self, ds: DatasetId) -> list[str]:
        """Why this dataset is in the plan, most recent reason last.

        The first entry is either the seed or the edge that first reached it; a
        `widened:` entry means precision was lost here and everything downstream
        inherited it.
        """
        return list(self.reasons.get(ds, ()))

    def explain(self, ds: DatasetId) -> str:
        """One dataset's place in the plan, in full: partitions, caveats, and cause.

        The answer to "why is this being rebuilt, and why so much of it?" — which is
        the question every plan larger than expected raises.
        """
        if ds not in self.dirty:
            return f"{ds} is not in this plan; nothing that reaches it changed"
        keys = sorted(str(k) for k in self.dirty[ds])
        lines = [f"{ds}: {len(keys)} partition(s) to rebuild"]
        lines.extend(f"  {k}" for k in keys[:10])
        if len(keys) > 10:
            lines.append(f"  … and {len(keys) - 10} more")
        if ds in self.widened:
            lines.append(
                "  ! widened to the whole dataset — a mapping on the way here could "
                "not be proven, so precision was lost"
            )
        if ds in self.cyclic:
            lines.append(
                "  ! on a cycle — the planner stopped enlarging and took the whole "
                "dataset rather than looping"
            )
        lines.append("  because:")
        lines.extend(f"    {reason}" for reason in self.why(ds))
        return "\n".join(lines)

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
    """Datasets, their partition specs, and the proven edges between them.

    Build one by hand for a test or a notebook; in a project it is built for you by
    `fathom ingest` and read back from the store. The constructors take strings
    wherever an identity or a spec is expected, so a graph reads as what it is:

    Example:
        >>> g = Graph()
        >>> g.add_dataset("duckdb/raw.events", "dt:day")
        >>> g.add_dataset("duckdb/gold.monthly", "dt:month")
        >>> _ = g.connect("duckdb/raw.events", "duckdb/gold.monthly", evidence="sql")
        >>> len(g), "duckdb/raw.events" in g
        (2, True)

    With no mapping given, `connect` uses the honest default: unbounded, meaning any
    change rebuilds the whole target. That is deliberately not inferred from the two
    specs — a daily source and a monthly target *usually* means a rollup, and a plan
    built on "usually" is a plan that occasionally serves stale data. State it:

        >>> from fathom.core.partitions import PartitionMapping
        >>> g = Graph()
        >>> daily, monthly = PartitionSpec.parse("dt:day"), PartitionSpec.parse("dt:month")
        >>> edge = g.connect("duckdb/raw.events", "duckdb/gold.monthly", evidence="sql",
        ...                  src_spec=daily, dst_spec=monthly,
        ...                  mapping=PartitionMapping.rollup(daily, monthly))
        >>> print(edge.mapping)
        {dt: dt@day->month}

    Traversal does not live here — see `fathom.query` for ancestors, descendants,
    paths, cycles, and subgraphs. This class holds the graph and plans over it.
    """

    def __init__(self) -> None:
        self._specs: dict[DatasetId, PartitionSpec] = {}
        self._out: dict[DatasetId, list[Edge]] = defaultdict(list)
        self._in: dict[DatasetId, list[Edge]] = defaultdict(list)
        self._cyclic_cache: frozenset[DatasetId] | None = None

    # -- construction ----------------------------------------------------------

    def add_dataset(self, ds: DatasetId | str, spec: PartitionSpec | str = UNPARTITIONED) -> None:
        """Register a dataset and its partition spec.

        Raises when a real spec would be replaced by a different real one: two
        sources disagreeing about partitioning silently corrupts every mapping
        composed across this dataset.

        Args:
            ds: The dataset, or a string `DatasetId.parse` accepts.
            spec: How it is partitioned, or a string `PartitionSpec.parse` accepts
                (``"dt:day, region"``). Omit for an unpartitioned dataset — which
                the planner can only ever invalidate whole.

        Raises:
            ValueError: A different non-empty spec is already registered.

        Example:
            >>> g = Graph()
            >>> g.add_dataset("duckdb/raw.events", "dt:day, region")
            >>> g.spec(DatasetId("duckdb", "raw.events")).names
            ('dt', 'region')
        """
        dataset = _as_dataset(ds)
        wanted = _as_spec(spec)
        existing = self._specs.get(dataset)
        if existing is not None and existing != wanted and len(existing) > 0:
            raise ValueError(
                f"conflicting partition specs for {dataset}: already registered as "
                f"`{existing}`, now given `{wanted}`. Two sources disagreeing about "
                f"partitioning corrupts every mapping composed across this dataset, so "
                f"this is refused rather than resolved. Fix the declaration in "
                f"fathom.yml, or alias the two identities if they are different datasets"
            )
        self._specs[dataset] = wanted

    def add_edge(self, edge: Edge) -> None:
        """Add a proven dependency. Parallel edges from different evidence are kept.

        Both endpoints are registered if new, unpartitioned. Prefer `connect` unless
        you already hold an `Edge`.
        """
        self._specs.setdefault(edge.src, UNPARTITIONED)
        self._specs.setdefault(edge.dst, UNPARTITIONED)
        self._out[edge.src].append(edge)
        self._in[edge.dst].append(edge)
        self._cyclic_cache = None

    def connect(
        self,
        src: DatasetId | str,
        dst: DatasetId | str,
        *,
        evidence: str = "declared",
        mapping: PartitionMapping | None = None,
        columns: Iterable[tuple[str, str]] = (),
        src_spec: PartitionSpec | str | None = None,
        dst_spec: PartitionSpec | str | None = None,
    ) -> Edge:
        """Register both endpoints and the edge between them, and return the edge.

        The one-call way to build a graph. Identities and specs may be strings.

        Args:
            src: The dataset that feeds.
            dst: The dataset that is fed.
            evidence: How this dependency was learned — ``sql``, ``dbt``,
                ``openlineage``, ``declared``. Carried so a surprising plan can be
                traced back to whatever claimed the dependency.
            mapping: Which output partitions one dirty input partition dirties.
                Defaults to unbounded — the honest answer when nothing proved a
                relationship, which costs a full rebuild of the target and never
                serves stale data. Pass `PartitionMapping.identity`,
                `.rollup`, or one built from `TimeWindow` to narrow it.
            columns: ``(source column, target column)`` pairs. Without them drift
                attributes to the dataset rather than to a column.
            src_spec: Partition spec for `src`, if not already registered.
            dst_spec: Partition spec for `dst`, if not already registered.

        Returns:
            The edge that was added.

        Example:
            >>> from fathom.core.partitions import PartitionMapping
            >>> g = Graph()
            >>> edge = g.connect("duckdb/a", "duckdb/b", evidence="sql",
            ...                  src_spec="dt:day", dst_spec="dt:day",
            ...                  mapping=PartitionMapping.identity(PartitionSpec.parse("dt:day")),
            ...                  columns=[("amount", "amount")])
            >>> print(edge.mapping)
            {dt: dt@day}
        """
        return link(
            self,
            _as_dataset(src),
            _as_dataset(dst),
            evidence=evidence,
            mapping=mapping,
            columns=columns,
            src_spec=None if src_spec is None else _as_spec(src_spec),
            dst_spec=None if dst_spec is None else _as_spec(dst_spec),
        )

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

    # -- orientation -----------------------------------------------------------

    def __len__(self) -> int:
        """How many datasets the graph holds."""
        return len(self._specs)

    def __contains__(self, ds: object) -> bool:
        """True when this dataset is in the graph. Accepts a string."""
        if isinstance(ds, str):
            try:
                ds = DatasetId.parse(ds)
            except ValueError:
                return False
        return ds in self._specs

    def __iter__(self) -> Iterator[DatasetId]:
        """Every dataset, sorted — so `for ds in graph` is stable across runs."""
        return iter(self.datasets)

    def __repr__(self) -> str:
        return f"<Graph {len(self._specs)} datasets, {len(self.edges)} edges>"

    def describe(self) -> str:
        """A few lines orienting a reader in an unfamiliar graph.

        Size, how much of it is precise enough to plan on, and the datasets nothing
        feeds. The counts here are the ones that predict how good a plan will be: an
        unbounded edge contributes nothing to narrowing a rebuild, and a dataset
        without a spec can only ever be invalidated whole.

        For the full picture see `fathom.metrics.coverage`, which this summarises.
        """
        edges = self.edges
        unbounded = sum(1 for e in edges if e.mapping.is_unbounded)
        specced = sum(1 for spec in self._specs.values() if spec.fields)
        sourced = {e.dst for e in edges}
        roots = [ds for ds in self.datasets if ds not in sourced]
        lines = [
            f"{len(self._specs)} datasets, {len(edges)} edges",
            f"{specced}/{len(self._specs)} datasets have a partition spec"
            " (the rest can only be rebuilt whole)",
        ]
        if edges:
            lines.append(
                f"{len(edges) - unbounded}/{len(edges)} edges carry a provable mapping"
                f" ({unbounded} unbounded, contributing nothing to a plan)"
            )
        lines.append(
            f"{len(roots)} source dataset(s) nothing else feeds: "
            + (", ".join(str(r) for r in roots[:5]) or "none")
            + (f", +{len(roots) - 5} more" if len(roots) > 5 else "")
        )
        return "\n".join(lines)

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

        The `plan` verb. Give it what changed at the source; it returns every
        partition of every downstream dataset that went stale, in build order.

        Over-approximates by construction: unprovable relationships widen to the whole
        dataset, and a dataset revisited too many times (a cycle) widens as well.

        Args:
            seeds: What changed, per dataset. A dataset absent from `seeds` is
                treated as unchanged — this is not a filter, it is the input.
            max_revisits: How many times a dataset *on a cycle* may be enlarged
                before it is widened to whole. Does not apply to acyclic datasets,
                which converge on their own.

        Returns:
            The plan. `is_empty` when nothing downstream is affected.

        Example:
            >>> from datetime import datetime
            >>> from fathom.core.partitions import PartitionMapping
            >>> daily, monthly = PartitionSpec.parse("dt:day"), PartitionSpec.parse("dt:month")
            >>> g = Graph()
            >>> _ = g.connect("duckdb/raw.events", "duckdb/gold.monthly", evidence="sql",
            ...               src_spec=daily, dst_spec=monthly,
            ...               mapping=PartitionMapping.rollup(daily, monthly))
            >>> plan = g.invalidate({
            ...     DatasetId("duckdb", "raw.events"): [KeyPredicate.of(dt=datetime(2026, 3, 14))]
            ... })
            >>> print(plan.summary())
            duckdb/raw.events
                dt=2026-03-14T00:00:00
            duckdb/gold.monthly
                dt=2026-03-01T00:00:00

        One dirty day, one dirty month. Note that the seeded dataset is in the plan
        too — it changed, so whatever reads it downstream is stale, and so is it.
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

        A path may contain ``*`` as a column name, meaning the edge is dataset-level:
        that dataset contributes, but which column is not known. Adapters without
        column lineage produce those, and they are still worth reporting — a
        candidate you can rule out beats no candidate at all.

        Args:
            target: The column that moved.
            max_depth: How many hops upstream to walk. Six covers a typical
                warehouse chain; raise it for deeper ones, at a cost in paths.

        Returns:
            Paths from `target` back towards its sources, nearest first. Each path
            starts at `target` itself.
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
