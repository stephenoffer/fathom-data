"""Traversal and analysis over the dependency graph.

`Graph` deliberately exposes only what the planner needs: specs, edges, and a
fixpoint walk. Everything a human or an agent actually asks of a lineage graph —
*what breaks if I drop this column*, *how far is this table from a source*, *which
two models share an upstream* — is a traversal, and traversals belong here rather
than accreting onto the planner.

Three properties hold across this module:

- **Order is deterministic.** Every function that returns a collection returns it
  sorted, so output is diffable and a test that passes today passes tomorrow.
- **Cycles never hang.** Self-referencing incremental models are normal in dbt
  projects. Every walk carries a visited set and a depth bound.
- **Absence is empty, not an error.** Asking for the descendants of a dataset that
  is not in the graph yields `[]`. Callers frequently probe with identities they
  are not yet sure exist.
"""

from __future__ import annotations

import fnmatch
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence

from .graph import Edge, Graph
from .partitions import PartitionMapping, compose, join
from .types import ColumnRef, DatasetId, PartitionSpec

__all__ = [
    "ancestors",
    "between",
    "blast_radius",
    "children",
    "column_ancestors",
    "column_descendants",
    "column_paths",
    "columns_of",
    "common_ancestors",
    "common_descendants",
    "connected_components",
    "copy_graph",
    "cycles",
    "dataset_index",
    "degree",
    "depth_of",
    "descendants",
    "distance",
    "edge_between",
    "edges_between",
    "effective_mapping",
    "fan_in",
    "fan_out",
    "find",
    "has_cycle",
    "has_path",
    "height_of",
    "in_degree",
    "in_namespace",
    "is_downstream_of",
    "is_isolated",
    "is_upstream_of",
    "isolated",
    "leaves",
    "levels",
    "lowest_common_ancestors",
    "merge_graphs",
    "namespaces",
    "neighbors",
    "out_degree",
    "parents",
    "partitioned_datasets",
    "paths_between",
    "prune",
    "reachable",
    "relatives",
    "reverse",
    "roots",
    "select",
    "shortest_path",
    "siblings",
    "subgraph",
    "topological_order",
    "unpartitioned_datasets",
    "upstream_subgraph",
    "downstream_subgraph",
    "without",
]

# Depth ceiling for any unbounded walk. Deeper than any real warehouse lineage and
# shallow enough that a pathological cycle cannot spin.
MAX_DEPTH = 64


# -- immediate neighbourhood ---------------------------------------------------


def parents(graph: Graph, ds: DatasetId) -> list[DatasetId]:
    """Datasets feeding `ds` directly."""
    return sorted({e.src for e in graph.in_edges(ds)}, key=str)


def children(graph: Graph, ds: DatasetId) -> list[DatasetId]:
    """Datasets `ds` feeds directly."""
    return sorted({e.dst for e in graph.out_edges(ds)}, key=str)


def neighbors(graph: Graph, ds: DatasetId) -> list[DatasetId]:
    """Parents and children together, `ds` itself excluded."""
    return sorted({*parents(graph, ds), *children(graph, ds)} - {ds}, key=str)


def siblings(graph: Graph, ds: DatasetId) -> list[DatasetId]:
    """Other datasets sharing at least one direct parent with `ds`."""
    out: set[DatasetId] = set()
    for parent in parents(graph, ds):
        out.update(children(graph, parent))
    return sorted(out - {ds}, key=str)


def in_degree(graph: Graph, ds: DatasetId) -> int:
    """Number of distinct datasets feeding `ds`."""
    return len(parents(graph, ds))


def out_degree(graph: Graph, ds: DatasetId) -> int:
    """Number of distinct datasets `ds` feeds."""
    return len(children(graph, ds))


def degree(graph: Graph, ds: DatasetId) -> int:
    """Total distinct neighbours, in either direction."""
    return len(neighbors(graph, ds))


def fan_in(graph: Graph, ds: DatasetId) -> int:
    """Edge count into `ds`, counting parallel edges from different evidence."""
    return len(graph.in_edges(ds))


def fan_out(graph: Graph, ds: DatasetId) -> int:
    """Edge count out of `ds`, counting parallel edges from different evidence."""
    return len(graph.out_edges(ds))


# -- transitive walks ----------------------------------------------------------


def _walk(
    graph: Graph,
    start: DatasetId,
    *,
    downstream: bool,
    max_depth: int,
) -> dict[DatasetId, int]:
    """Breadth-first reachability from `start`, mapping dataset to hop count.

    `start` itself is excluded unless a cycle genuinely returns to it, which is
    information a caller wants rather than noise to suppress.
    """
    seen: dict[DatasetId, int] = {}
    frontier: deque[tuple[DatasetId, int]] = deque([(start, 0)])
    visited = {start}
    while frontier:
        node, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        edges = graph.out_edges(node) if downstream else graph.in_edges(node)
        for edge in edges:
            nxt = edge.dst if downstream else edge.src
            if nxt not in seen or seen[nxt] > depth + 1:
                seen[nxt] = depth + 1
            if nxt in visited:
                continue
            visited.add(nxt)
            frontier.append((nxt, depth + 1))
    return seen


def descendants(graph: Graph, ds: DatasetId, *, max_depth: int = MAX_DEPTH) -> list[DatasetId]:
    """Everything reachable downstream of `ds`."""
    return sorted(_walk(graph, ds, downstream=True, max_depth=max_depth), key=str)


def ancestors(graph: Graph, ds: DatasetId, *, max_depth: int = MAX_DEPTH) -> list[DatasetId]:
    """Everything `ds` transitively depends on."""
    return sorted(_walk(graph, ds, downstream=False, max_depth=max_depth), key=str)


def relatives(graph: Graph, ds: DatasetId, *, max_depth: int = MAX_DEPTH) -> list[DatasetId]:
    """Ancestors and descendants together — the full lineage context of one dataset."""
    both = set(ancestors(graph, ds, max_depth=max_depth))
    both |= set(descendants(graph, ds, max_depth=max_depth))
    return sorted(both - {ds}, key=str)


def reachable(
    graph: Graph, seeds: Iterable[DatasetId], *, downstream: bool = True, max_depth: int = MAX_DEPTH
) -> list[DatasetId]:
    """Union of reachability from several seeds, seeds included."""
    out: set[DatasetId] = set()
    for seed in seeds:
        out.add(seed)
        out.update(_walk(graph, seed, downstream=downstream, max_depth=max_depth))
    return sorted(out, key=str)


def blast_radius(graph: Graph, ds: DatasetId, *, max_depth: int = MAX_DEPTH) -> int:
    """How many datasets a change to `ds` can reach.

    The number to put in a pull request comment when someone edits a model.
    """
    return len(descendants(graph, ds, max_depth=max_depth))


def is_upstream_of(graph: Graph, candidate: DatasetId, target: DatasetId) -> bool:
    """True when `candidate` feeds `target`, directly or transitively."""
    return candidate in _walk(graph, target, downstream=False, max_depth=MAX_DEPTH)


def is_downstream_of(graph: Graph, candidate: DatasetId, target: DatasetId) -> bool:
    """True when `candidate` is fed by `target`, directly or transitively."""
    return candidate in _walk(graph, target, downstream=True, max_depth=MAX_DEPTH)


def has_path(graph: Graph, src: DatasetId, dst: DatasetId) -> bool:
    """True when data can flow from `src` to `dst`."""
    return is_downstream_of(graph, dst, src)


def distance(graph: Graph, src: DatasetId, dst: DatasetId) -> int | None:
    """Fewest hops from `src` to `dst`, or None when unreachable."""
    if src == dst:
        return 0
    return _walk(graph, src, downstream=True, max_depth=MAX_DEPTH).get(dst)


# -- paths ---------------------------------------------------------------------


def paths_between(
    graph: Graph, src: DatasetId, dst: DatasetId, *, max_depth: int = 12, limit: int = 256
) -> list[list[DatasetId]]:
    """Every simple path from `src` to `dst`, shortest first.

    Bounded twice on purpose. A diamond-heavy warehouse graph has exponentially many
    paths, and a caller asking "how does this reach that" wants a handful of
    representative routes, not all of them.
    """
    found: list[list[DatasetId]] = []
    stack: list[tuple[DatasetId, list[DatasetId]]] = [(src, [src])]
    while stack and len(found) < limit:
        node, path = stack.pop(0)
        if len(path) > max_depth:
            continue
        for child in children(graph, node):
            if child == dst:
                found.append([*path, child])
                if len(found) >= limit:
                    break
            elif child not in path:
                stack.append((child, [*path, child]))
    return sorted(found, key=lambda p: (len(p), [str(n) for n in p]))


def shortest_path(graph: Graph, src: DatasetId, dst: DatasetId) -> list[DatasetId] | None:
    """One shortest route from `src` to `dst`, or None when there is none."""
    if src == dst:
        return [src]
    frontier: deque[list[DatasetId]] = deque([[src]])
    visited = {src}
    while frontier:
        path = frontier.popleft()
        for child in children(graph, path[-1]):
            if child == dst:
                return [*path, child]
            if child in visited:
                continue
            visited.add(child)
            frontier.append([*path, child])
    return None


def between(graph: Graph, src: DatasetId, dst: DatasetId) -> list[DatasetId]:
    """Every dataset lying on some path from `src` to `dst`, endpoints included."""
    forward = set(descendants(graph, src)) | {src}
    backward = set(ancestors(graph, dst)) | {dst}
    return sorted(forward & backward, key=str)


def effective_mapping(graph: Graph, path: Sequence[DatasetId]) -> PartitionMapping:
    """Collapse a whole path into the single partition mapping it implies.

    Composing edge by edge is what makes "one dirty day at the source" resolve to a
    concrete set of months at the far end. Where a hop has several parallel edges
    they are joined first, which widens — two disagreeing accounts of the same
    dependency must be covered by the result, not arbitrated between.
    """
    if len(path) < 2:
        return PartitionMapping.identity(graph.spec(path[0])) if path else PartitionMapping()

    accumulated: PartitionMapping | None = None
    for src, dst in zip(path, path[1:], strict=False):
        hop = edge_between(graph, src, dst)
        if hop is None:
            return PartitionMapping.unknown(graph.spec(path[-1]))
        accumulated = hop if accumulated is None else compose(accumulated, hop)
    return accumulated if accumulated is not None else PartitionMapping()


# -- edges ---------------------------------------------------------------------


def edges_between(graph: Graph, src: DatasetId, dst: DatasetId) -> list[Edge]:
    """All parallel edges from `src` to `dst`, sorted by evidence."""
    return sorted((e for e in graph.out_edges(src) if e.dst == dst), key=lambda e: e.evidence)


def edge_between(graph: Graph, src: DatasetId, dst: DatasetId) -> PartitionMapping | None:
    """One mapping covering every parallel edge from `src` to `dst`.

    Parallel edges arise when the same dependency is learned twice — a dbt manifest
    and a query log both reporting it. Joining rather than picking keeps the result
    an over-approximation of both accounts.
    """
    found = edges_between(graph, src, dst)
    if not found:
        return None
    merged = found[0].mapping
    for edge in found[1:]:
        merged = join(merged, edge.mapping)
    return merged


# -- shape ---------------------------------------------------------------------


def roots(graph: Graph) -> list[DatasetId]:
    """Datasets with no upstream in this graph — the sources."""
    return sorted((ds for ds in graph.datasets if not graph.in_edges(ds)), key=str)


def leaves(graph: Graph) -> list[DatasetId]:
    """Datasets with no downstream — the things people actually consume."""
    return sorted((ds for ds in graph.datasets if not graph.out_edges(ds)), key=str)


def isolated(graph: Graph) -> list[DatasetId]:
    """Datasets with no edges at all.

    Usually a normalization failure: the same table spelled two ways, so one copy
    holds all the edges and the other floats free.
    """
    return sorted(
        (ds for ds in graph.datasets if not graph.in_edges(ds) and not graph.out_edges(ds)),
        key=str,
    )


def is_isolated(graph: Graph, ds: DatasetId) -> bool:
    """True when nothing connects to `ds` in either direction."""
    return not graph.in_edges(ds) and not graph.out_edges(ds)


def depth_of(graph: Graph, ds: DatasetId) -> int:
    """Longest distance from any root to `ds`. Roots are depth 0."""
    return max((len(p) - 1 for p in _root_paths(graph, ds)), default=0)


def height_of(graph: Graph, ds: DatasetId) -> int:
    """Longest distance from `ds` to any leaf."""
    best = 0
    for leaf in leaves(graph):
        for path in paths_between(graph, ds, leaf, limit=32):
            best = max(best, len(path) - 1)
    return best


def _root_paths(graph: Graph, ds: DatasetId) -> list[list[DatasetId]]:
    found: list[list[DatasetId]] = []
    for root in roots(graph):
        if root == ds:
            found.append([ds])
            continue
        found.extend(paths_between(graph, root, ds, limit=32))
    return found


def levels(graph: Graph) -> dict[int, list[DatasetId]]:
    """Datasets bucketed by depth from the sources.

    A build schedule in the simplest possible form: everything at level *n* can run
    once level *n-1* has finished.
    """
    out: dict[int, list[DatasetId]] = defaultdict(list)
    for ds in graph.datasets:
        out[depth_of(graph, ds)].append(ds)
    return {k: sorted(v, key=str) for k, v in sorted(out.items())}


def topological_order(graph: Graph) -> list[DatasetId]:
    """Build order over the whole graph. Datasets in a cycle come last, by name."""
    indegree = {ds: 0 for ds in graph.datasets}
    for ds in graph.datasets:
        for edge in graph.out_edges(ds):
            if edge.dst != ds:
                indegree[edge.dst] += 1

    ready = deque(sorted((ds for ds, n in indegree.items() if n == 0), key=str))
    out: list[DatasetId] = []
    while ready:
        node = ready.popleft()
        out.append(node)
        for child in children(graph, node):
            if child == node:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return out + sorted(set(graph.datasets) - set(out), key=str)


def cycles(graph: Graph) -> list[list[DatasetId]]:
    """Strongly connected components of size >1, plus any self-loop.

    A cycle is not necessarily a bug — an incremental model reading its own output
    is idiomatic — but it is always something the planner has to widen around, so
    naming them is how a user understands a surprising rebuild.
    """
    index: dict[DatasetId, int] = {}
    low: dict[DatasetId, int] = {}
    stack: list[DatasetId] = []
    on_stack: set[DatasetId] = set()
    found: list[list[DatasetId]] = []
    counter = 0

    def strongconnect(root: DatasetId) -> None:
        nonlocal counter
        # Explicit stack: warehouse graphs are wide but a recursive Tarjan still
        # blows the interpreter limit on a long enough chain.
        work: list[tuple[DatasetId, int]] = [(root, 0)]
        while work:
            node, child_index = work[-1]
            if child_index == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            kids = children(graph, node)
            recursed = False
            while child_index < len(kids):
                kid = kids[child_index]
                child_index += 1
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
            if child_index >= len(kids):
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component: list[DatasetId] = []
                    while True:
                        popped = stack.pop()
                        on_stack.discard(popped)
                        component.append(popped)
                        if popped == node:
                            break
                    if len(component) > 1:
                        found.append(sorted(component, key=str))

    for ds in graph.datasets:
        if ds not in index:
            strongconnect(ds)

    for ds in graph.datasets:
        if any(e.dst == ds for e in graph.out_edges(ds)):
            found.append([ds])

    return sorted(found, key=lambda c: [str(n) for n in c])


def has_cycle(graph: Graph) -> bool:
    """True when any dataset can reach itself."""
    return bool(cycles(graph))


def connected_components(graph: Graph) -> list[list[DatasetId]]:
    """Weakly connected components, largest first.

    More than one usually means normalization failed somewhere: a real warehouse is
    almost always one connected mass plus a few genuinely standalone tables.
    """
    remaining = set(graph.datasets)
    out: list[list[DatasetId]] = []
    while remaining:
        seed = min(remaining, key=str)
        component = {seed}
        frontier = [seed]
        while frontier:
            node = frontier.pop()
            for nxt in neighbors(graph, node):
                if nxt not in component:
                    component.add(nxt)
                    frontier.append(nxt)
        remaining -= component
        out.append(sorted(component, key=str))
    return sorted(out, key=lambda c: (-len(c), str(c[0])))


def common_ancestors(graph: Graph, a: DatasetId, b: DatasetId) -> list[DatasetId]:
    """Datasets feeding both `a` and `b`.

    The first question asked when two dashboards disagree.
    """
    return sorted(set(ancestors(graph, a)) & set(ancestors(graph, b)), key=str)


def common_descendants(graph: Graph, a: DatasetId, b: DatasetId) -> list[DatasetId]:
    """Datasets fed by both `a` and `b` — where two changes will collide."""
    return sorted(set(descendants(graph, a)) & set(descendants(graph, b)), key=str)


def lowest_common_ancestors(graph: Graph, a: DatasetId, b: DatasetId) -> list[DatasetId]:
    """Common ancestors nearest the pair, by summed hop distance.

    The nearest shared input is the one worth investigating; a shared root six hops
    up is true and unhelpful.
    """
    shared = common_ancestors(graph, a, b)
    if not shared:
        return []
    scored: list[tuple[int, DatasetId]] = []
    for node in shared:
        da, db = distance(graph, node, a), distance(graph, node, b)
        if da is None or db is None:
            continue
        scored.append((da + db, node))
    if not scored:
        return []
    best = min(score for score, _ in scored)
    return sorted((node for score, node in scored if score == best), key=str)


# -- column level --------------------------------------------------------------


def columns_of(graph: Graph, ds: DatasetId) -> list[str]:
    """Column names the graph knows about for `ds`, from edges in either direction.

    A graph built purely from dataset-level lineage reports nothing here, which is
    honest: the edges carry no column detail to report.
    """
    found: set[str] = set()
    for edge in graph.in_edges(ds):
        found.update(target for _, target in edge.columns)
    for edge in graph.out_edges(ds):
        found.update(source for source, _ in edge.columns)
    return sorted(found)


def column_descendants(graph: Graph, ref: ColumnRef, *, max_depth: int = 12) -> list[ColumnRef]:
    """Columns derived from `ref`, transitively.

    What a column-level impact analysis answers: rename this field and these are the
    downstream fields that stop working.
    """
    seen: set[ColumnRef] = set()
    frontier = [(ref, 0)]
    while frontier:
        current, depth = frontier.pop()
        if depth >= max_depth:
            continue
        for edge in graph.out_edges(current.dataset):
            for source, target in edge.columns:
                if source != current.column:
                    continue
                nxt = ColumnRef(edge.dst, target)
                if nxt in seen:
                    continue
                seen.add(nxt)
                frontier.append((nxt, depth + 1))
    return sorted(seen - {ref}, key=str)


def column_ancestors(graph: Graph, ref: ColumnRef, *, max_depth: int = 12) -> list[ColumnRef]:
    """Columns `ref` is derived from, transitively."""
    seen: set[ColumnRef] = set()
    frontier = [(ref, 0)]
    while frontier:
        current, depth = frontier.pop()
        if depth >= max_depth:
            continue
        for edge in graph.in_edges(current.dataset):
            for source, target in edge.columns:
                if target != current.column:
                    continue
                nxt = ColumnRef(edge.src, source)
                if nxt in seen:
                    continue
                seen.add(nxt)
                frontier.append((nxt, depth + 1))
    return sorted(seen - {ref}, key=str)


def column_paths(graph: Graph, ref: ColumnRef, *, max_depth: int = 6) -> list[list[ColumnRef]]:
    """Column-level routes feeding `ref`, nearest first."""
    return graph.upstream_columns(ref, max_depth=max_depth)


# -- selection and reshaping ---------------------------------------------------


def find(graph: Graph, pattern: str) -> list[DatasetId]:
    """Datasets whose string form matches a glob.

    ``find(g, "*.gold_*")`` and ``find(g, "s3://lake/*")`` both work, because a
    `DatasetId` renders as `namespace/name` and globs do not care which is which.
    """
    return sorted((ds for ds in graph.datasets if fnmatch.fnmatch(str(ds), pattern)), key=str)


def select(graph: Graph, predicate: Callable[[DatasetId], bool]) -> list[DatasetId]:
    """Datasets satisfying an arbitrary predicate."""
    return sorted((ds for ds in graph.datasets if predicate(ds)), key=str)


def namespaces(graph: Graph) -> list[str]:
    """Distinct namespaces present, which is roughly the systems in play."""
    return sorted({ds.namespace for ds in graph.datasets})


def in_namespace(graph: Graph, namespace: str) -> list[DatasetId]:
    """Datasets belonging to one system."""
    return sorted((ds for ds in graph.datasets if ds.namespace == namespace), key=str)


def partitioned_datasets(graph: Graph) -> list[DatasetId]:
    """Datasets with a declared partition spec — the ones the planner can be precise about."""
    return sorted((ds for ds in graph.datasets if graph.spec(ds).fields), key=str)


def unpartitioned_datasets(graph: Graph) -> list[DatasetId]:
    """Datasets the planner can only ever rebuild whole."""
    return sorted((ds for ds in graph.datasets if not graph.spec(ds).fields), key=str)


def subgraph(graph: Graph, datasets: Iterable[DatasetId]) -> Graph:
    """The induced subgraph over `datasets`, keeping only edges with both ends inside."""
    keep = set(datasets)
    out = Graph()
    for ds in sorted(keep, key=str):
        out.add_dataset(ds, graph.spec(ds))
    for edge in graph.edges:
        if edge.src in keep and edge.dst in keep:
            out.add_edge(edge)
    return out


def upstream_subgraph(graph: Graph, ds: DatasetId, *, max_depth: int = MAX_DEPTH) -> Graph:
    """Everything feeding `ds`, as a graph you can plan or render on its own."""
    return subgraph(graph, [ds, *ancestors(graph, ds, max_depth=max_depth)])


def downstream_subgraph(graph: Graph, ds: DatasetId, *, max_depth: int = MAX_DEPTH) -> Graph:
    """Everything `ds` reaches, as a standalone graph."""
    return subgraph(graph, [ds, *descendants(graph, ds, max_depth=max_depth)])


def prune(graph: Graph, keep: Iterable[DatasetId]) -> Graph:
    """Alias for `subgraph`, named for the case where you are dropping rather than picking."""
    return subgraph(graph, keep)


def without(graph: Graph, datasets: Iterable[DatasetId]) -> Graph:
    """The graph with `datasets` and their edges removed."""
    drop = set(datasets)
    return subgraph(graph, [ds for ds in graph.datasets if ds not in drop])


def reverse(graph: Graph) -> Graph:
    """Every edge flipped.

    Mappings do not invert — a monthly rollup cannot be run backwards to name the
    days that produced it — so reversed edges carry `PartitionMapping.unknown`.
    The result is for reachability questions, not for planning.
    """
    out = Graph()
    for ds in graph.datasets:
        out.add_dataset(ds, graph.spec(ds))
    for edge in graph.edges:
        out.add_edge(
            Edge(
                src=edge.dst,
                dst=edge.src,
                mapping=PartitionMapping.unknown(graph.spec(edge.src)),
                columns=tuple((t, s) for s, t in edge.columns),
                evidence=f"reversed:{edge.evidence}",
            )
        )
    return out


def copy_graph(graph: Graph) -> Graph:
    """A shallow copy. Datasets and edges are frozen, so this is safe to mutate."""
    return subgraph(graph, graph.datasets)


def merge_graphs(*graphs: Graph) -> Graph:
    """Union several graphs.

    Specs merge by the same rule `Graph.add_dataset` uses — a real spec is never
    overwritten by the unpartitioned default — so merging a query-log graph into a
    catalog graph keeps the catalog's partitioning.
    """
    out = Graph()
    specs: dict[DatasetId, PartitionSpec] = {}
    for graph in graphs:
        for ds in graph.datasets:
            spec = graph.spec(ds)
            if spec.fields or ds not in specs:
                specs.setdefault(ds, spec)
                if spec.fields:
                    specs[ds] = spec
    for ds, spec in sorted(specs.items(), key=lambda kv: str(kv[0])):
        out.add_dataset(ds, spec)
    seen: set[tuple[DatasetId, DatasetId, str]] = set()
    for graph in graphs:
        for edge in graph.edges:
            key = (edge.src, edge.dst, edge.evidence)
            if key in seen:
                continue
            seen.add(key)
            out.add_edge(edge)
    return out


def dataset_index(graph: Graph) -> Mapping[str, DatasetId]:
    """Map every dataset's string form to its identity, for lookup by name."""
    return {str(ds): ds for ds in graph.datasets}
