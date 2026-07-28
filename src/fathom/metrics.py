"""Measuring the graph, and the honesty of what it claims.

Two different questions live here, and conflating them is how lineage tools end up
looking impressive and being useless:

- **Shape.** How big, how deep, how tangled. Descriptive, and mostly useful for
  sizing a rendering or spotting a normalization failure.
- **Coverage.** What fraction of the graph carries information precise enough to
  act on. An edge with an `UNBOUNDED` mapping is in the graph and contributes
  nothing to a plan; a dataset with no partition spec can only ever be rebuilt
  whole. A tool reporting "40,000 edges" while 90% of them are unbounded is
  reporting its own inventory, not your lineage.

`coverage()` is the number to publish. It goes up when specs are declared and
column lineage is extracted, and it predicts precisely how much compute a plan can
save — which is the only claim this project makes that money rides on.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .graph import Graph, InvalidationPlan
from .partitions import Passthrough, TimeWindow
from .query import (
    ancestors,
    connected_components,
    cycles,
    depth_of,
    descendants,
    isolated,
    leaves,
    levels,
    roots,
)
from .types import DatasetId

__all__ = [
    "Coverage",
    "GraphStats",
    "HealthReport",
    "average_degree",
    "bottlenecks",
    "bounded_edge_ratio",
    "column_lineage_coverage",
    "connectivity",
    "coverage",
    "degree_centrality",
    "density",
    "diameter",
    "evidence_breakdown",
    "graph_stats",
    "health_report",
    "health_score",
    "hubs",
    "longest_chain",
    "most_depended_on",
    "namespace_breakdown",
    "plan_efficiency",
    "precision_ceiling",
    "reach_score",
    "spec_coverage",
    "suspicious_datasets",
    "width",
]


# -- shape ---------------------------------------------------------------------


@dataclass(frozen=True)
class GraphStats:
    """Descriptive statistics for one graph."""

    datasets: int = 0
    edges: int = 0
    roots: int = 0
    leaves: int = 0
    isolated: int = 0
    components: int = 0
    cycles: int = 0
    max_depth: int = 0
    max_width: int = 0
    namespaces: int = 0

    def summary(self) -> str:
        parts = [
            f"{self.datasets} dataset(s)",
            f"{self.edges} edge(s)",
            f"depth {self.max_depth}",
            f"width {self.max_width}",
            f"{self.namespaces} namespace(s)",
        ]
        if self.components > 1:
            parts.append(f"{self.components} disconnected component(s)")
        if self.isolated:
            parts.append(f"{self.isolated} isolated")
        if self.cycles:
            parts.append(f"{self.cycles} cycle(s)")
        return ", ".join(parts)


def graph_stats(graph: Graph) -> GraphStats:
    """Everything descriptive about a graph, in one pass a caller can print."""
    by_level = levels(graph)
    return GraphStats(
        datasets=len(graph.datasets),
        edges=len(graph.edges),
        roots=len(roots(graph)),
        leaves=len(leaves(graph)),
        isolated=len(isolated(graph)),
        components=len(connected_components(graph)),
        cycles=len(cycles(graph)),
        max_depth=max(by_level, default=0),
        max_width=max((len(v) for v in by_level.values()), default=0),
        namespaces=len({ds.namespace for ds in graph.datasets}),
    )


def density(graph: Graph) -> float:
    """Edges as a fraction of the maximum possible for this many datasets.

    Near zero for a healthy warehouse. A high value usually means dataset-level
    edges were emitted where column-level ones were meant.
    """
    n = len(graph.datasets)
    if n < 2:
        return 0.0
    return len(graph.edges) / (n * (n - 1))


def average_degree(graph: Graph) -> float:
    """Mean number of edges touching a dataset."""
    n = len(graph.datasets)
    return 0.0 if n == 0 else 2 * len(graph.edges) / n


def diameter(graph: Graph) -> int:
    """The longest source-to-consumer chain, in hops.

    The lower bound on how many sequential build steps a full refresh takes.
    """
    return max((depth_of(graph, ds) for ds in graph.datasets), default=0)


def width(graph: Graph) -> int:
    """The largest number of datasets that can be built in parallel at one level."""
    return max((len(v) for v in levels(graph).values()), default=0)


def longest_chain(graph: Graph) -> list[DatasetId]:
    """One deepest path through the graph, source to consumer.

    This is the critical path of a full rebuild: shortening it is the only thing
    that reduces wall-clock time when everything else is already parallel.
    """
    from .query import paths_between

    best: list[DatasetId] = []
    for root in roots(graph):
        for leaf in leaves(graph):
            for path in paths_between(graph, root, leaf, limit=16):
                if len(path) > len(best):
                    best = path
    return best


def connectivity(graph: Graph) -> float:
    """Fraction of datasets in the largest connected component.

    Below about 0.8 in a real warehouse, suspect identity normalization rather than
    genuinely independent pipelines.
    """
    components = connected_components(graph)
    if not components or not graph.datasets:
        return 0.0
    return len(components[0]) / len(graph.datasets)


# -- importance ----------------------------------------------------------------


def degree_centrality(graph: Graph) -> dict[DatasetId, float]:
    """Each dataset's neighbour count, normalized by the largest possible."""
    n = len(graph.datasets)
    if n < 2:
        return {ds: 0.0 for ds in graph.datasets}
    out: dict[DatasetId, float] = {}
    for ds in graph.datasets:
        neighbours = {e.src for e in graph.in_edges(ds)} | {e.dst for e in graph.out_edges(ds)}
        out[ds] = len(neighbours - {ds}) / (n - 1)
    return out


def reach_score(graph: Graph) -> dict[DatasetId, int]:
    """How many datasets each one can reach downstream.

    The ranking to use when deciding what to profile first: breaking the top of this
    list breaks the most things.
    """
    return {ds: len(descendants(graph, ds)) for ds in graph.datasets}


def most_depended_on(graph: Graph, *, limit: int = 10) -> list[tuple[DatasetId, int]]:
    """Datasets with the most transitive consumers, highest first."""
    scored = sorted(reach_score(graph).items(), key=lambda kv: (-kv[1], str(kv[0])))
    return [(ds, n) for ds, n in scored[:limit] if n > 0]


def hubs(graph: Graph, *, limit: int = 10) -> list[tuple[DatasetId, int]]:
    """Datasets with the most direct neighbours — where the graph is knotted."""
    scored = [
        (
            ds,
            len(
                ({e.src for e in graph.in_edges(ds)} | {e.dst for e in graph.out_edges(ds)}) - {ds}
            ),
        )
        for ds in graph.datasets
    ]
    scored.sort(key=lambda kv: (-kv[1], str(kv[0])))
    return scored[:limit]


def bottlenecks(graph: Graph, *, limit: int = 10) -> list[tuple[DatasetId, int]]:
    """Datasets every route between two large regions must pass through.

    Approximated by the product of transitive ancestors and descendants, which is
    the number of source-to-consumer pairs whose lineage this dataset sits on.
    """
    scored: list[tuple[DatasetId, int]] = []
    for ds in graph.datasets:
        up, down = len(ancestors(graph, ds)), len(descendants(graph, ds))
        if up and down:
            scored.append((ds, up * down))
    scored.sort(key=lambda kv: (-kv[1], str(kv[0])))
    return scored[:limit]


# -- coverage ------------------------------------------------------------------


@dataclass(frozen=True)
class Coverage:
    """How much of the graph carries information precise enough to plan with."""

    datasets: int = 0
    specced: int = 0
    edges: int = 0
    bounded_edges: int = 0
    column_edges: int = 0
    fields: int = 0
    bounded_fields: int = 0

    @property
    def spec_ratio(self) -> float:
        """Fraction of datasets whose partitioning we know."""
        return 0.0 if self.datasets == 0 else self.specced / self.datasets

    @property
    def edge_ratio(self) -> float:
        """Fraction of edges carrying at least one bounded field mapping."""
        return 0.0 if self.edges == 0 else self.bounded_edges / self.edges

    @property
    def column_ratio(self) -> float:
        """Fraction of edges carrying column-level detail."""
        return 0.0 if self.edges == 0 else self.column_edges / self.edges

    @property
    def field_ratio(self) -> float:
        """Fraction of individual field mappings that are provably bounded.

        The single most predictive number in this dataclass: it is very close to the
        ceiling on how much of a rebuild a plan can skip.
        """
        return 0.0 if self.fields == 0 else self.bounded_fields / self.fields

    def summary(self) -> str:
        return (
            f"coverage: {self.spec_ratio:.0%} of datasets specced, "
            f"{self.edge_ratio:.0%} of edges bounded, "
            f"{self.column_ratio:.0%} column-level, "
            f"{self.field_ratio:.0%} of field mappings provable"
        )


def coverage(graph: Graph) -> Coverage:
    """Measure how much of the graph is precise enough to be worth planning against."""
    specced = sum(1 for ds in graph.datasets if graph.spec(ds).fields)
    bounded_edges = 0
    column_edges = 0
    fields = 0
    bounded_fields = 0
    for edge in graph.edges:
        if edge.columns:
            column_edges += 1
        any_bounded = False
        for _, fm in edge.mapping.fields:
            fields += 1
            if isinstance(fm, TimeWindow | Passthrough):
                bounded_fields += 1
                any_bounded = True
        if any_bounded:
            bounded_edges += 1
    return Coverage(
        datasets=len(graph.datasets),
        specced=specced,
        edges=len(graph.edges),
        bounded_edges=bounded_edges,
        column_edges=column_edges,
        fields=fields,
        bounded_fields=bounded_fields,
    )


def spec_coverage(graph: Graph) -> float:
    """Fraction of datasets with a declared partition spec."""
    return coverage(graph).spec_ratio


def bounded_edge_ratio(graph: Graph) -> float:
    """Fraction of edges the planner can be precise across."""
    return coverage(graph).edge_ratio


def column_lineage_coverage(graph: Graph) -> float:
    """Fraction of edges carrying column-level detail.

    Drives two things directly: whether drift attribution can name a column, and
    whether a PII label propagates to a column or only to a dataset.
    """
    return coverage(graph).column_ratio


def precision_ceiling(graph: Graph) -> float:
    """An upper bound on the fraction of a rebuild any plan could skip.

    Not a promise — the actual saving depends on how much really changed. It is the
    number that says whether more spec declaration is worth the effort before
    anyone runs a shadow week.
    """
    return coverage(graph).field_ratio


def evidence_breakdown(graph: Graph) -> dict[str, int]:
    """Edge counts by evidence prefix — `sql`, `dbt`, `openlineage`, `declared`.

    Reading heavily `declared` means the graph is hand-maintained and will rot.
    """
    counter: Counter[str] = Counter()
    for edge in graph.edges:
        counter[edge.evidence.split(":", 1)[0]] += 1
    return dict(sorted(counter.items()))


def namespace_breakdown(graph: Graph) -> dict[str, int]:
    """Dataset counts per namespace — how much of each system is represented."""
    counter: Counter[str] = Counter(ds.namespace for ds in graph.datasets)
    return dict(sorted(counter.items()))


# -- health --------------------------------------------------------------------


def suspicious_datasets(graph: Graph) -> dict[str, list[DatasetId]]:
    """Datasets that look like extraction or normalization failures.

    Not errors — every category here is legitimate sometimes. They are the places to
    look first when a plan is surprisingly wide or a table is missing from it.
    """
    out: dict[str, list[DatasetId]] = defaultdict(list)
    for ds in graph.datasets:
        if not graph.in_edges(ds) and not graph.out_edges(ds):
            out["isolated"].append(ds)
        if graph.in_edges(ds) and not graph.spec(ds).fields:
            out["derived_without_spec"].append(ds)
        if len(graph.in_edges(ds)) > 20:
            out["very_wide_join"].append(ds)
        if any(e.dst == ds for e in graph.out_edges(ds)):
            out["self_referencing"].append(ds)
    return {k: sorted(v, key=str) for k, v in sorted(out.items())}


def health_score(graph: Graph) -> float:
    """A single 0–1 number blending coverage and connectivity.

    Deliberately blunt. It exists so a team can watch one number move week over
    week; anything it flags should be diagnosed with `health_report`.
    """
    if not graph.datasets:
        return 0.0
    c = coverage(graph)
    parts = [
        (c.spec_ratio, 0.3),
        (c.edge_ratio, 0.3),
        (c.column_ratio, 0.2),
        (connectivity(graph), 0.2),
    ]
    return round(sum(value * weight for value, weight in parts), 4)


@dataclass
class HealthReport:
    """Coverage, shape, and the specific things worth fixing."""

    score: float = 0.0
    stats: GraphStats = field(default_factory=GraphStats)
    coverage: Coverage = field(default_factory=Coverage)
    suspicious: dict[str, list[DatasetId]] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"lineage health: {self.score:.0%}",
            f"  {self.stats.summary()}",
            f"  {self.coverage.summary()}",
        ]
        for note in self.recommendations:
            lines.append(f"  → {note}")
        return "\n".join(lines)


def health_report(graph: Graph) -> HealthReport:
    """Health, with recommendations ordered by how much precision each would buy."""
    c = coverage(graph)
    suspicious = suspicious_datasets(graph)
    notes: list[str] = []

    unspecced = c.datasets - c.specced
    if unspecced:
        notes.append(
            f"declare partition specs for {unspecced} dataset(s); "
            "each one is currently rebuilt whole"
        )
    if c.edges and c.edge_ratio < 0.5:
        notes.append(
            f"{c.edges - c.bounded_edges} edge(s) have no provable partition mapping; "
            "these widen every plan that crosses them"
        )
    if c.edges and c.column_ratio < 0.5:
        notes.append(
            "under half of edges carry column lineage; drift attribution and PII "
            "propagation will name datasets rather than columns"
        )
    if len(connected_components(graph)) > 1:
        notes.append(
            f"{len(connected_components(graph))} disconnected components; "
            "check identity normalization before assuming they are unrelated"
        )
    if suspicious.get("isolated"):
        notes.append(f"{len(suspicious['isolated'])} dataset(s) have no edges at all")

    return HealthReport(
        score=health_score(graph),
        stats=graph_stats(graph),
        coverage=c,
        suspicious=suspicious,
        recommendations=notes,
    )


# -- plans ---------------------------------------------------------------------


def plan_efficiency(graph: Graph, plan: InvalidationPlan) -> dict[str, float]:
    """How much of the graph a plan avoided, and how much precision it lost.

    `widened_ratio` is the diagnostic: a plan touching few datasets but widening
    most of them is one bad edge away from rebuilding the warehouse.
    """
    total = len(graph.datasets)
    affected = len(plan.dirty)
    widened = len(plan.widened)
    return {
        "datasets_total": float(total),
        "datasets_affected": float(affected),
        "datasets_skipped": float(total - affected),
        "skip_ratio": 0.0 if total == 0 else (total - affected) / total,
        "widened_ratio": 0.0 if affected == 0 else widened / affected,
        "partitions": float(sum(len(keys) for keys in plan.dirty.values())),
    }
