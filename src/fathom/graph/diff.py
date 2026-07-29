"""Comparing two versions of the same artifact.

A lineage graph is only useful if you notice when it changes. The interesting
question in review is never "what does the graph look like" but "what did this pull
request do to it" — an edge that disappeared means a dependency the planner will
stop propagating through, and a mapping that widened means partitions that used to
be skipped will now be rebuilt.

Structural only. Two profiles disagreeing about a column's type is
`observe.schema`; two label sets disagreeing is `govern.diff`. Each sits in the layer
that owns the thing being compared, so this module needs to know nothing about
profiles or policy.

Everything here returns a dataclass with an `is_empty` and a `summary()`, so a CI
job can gate on the first and post the second.

Direction matters in the naming. `before` is the committed state, `after` is the
proposal. `added` means present in `after` only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..core.partitions import PartitionMapping, leq
from ..core.types import DatasetId, PartitionField, PartitionSpec
from ..core.util import markdown as md
from .model import Edge, Graph, InvalidationPlan

__all__ = [
    "EdgeChange",
    "GraphDiff",
    "PlanDiff",
    "SpecChange",
    "diff_graphs",
    "diff_plans",
    "any_narrowing",
    "changed_datasets",
    "diff_specs",
    "edge_key",
    "mapping_widened",
    "mapping_narrowed",
    "review_comment",
]


def edge_key(edge: Edge) -> tuple[str, str, str]:
    """The identity an edge is compared on: endpoints plus how we learned it.

    Evidence is part of the key because the same dependency learned from a dbt
    manifest and from a query log are two independent claims, and losing one is a
    real change even when the other still covers it.
    """
    return (str(edge.src), str(edge.dst), edge.evidence)


def mapping_widened(before: PartitionMapping, after: PartitionMapping) -> bool:
    """True when `after` covers strictly more than `before`.

    A widening is the change that costs money: partitions that used to be skipped
    are now rebuilt. Worth failing a build over when it was not intended.
    """
    return leq(before, after) and not leq(after, before)


def mapping_narrowed(before: PartitionMapping, after: PartitionMapping) -> bool:
    """True when `after` covers strictly less than `before`.

    A narrowing is the change that risks correctness: partitions that used to be
    rebuilt no longer are. Always worth a human reading the reason.
    """
    return leq(after, before) and not leq(before, after)


@dataclass(frozen=True)
class EdgeChange:
    """One edge whose mapping or columns moved between two graph versions."""

    src: DatasetId
    dst: DatasetId
    evidence: str
    before: PartitionMapping
    after: PartitionMapping
    columns_before: tuple[tuple[str, str], ...] = ()
    columns_after: tuple[tuple[str, str], ...] = ()

    @property
    def widened(self) -> bool:
        """Edges whose mapping became less precise — the ones that cost compute."""
        return mapping_widened(self.before, self.after)

    @property
    def narrowed(self) -> bool:
        """Edges whose mapping became more precise."""
        return mapping_narrowed(self.before, self.after)

    @property
    def columns_changed(self) -> bool:
        """Edges whose column-level detail changed."""
        return set(self.columns_before) != set(self.columns_after)

    def __str__(self) -> str:
        direction = "widened" if self.widened else "narrowed" if self.narrowed else "changed"
        return (
            f"{self.src} -> {self.dst} [{self.evidence}] {direction}: {self.before} => {self.after}"
        )


@dataclass(frozen=True)
class SpecChange:
    """A dataset whose partition spec changed.

    Almost always more consequential than it looks: every mapping into or out of
    this dataset was derived against the old spec.
    """

    dataset: DatasetId
    before: PartitionSpec
    after: PartitionSpec

    @property
    def added_fields(self) -> tuple[PartitionField, ...]:
        """Fields present after and not before."""
        names = {f.name for f in self.before.fields}
        return tuple(f for f in self.after.fields if f.name not in names)

    @property
    def removed_fields(self) -> tuple[PartitionField, ...]:
        """Fields present before and not after; the breaking direction."""
        names = {f.name for f in self.after.fields}
        return tuple(f for f in self.before.fields if f.name not in names)

    @property
    def regrained(self) -> tuple[str, ...]:
        """Fields kept but re-grained — a day column that became a month, or the reverse."""
        after_by_name = {f.name: f for f in self.after.fields}
        return tuple(
            f.name
            for f in self.before.fields
            if f.name in after_by_name and after_by_name[f.name].grain != f.grain
        )

    def __str__(self) -> str:
        return f"{self.dataset}: {self.before.names} => {self.after.names}"


@dataclass
class GraphDiff:
    """Everything that moved between two graph versions."""

    added_datasets: list[DatasetId] = field(default_factory=list)
    removed_datasets: list[DatasetId] = field(default_factory=list)
    added_edges: list[Edge] = field(default_factory=list)
    removed_edges: list[Edge] = field(default_factory=list)
    changed_edges: list[EdgeChange] = field(default_factory=list)
    changed_specs: list[SpecChange] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when nothing changed."""
        return not (
            self.added_datasets
            or self.removed_datasets
            or self.added_edges
            or self.removed_edges
            or self.changed_edges
            or self.changed_specs
        )

    @property
    def widenings(self) -> list[EdgeChange]:
        """Mapping changes that will cost extra compute."""
        return [c for c in self.changed_edges if c.widened]

    @property
    def narrowings(self) -> list[EdgeChange]:
        """Mapping changes that could cause a partition to be missed."""
        return [c for c in self.changed_edges if c.narrowed]

    @property
    def is_safe(self) -> bool:
        """True when nothing narrowed and no edge vanished.

        The condition a merge gate should use: widening is expensive but correct,
        while narrowing and removal are the two ways a graph edit serves stale data.
        """
        return not self.narrowings and not self.removed_edges

    def summary(self) -> str:
        """The diff as text, suitable for a pull-request comment."""
        if self.is_empty:
            return "graph: no change"
        parts = [
            f"+{len(self.added_datasets)}/-{len(self.removed_datasets)} datasets",
            f"+{len(self.added_edges)}/-{len(self.removed_edges)} edges",
            f"{len(self.changed_edges)} mapping change(s)",
        ]
        if self.changed_specs:
            parts.append(f"{len(self.changed_specs)} spec change(s)")
        head = "graph: " + ", ".join(parts)
        if not self.is_safe:
            head += "  [UNSAFE: an edge was removed or narrowed]"
        return head


def diff_graphs(before: Graph, after: Graph) -> GraphDiff:
    """Compare two graph versions edge by edge."""
    before_ds, after_ds = set(before.datasets), set(after.datasets)
    out = GraphDiff(
        added_datasets=sorted(after_ds - before_ds, key=str),
        removed_datasets=sorted(before_ds - after_ds, key=str),
    )

    before_edges = {edge_key(e): e for e in before.edges}
    after_edges = {edge_key(e): e for e in after.edges}

    for key in sorted(set(after_edges) - set(before_edges)):
        out.added_edges.append(after_edges[key])
    for key in sorted(set(before_edges) - set(after_edges)):
        out.removed_edges.append(before_edges[key])
    for key in sorted(set(before_edges) & set(after_edges)):
        b, a = before_edges[key], after_edges[key]
        if b.mapping == a.mapping and set(b.columns) == set(a.columns):
            continue
        out.changed_edges.append(
            EdgeChange(
                src=a.src,
                dst=a.dst,
                evidence=a.evidence,
                before=b.mapping,
                after=a.mapping,
                columns_before=b.columns,
                columns_after=a.columns,
            )
        )

    for ds in sorted(before_ds & after_ds, key=str):
        if before.spec(ds) != after.spec(ds):
            out.changed_specs.append(SpecChange(ds, before.spec(ds), after.spec(ds)))
    return out


def diff_specs(before: PartitionSpec, after: PartitionSpec) -> SpecChange:
    """Compare two partition specs directly, without a graph around them."""
    return SpecChange(DatasetId("", ""), before, after)


# -- schemas -------------------------------------------------------------------


@dataclass
class PlanDiff:
    """How two rebuild plans differ. Used to grade a graph edit against its cost."""

    only_before: dict[DatasetId, frozenset[str]] = field(default_factory=dict)
    only_after: dict[DatasetId, frozenset[str]] = field(default_factory=dict)
    partition_delta: int = 0

    @property
    def is_empty(self) -> bool:
        """True when nothing changed."""
        return not self.only_before and not self.only_after

    def summary(self) -> str:
        """The diff as text, suitable for a pull-request comment."""
        if self.is_empty:
            return "plan: identical"
        direction = "more" if self.partition_delta > 0 else "fewer"
        return (
            f"plan: {abs(self.partition_delta)} {direction} partition(s), "
            f"{len(self.only_after)} dataset(s) newly affected, "
            f"{len(self.only_before)} no longer affected"
        )


def diff_plans(before: InvalidationPlan, after: InvalidationPlan) -> PlanDiff:
    """Compare two plans by rendered partition key, per dataset."""
    out = PlanDiff()
    b = {ds: frozenset(str(k) for k in keys) for ds, keys in before.dirty.items()}
    a = {ds: frozenset(str(k) for k in keys) for ds, keys in after.dirty.items()}
    for ds in sorted(set(b) | set(a), key=str):
        gone = b.get(ds, frozenset()) - a.get(ds, frozenset())
        new = a.get(ds, frozenset()) - b.get(ds, frozenset())
        if gone:
            out.only_before[ds] = gone
        if new:
            out.only_after[ds] = new
    out.partition_delta = sum(len(v) for v in a.values()) - sum(len(v) for v in b.values())
    return out


# -- labels --------------------------------------------------------------------


def changed_datasets(diff: GraphDiff) -> list[DatasetId]:
    """Every dataset touched by a graph diff, in any way."""
    touched: set[DatasetId] = set(diff.added_datasets) | set(diff.removed_datasets)
    for change in diff.changed_edges:
        touched.update({change.src, change.dst})
    for edge in [*diff.added_edges, *diff.removed_edges]:
        touched.update({edge.src, edge.dst})
    touched.update(change.dataset for change in diff.changed_specs)
    return sorted(touched, key=str)


def review_comment(diff: GraphDiff, *, limit: int = 20) -> str:
    """A Markdown summary sized for a pull request comment.

    Leads with the unsafe changes, because a reviewer skims the first three lines
    and a narrowed mapping buried under forty additions is a narrowing nobody read.
    """
    if diff.is_empty:
        return "**Lineage:** no change."

    lines = [f"**Lineage:** {diff.summary()}", ""]
    if diff.narrowings or diff.removed_edges:
        lines.append("#### Needs review")
        for change in diff.narrowings[:limit]:
            lines.append(
                f"- narrowed `{change.src}` → `{change.dst}`: `{change.before}` → `{change.after}`"
            )
        lines.append(
            md.bullets(
                f"removed {md.code(edge.src)} → {md.code(edge.dst)} [{edge.evidence}]"
                for edge in diff.removed_edges[:limit]
            )
            if diff.removed_edges
            else ""
        )
        lines.append("")
    if diff.widenings:
        lines.append("#### Will cost more compute")
        for change in diff.widenings[:limit]:
            lines.append(
                f"- widened `{change.src}` → `{change.dst}`: `{change.before}` → `{change.after}`"
            )
        lines.append("")
    if diff.added_edges:
        lines.append(f"#### Added ({len(diff.added_edges)} edge(s))")
        lines.append(
            md.bullets(
                f"{md.code(edge.src)} → {md.code(edge.dst)} "
                f"{md.code(edge.mapping)} [{edge.evidence}]"
                for edge in diff.added_edges[:limit]
            )
        )
    return "\n".join(lines)


def any_narrowing(diffs: Iterable[GraphDiff]) -> bool:
    """True when any diff in a sequence narrows a mapping. For a multi-repo gate."""
    return any(d.narrowings for d in diffs)
