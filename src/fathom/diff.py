"""Comparing two versions of the same artifact.

A lineage graph is only useful if you notice when it changes. The interesting
question in review is never "what does the graph look like" but "what did this pull
request do to it" — an edge that disappeared means a dependency the planner will
stop propagating through, and a mapping that widened means partitions that used to
be skipped will now be rebuilt.

Everything here returns a dataclass with an `is_empty` and a `summary()`, so a CI
job can gate on the first and post the second.

Direction matters in the naming. `before` is the committed state, `after` is the
proposal. `added` means present in `after` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .graph import Edge, Graph, InvalidationPlan
from .partitions import PartitionMapping, leq
from .policy import Label, LabelSet
from .profile import ColumnProfile, Profile, Severity
from .types import ColumnRef, DatasetId, PartitionField, PartitionSpec

__all__ = [
    "ColumnChange",
    "EdgeChange",
    "GraphDiff",
    "LabelDiff",
    "PlanDiff",
    "SchemaDiff",
    "SpecChange",
    "breaking_schema_changes",
    "diff_graphs",
    "diff_labels",
    "diff_plans",
    "diff_profiles",
    "diff_schemas",
    "any_narrowing",
    "changed_datasets",
    "diff_specs",
    "edge_key",
    "is_breaking",
    "mapping_widened",
    "mapping_narrowed",
    "review_comment",
    "worst_severity",
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
        return mapping_widened(self.before, self.after)

    @property
    def narrowed(self) -> bool:
        return mapping_narrowed(self.before, self.after)

    @property
    def columns_changed(self) -> bool:
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
        names = {f.name for f in self.before.fields}
        return tuple(f for f in self.after.fields if f.name not in names)

    @property
    def removed_fields(self) -> tuple[PartitionField, ...]:
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


@dataclass(frozen=True)
class ColumnChange:
    """One column whose type or nullability moved."""

    column: str
    before: ColumnProfile | None
    after: ColumnProfile | None

    @property
    def kind(self) -> str:
        if self.before is None:
            return "added"
        if self.after is None:
            return "removed"
        if self.before.dtype != self.after.dtype:
            return "retyped"
        return "changed"

    @property
    def is_breaking(self) -> bool:
        """A removal or a type change breaks a consumer; an addition does not."""
        return self.kind in {"removed", "retyped"}

    def __str__(self) -> str:
        if self.kind == "added":
            return f"+ {self.column}"
        if self.kind == "removed":
            return f"- {self.column}"
        b = self.before.dtype if self.before else "?"
        a = self.after.dtype if self.after else "?"
        return f"~ {self.column}: {b} => {a}"


@dataclass
class SchemaDiff:
    """Column-level differences between two profiles of the same dataset."""

    dataset: DatasetId
    changes: list[ColumnChange] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.changes

    @property
    def breaking(self) -> list[ColumnChange]:
        return [c for c in self.changes if c.is_breaking]

    @property
    def added(self) -> list[str]:
        return [c.column for c in self.changes if c.kind == "added"]

    @property
    def removed(self) -> list[str]:
        return [c.column for c in self.changes if c.kind == "removed"]

    @property
    def retyped(self) -> list[str]:
        return [c.column for c in self.changes if c.kind == "retyped"]

    def summary(self) -> str:
        if self.is_empty:
            return f"{self.dataset}: schema unchanged"
        verdict = " [BREAKING]" if self.breaking else ""
        return f"{self.dataset}: {len(self.changes)} column change(s){verdict}\n" + "\n".join(
            f"  {c}" for c in self.changes
        )


def diff_schemas(before: Profile, after: Profile) -> SchemaDiff:
    """Structural comparison of two profiles, ignoring statistics.

    Distinct from `profile.drift`, which asks whether the *values* moved. This asks
    whether the *shape* moved, which is the thing that breaks a downstream query
    outright rather than making its answer wrong.
    """
    out = SchemaDiff(dataset=after.dataset)
    names = sorted(set(before.column_names) | set(after.column_names))
    for name in names:
        b, a = before.column(name), after.column(name)
        if b is None or a is None or b.dtype != a.dtype:
            out.changes.append(ColumnChange(name, b, a))
    return out


def breaking_schema_changes(before: Profile, after: Profile) -> list[ColumnChange]:
    """Only the changes that break a consumer outright."""
    return diff_schemas(before, after).breaking


def is_breaking(before: Profile, after: Profile) -> bool:
    """True when any column was removed or retyped."""
    return bool(breaking_schema_changes(before, after))


# -- profiles ------------------------------------------------------------------


def diff_profiles(
    before: Profile,
    after: Profile,
    *,
    null_rate_tolerance: float = 0.05,
    row_count_tolerance: float = 0.25,
) -> tuple[SchemaDiff, list[str]]:
    """Schema differences and value drift together, as one review artifact.

    Returns the structural diff plus rendered drift lines. Both matter and they fail
    for different reasons, so they stay separate rather than merging into one list
    whose entries mean different things.
    """
    from .profile import drift

    findings = drift(
        before,
        after,
        null_rate_tolerance=null_rate_tolerance,
        row_count_tolerance=row_count_tolerance,
    )
    ordered = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
    lines = [str(f) for f in sorted(findings, key=lambda f: ordered[f.severity])]
    return diff_schemas(before, after), lines


# -- plans ---------------------------------------------------------------------


@dataclass
class PlanDiff:
    """How two rebuild plans differ. Used to grade a graph edit against its cost."""

    only_before: dict[DatasetId, frozenset[str]] = field(default_factory=dict)
    only_after: dict[DatasetId, frozenset[str]] = field(default_factory=dict)
    partition_delta: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.only_before and not self.only_after

    def summary(self) -> str:
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


@dataclass
class LabelDiff:
    """Label changes between two runs of inference and propagation."""

    added: dict[ColumnRef, set[Label]] = field(default_factory=dict)
    removed: dict[ColumnRef, set[Label]] = field(default_factory=dict)
    reconfidenced: list[tuple[ColumnRef, Label, Label]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.reconfidenced)

    @property
    def new_pii(self) -> list[ColumnRef]:
        """Columns that newly carry a personal-data label.

        The one line of this diff that belongs in a compliance review.
        """
        return sorted(
            (ref for ref, labels in self.added.items() if any(x.name == "pii" for x in labels)),
            key=str,
        )

    def summary(self) -> str:
        if self.is_empty:
            return "labels: no change"
        head = (
            f"labels: +{sum(len(v) for v in self.added.values())} "
            f"-{sum(len(v) for v in self.removed.values())}, "
            f"{len(self.reconfidenced)} confidence change(s)"
        )
        if self.new_pii:
            head += f"  [{len(self.new_pii)} column(s) newly labelled pii]"
        return head


def diff_labels(before: LabelSet, after: LabelSet) -> LabelDiff:
    """Compare two label sets by name, treating a confidence move as its own event."""
    out = LabelDiff()
    for ref in sorted(set(before) | set(after), key=str):
        b = {label.name: label for label in before.get(ref, set())}
        a = {label.name: label for label in after.get(ref, set())}
        added = {a[name] for name in set(a) - set(b)}
        removed = {b[name] for name in set(b) - set(a)}
        if added:
            out.added[ref] = added
        if removed:
            out.removed[ref] = removed
        for name in sorted(set(a) & set(b)):
            if a[name].confidence != b[name].confidence or a[name].confirmed != b[name].confirmed:
                out.reconfidenced.append((ref, b[name], a[name]))
    return out


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
        for edge in diff.removed_edges[:limit]:
            lines.append(f"- removed `{edge.src}` → `{edge.dst}` [{edge.evidence}]")
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
        for edge in diff.added_edges[:limit]:
            lines.append(f"- `{edge.src}` → `{edge.dst}` `{edge.mapping}` [{edge.evidence}]")
    return "\n".join(lines)


def any_narrowing(diffs: Iterable[GraphDiff]) -> bool:
    """True when any diff in a sequence narrows a mapping. For a multi-repo gate."""
    return any(d.narrowings for d in diffs)


def worst_severity(findings: Sequence[str]) -> str:
    """The highest severity present in rendered drift lines."""
    joined = " ".join(findings)
    for level in ("error", "warn", "info"):
        if f"[{level}]" in joined:
            return level
    return "none"
