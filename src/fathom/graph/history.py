"""The graph's own revision history.

`diff` compares two graphs you happen to be holding. That answers "what changed in
this pull request" and nothing else. The questions that arrive after an incident are
different in kind, and all of them are about time:

- Six days of downstream data stopped being invalidated. When did that edge narrow?
- Who narrowed it, and what reason did they give?
- This dataset's partition spec is wrong. How long has it been wrong?

A lineage graph without a history can answer none of them. It is a photograph of now,
and every incident review is an exercise in reconstructing a past state from commit
messages in whichever repository happened to generate the edges.

**What is stored, and what is not.** A revision keeps the `GraphDiff` from its
predecessor plus a content digest of the whole graph — not a copy of the graph.
Snapshots are what make history unaffordable and therefore switched off; diffs are
small and answer every question above. The tradeoff is real and worth stating: this
can tell you exactly when and how an edge changed, and it cannot hand you the graph
as it stood last March. `digest_at` lets you *verify* a graph you still have is the
one a revision described, which is the part that matters for an audit.

**Ordering.** Revisions form a chain, and `record` refuses one whose parent is not
the current head. A history that silently accepts a revision computed against a stale
graph would attribute somebody's change to whoever committed next.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ..core.types import DatasetId
from ..core.util.clock import as_utc, now
from .diff import EdgeChange, GraphDiff, diff_graphs
from .model import Graph

__all__ = [
    "History",
    "Revision",
    "authors_of",
    "graph_digest",
    "narrowings_of",
    "record",
    "replay",
    "revisions_touching",
    "since",
    "timeline",
    "unsafe_revisions",
]


def graph_digest(graph: Graph) -> str:
    """A stable content digest of a whole graph.

    Covers datasets, their specs, and every edge including evidence, so two graphs
    share a digest exactly when they would plan identically. Sorted before hashing,
    because edge insertion order is an artifact of ingest and not a property of the
    graph.
    """
    parts: list[str] = []
    for ds in graph.datasets:
        spec = graph.spec(ds)
        fields = ",".join(
            f"{f.name}:{f.kind}:{f.grain.label if f.grain else ''}" for f in spec.fields
        )
        parts.append(f"D|{ds}|{fields}")
    for edge in sorted(graph.edges, key=lambda e: (str(e.src), str(e.dst), e.evidence)):
        columns = ",".join(f"{s}->{t}" for s, t in sorted(edge.columns))
        parts.append(f"E|{edge.src}|{edge.dst}|{edge.mapping}|{edge.evidence}|{columns}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Revision:
    """One recorded state of the graph, and how it differs from its predecessor."""

    digest: str
    at: datetime
    author: str
    note: str
    diff: GraphDiff
    parent: str | None = None
    datasets: int = 0
    edges: int = 0

    @property
    def is_safe(self) -> bool:
        """True when this revision narrowed nothing and removed no edge."""
        return self.diff.is_safe

    @property
    def is_initial(self) -> bool:
        """True when this is the first recorded state, with nothing to diff."""
        return self.parent is None

    def __str__(self) -> str:
        when = as_utc(self.at).date().isoformat()
        what = "initial" if self.is_initial else ("safe" if self.is_safe else "UNSAFE")
        return f"{self.digest} {when} {self.author}: {self.note or '(no note)'} [{what}]"


@dataclass
class History:
    """An append-only chain of graph revisions, newest last."""

    revisions: list[Revision] = field(default_factory=list)

    @property
    def head(self) -> Revision | None:
        """The newest recorded revision, or None."""
        return self.revisions[-1] if self.revisions else None

    def __len__(self) -> int:
        return len(self.revisions)

    def __iter__(self) -> Iterator[Revision]:
        return iter(self.revisions)

    def get(self, digest: str) -> Revision | None:
        """One revision by id, or None."""
        return next((r for r in self.revisions if r.digest == digest), None)

    def digest_at(self, moment: datetime) -> str | None:
        """The digest of the graph as it stood at `moment`.

        Enough to verify that a graph you still hold is the one in force at a past
        date, which is what an audit needs. It cannot reconstruct the graph itself —
        see the module docstring for why revisions store diffs rather than snapshots.
        """
        target = as_utc(moment)
        found = [r for r in self.revisions if as_utc(r.at) <= target]
        return found[-1].digest if found else None

    def summary(self) -> str:
        """The history as text, newest first."""
        if not self.revisions:
            return "no revisions recorded"
        unsafe = [r for r in self.revisions if not r.is_safe and not r.is_initial]
        lines = [
            f"{len(self.revisions)} revision(s), head {self.head.digest if self.head else '-'}"
        ]
        if unsafe:
            lines.append(f"    {len(unsafe)} narrowed or removed an edge:")
            lines.extend(f"        {r}" for r in unsafe[-5:])
        return "\n".join(lines)


def record(
    history: History,
    graph: Graph,
    *,
    author: str = "",
    note: str = "",
    at: datetime | None = None,
    previous: Graph | None = None,
) -> Revision:
    """Append `graph` to `history` as a new revision.

    `previous` is the graph the head revision described. It is required whenever the
    history is non-empty, and its digest must match the head — a revision computed
    against a stale graph would attribute one person's change to the next person to
    commit, which is worse than having no history at all.

    Recording an unchanged graph is a no-op that returns the existing head, so a
    scheduled ingest that found nothing new does not fill the history with noise.
    """
    digest = graph_digest(graph)
    head = history.head

    if head is None:
        revision = Revision(
            digest=digest,
            at=as_utc(at or now()),
            author=author,
            note=note,
            diff=GraphDiff(),
            parent=None,
            datasets=len(graph.datasets),
            edges=len(graph.edges),
        )
        history.revisions.append(revision)
        return revision

    if digest == head.digest:
        return head

    if previous is None:
        raise ValueError(
            "recording against a non-empty history needs `previous`, the graph the "
            f"head revision {head.digest} described"
        )
    if graph_digest(previous) != head.digest:
        raise ValueError(
            f"`previous` has digest {graph_digest(previous)} but the head revision is "
            f"{head.digest}; recording against a stale graph would misattribute the change"
        )

    revision = Revision(
        digest=digest,
        at=as_utc(at or now()),
        author=author,
        note=note,
        diff=diff_graphs(previous, graph),
        parent=head.digest,
        datasets=len(graph.datasets),
        edges=len(graph.edges),
    )
    history.revisions.append(revision)
    return revision


def unsafe_revisions(history: History) -> list[Revision]:
    """Revisions that narrowed a mapping or removed an edge, oldest first.

    These are the two ways a graph edit serves stale data, so this is the list an
    incident review starts from.
    """
    return [r for r in history if not r.is_initial and not r.is_safe]


def revisions_touching(
    history: History, src: DatasetId, dst: DatasetId
) -> list[tuple[Revision, str]]:
    """Every revision that changed the edge between two datasets, with what it did."""
    out: list[tuple[Revision, str]] = []
    for revision in history:
        for edge in revision.diff.added_edges:
            if (edge.src, edge.dst) == (src, dst):
                out.append((revision, "added"))
        for edge in revision.diff.removed_edges:
            if (edge.src, edge.dst) == (src, dst):
                out.append((revision, "removed"))
        for change in revision.diff.changed_edges:
            if (change.src, change.dst) == (src, dst):
                out.append((revision, _verb(change)))
    return out


def _verb(change: EdgeChange) -> str:
    if change.narrowed:
        return "narrowed"
    if change.widened:
        return "widened"
    return "changed"


def narrowings_of(
    history: History, src: DatasetId, dst: DatasetId
) -> list[tuple[Revision, EdgeChange]]:
    """When an edge narrowed, and by whom.

    The question an incident asks: six days of downstream data stopped being
    invalidated, so when did that window shrink and who shrank it.
    """
    out: list[tuple[Revision, EdgeChange]] = []
    for revision in history:
        for change in revision.diff.narrowings:
            if (change.src, change.dst) == (src, dst):
                out.append((revision, change))
    return out


def authors_of(history: History, src: DatasetId, dst: DatasetId) -> list[str]:
    """Everyone who has changed one edge, in the order they first did."""
    seen: list[str] = []
    for revision, _ in revisions_touching(history, src, dst):
        if revision.author and revision.author not in seen:
            seen.append(revision.author)
    return seen


def since(history: History, moment: datetime) -> list[Revision]:
    """Revisions recorded after `moment`."""
    target = as_utc(moment)
    return [r for r in history if as_utc(r.at) > target]


def replay(revisions: Iterable[Revision]) -> GraphDiff:
    """Fold a run of revisions into one combined diff.

    Useful for "what changed across the whole release" without re-diffing the
    endpoints. Edge changes accumulate; a dataset added and then removed within the
    run cancels, because the net effect is what a reader of a release note wants.
    """
    combined = GraphDiff()
    for revision in revisions:
        diff = revision.diff
        for ds in diff.added_datasets:
            if ds in combined.removed_datasets:
                combined.removed_datasets.remove(ds)
            else:
                combined.added_datasets.append(ds)
        for ds in diff.removed_datasets:
            if ds in combined.added_datasets:
                combined.added_datasets.remove(ds)
            else:
                combined.removed_datasets.append(ds)
        combined.added_edges.extend(diff.added_edges)
        combined.removed_edges.extend(diff.removed_edges)
        combined.changed_edges.extend(diff.changed_edges)
        combined.changed_specs.extend(diff.changed_specs)
    return combined


def timeline(history: History, *, limit: int = 20) -> str:
    """The last `limit` revisions, newest first, one line each."""
    recent: Sequence[Revision] = list(history.revisions)[-limit:]
    return "\n".join(str(r) for r in reversed(recent)) or "no revisions recorded"
