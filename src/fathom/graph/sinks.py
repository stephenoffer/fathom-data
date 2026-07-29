"""The last hop, where a number stops being data and becomes a published claim.

Lineage conventionally stops at the edge of the warehouse. `descendants` returns
tables, and every one of them is a thing an engineer can rebuild. That is the wrong
boundary for the most expensive question anybody asks of a lineage graph:

    We restated this metric. What have we already told people?

A dashboard, a board pack, a regulatory filing, a customer-facing export, and a
served model endpoint are all downstream of that metric, and none of them is a table.
They are where a wrong number turns into a wrong decision, an amended filing, or a
disclosure — and they are precisely the hops the graph cannot see, so the impact
assessment gets done from memory by whoever has been there longest.

**Sinks are datasets.** Same argument the `ai` package makes about models: give a
dashboard a `DatasetId` and the existing traversal, policy propagation, and erasure
machinery already reach it. There is no second graph for published artefacts, because
a second graph is a graph that disagrees with the first one.

**Sinks are terminal.** Nothing is downstream of a filing. `record_publication`
refuses to add an edge *out* of a sink, because a sink that feeds a table is either a
modelling error or a genuinely circular reporting process, and both deserve to fail
loudly rather than quietly extend a restatement cone forever.

**Filings are separated from everything else.** `restatement_impact` reports them on
their own line, and `has_regulatory_exposure` exists as a distinct question. A wrong
dashboard is embarrassing and a wrong filing is a legal event; collapsing the two into
one "affected artefacts" count is how the second gets missed inside the first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ..core.partitions import PartitionMapping
from ..core.types import UNPARTITIONED, DatasetId
from .model import Edge, Graph, link
from .query import ancestors, descendants, paths_between

__all__ = [
    "REGULATORY",
    "RestatementImpact",
    "SinkKind",
    "by_kind",
    "coverage",
    "dashboard",
    "declare_many",
    "describe",
    "endpoint",
    "export",
    "filing",
    "has_regulatory_exposure",
    "is_sink",
    "kind_of",
    "notebook",
    "notice_text",
    "of_kind",
    "publication_paths",
    "published_datasets",
    "record_publication",
    "report",
    "restatement_impact",
    "riskiest",
    "sinks_in",
    "sinks_of",
    "sources_of",
    "unpublished",
]


class SinkKind(StrEnum):
    """What a published artefact is, ordered roughly by cost of being wrong."""

    NOTEBOOK = "notebook"  # someone's analysis; wrong is cheap
    DASHBOARD = "dashboard"  # a decision surface
    EXPORT = "export"  # data handed to a third party
    ENDPOINT = "endpoint"  # a served model or API, wrong in real time
    REPORT = "report"  # a periodic pack somebody signs
    FILING = "filing"  # submitted to a regulator; wrong is a legal event


_SCHEMES: dict[SinkKind, str] = {
    SinkKind.NOTEBOOK: "notebook",
    SinkKind.DASHBOARD: "dashboard",
    SinkKind.EXPORT: "export",
    SinkKind.ENDPOINT: "endpoint",
    SinkKind.REPORT: "report",
    SinkKind.FILING: "filing",
}

_BY_SCHEME = {scheme: kind for kind, scheme in _SCHEMES.items()}

_NAMESPACE = re.compile(r"^(?P<scheme>[a-z]+)://(?P<instance>.*)$")

# Kinds whose being wrong is a reportable event rather than an inconvenience.
REGULATORY = frozenset({SinkKind.FILING, SinkKind.REPORT})


def _identity(kind: SinkKind, name: str, instance: str) -> DatasetId:
    cleaned = name.strip().strip("/")
    if not cleaned:
        raise ValueError(f"a {kind.value} needs a name")
    return DatasetId(namespace=f"{_SCHEMES[kind]}://{instance.strip().lower()}", name=cleaned)


# -- constructors --------------------------------------------------------------


def dashboard(name: str, *, tool: str = "local") -> DatasetId:
    """A dashboard or tile. ``dashboard("revenue/exec", tool="looker")``."""
    return _identity(SinkKind.DASHBOARD, name, tool)


def report(name: str, *, publisher: str = "local") -> DatasetId:
    """A periodic pack somebody signs — a board deck, a monthly close."""
    return _identity(SinkKind.REPORT, name, publisher)


def filing(name: str, *, regulator: str = "local") -> DatasetId:
    """A submission to a regulator. Being wrong here is an amendment, not a refresh."""
    return _identity(SinkKind.FILING, name, regulator)


def export(name: str, *, recipient: str = "local") -> DatasetId:
    """Data handed to a third party, which cannot be recalled once sent."""
    return _identity(SinkKind.EXPORT, name, recipient)


def endpoint(name: str, *, service: str = "local") -> DatasetId:
    """A served model or API — wrong in real time, for everybody, until fixed."""
    return _identity(SinkKind.ENDPOINT, name, service)


def notebook(name: str, *, workspace: str = "local") -> DatasetId:
    """Someone's analysis. Included because it is where a number escapes unreviewed."""
    return _identity(SinkKind.NOTEBOOK, name, workspace)


# -- identity ------------------------------------------------------------------


def of_kind(kind: SinkKind | str, name: str, instance: str = "local") -> DatasetId:
    """Build a sink identity from a kind named as a string.

    The seam config parsing needs: `fathom.yml` says `kind: filing`, and `cli.config`
    must not import this module to validate that. It keeps its own literal set of
    valid kinds, and an unknown one raises here rather than producing an identity in
    a namespace nothing recognizes.
    """
    resolved = kind if isinstance(kind, SinkKind) else SinkKind(str(kind).lower())
    return _identity(resolved, name, instance)


def kind_of(ds: DatasetId) -> SinkKind | None:
    """Which kind of sink an identity denotes, or `None` for anything else."""
    match = _NAMESPACE.match(ds.namespace)
    return _BY_SCHEME.get(match.group("scheme")) if match else None


def is_sink(ds: DatasetId) -> bool:
    """True when this dataset is something people or systems consume."""
    return kind_of(ds) is not None


def describe(ds: DatasetId) -> str:
    """A one-line description of a sink, for a report a person reads."""
    kind = kind_of(ds)
    if kind is None:
        return str(ds)
    match = _NAMESPACE.match(ds.namespace)
    where = match.group("instance") if match else ""
    return f"{kind.value} {ds.name}" + (f" on {where}" if where else "")


# -- recording -----------------------------------------------------------------


def record_publication(
    graph: Graph,
    sink: DatasetId,
    inputs: Iterable[DatasetId],
    *,
    evidence: str = "declared",
) -> list[Edge]:
    """Record that `sink` publishes numbers derived from `inputs`.

    The mapping is `unknown` and deliberately so: a dashboard tile does not have
    partitions, and claiming a partition relationship into something with no partition
    spec would be a precision the graph has not earned. The honest answer is that any
    dirty input taints the whole artefact, which is exactly what `unknown` means.

    Raises when `sink` is not a sink identity, and when it already has outgoing edges.
    Nothing is downstream of a filing; a sink that feeds a table would extend every
    restatement cone through it forever.
    """
    if not is_sink(sink):
        raise ValueError(
            f"{sink} is not a sink identity; build one with dashboard(), report(), "
            "filing(), export(), endpoint(), or notebook()"
        )
    if graph.out_edges(sink):
        raise ValueError(f"{sink} is a sink and already has outgoing edges; sinks are terminal")

    made: list[Edge] = []
    for source in inputs:
        if is_sink(source):
            raise ValueError(f"{source} is a sink and cannot be an input to {sink}")
        made.append(
            link(
                graph,
                source,
                sink,
                evidence=evidence,
                mapping=PartitionMapping.unknown(UNPARTITIONED),
                dst_spec=UNPARTITIONED,
            )
        )
    return made


# -- the restatement question --------------------------------------------------


def sinks_of(graph: Graph, ds: DatasetId) -> list[DatasetId]:
    """Every published artefact downstream of a dataset, including through other tables."""
    return sorted((d for d in descendants(graph, ds) if is_sink(d)), key=str)


def publication_paths(graph: Graph, ds: DatasetId, sink: DatasetId) -> list[list[DatasetId]]:
    """How a dataset reaches one published artefact — what to explain in the notice."""
    return paths_between(graph, ds, sink)


@dataclass
class RestatementImpact:
    """What has already been told to whom, if this dataset was wrong."""

    dataset: DatasetId
    by_kind: dict[SinkKind, list[DatasetId]] = field(default_factory=dict)
    tables: list[DatasetId] = field(default_factory=list)

    @property
    def sinks(self) -> list[DatasetId]:
        """Every consuming dataset in the graph."""
        return sorted((s for group in self.by_kind.values() for s in group), key=str)

    @property
    def regulatory(self) -> list[DatasetId]:
        """Filings and signed reports — the ones whose remedy is an amendment."""
        return sorted((s for kind in REGULATORY for s in self.by_kind.get(kind, [])), key=str)

    @property
    def is_published(self) -> bool:
        """True when this sink leaves the organisation."""
        return bool(self.by_kind)

    def summary(self) -> str:
        """The impact as text, regulatory sinks called out first."""
        if not self.is_published:
            return (
                f"{self.dataset}: nothing published downstream "
                f"({len(self.tables)} table(s) affected)"
            )
        lines = [
            f"{self.dataset}: {len(self.sinks)} published artefact(s) affected, "
            f"across {len(self.tables)} table(s)"
        ]
        # Most consequential first, which is the order the enum is declared in reverse.
        for kind in sorted(self.by_kind, key=lambda k: -list(SinkKind).index(k)):
            named = ", ".join(d.name for d in sorted(self.by_kind[kind], key=str))
            lines.append(f"    {kind.value}: {named}")
        if self.regulatory:
            lines.append(
                f"    {len(self.regulatory)} of these are filings or signed reports — "
                "the remedy there is an amendment, not a refresh"
            )
        return "\n".join(lines)


def restatement_impact(graph: Graph, ds: DatasetId) -> RestatementImpact:
    """Everything already published that would need restating if `ds` was wrong.

    The question conventional lineage cannot answer, because it stops at the tables.
    """
    impact = RestatementImpact(dataset=ds)
    for found in descendants(graph, ds):
        kind = kind_of(found)
        if kind is None:
            impact.tables.append(found)
        else:
            impact.by_kind.setdefault(kind, []).append(found)
    impact.tables.sort(key=str)
    return impact


def has_regulatory_exposure(graph: Graph, ds: DatasetId) -> bool:
    """True when a filing or signed report is downstream.

    A separate question from "is anything published", because a wrong dashboard is
    embarrassing and a wrong filing is a legal event, and one count hides the other.
    """
    return bool(restatement_impact(graph, ds).regulatory)


def unpublished(graph: Graph) -> list[DatasetId]:
    """Datasets with no published artefact anywhere downstream.

    Not a defect — most tables are intermediate. It is the complement of the set worth
    protecting hardest, and pairs with `observe.usage` to tell an intermediate table
    from a genuinely dead one.
    """
    return sorted(
        (
            ds
            for ds in graph.datasets
            if not is_sink(ds) and not any(is_sink(d) for d in descendants(graph, ds))
        ),
        key=str,
    )


def published_datasets(graph: Graph) -> list[DatasetId]:
    """Datasets that reach at least one published artefact."""
    return sorted((ds for ds in graph.datasets if not is_sink(ds) and sinks_of(graph, ds)), key=str)


def sources_of(graph: Graph, sink: DatasetId) -> list[DatasetId]:
    """Every dataset a published artefact draws on, transitively.

    The inverse question, and the one an audit asks: this filing said X, so where did
    X come from.
    """
    return sorted((d for d in ancestors(graph, sink) if not is_sink(d)), key=str)


def sinks_in(graph: Graph) -> list[DatasetId]:
    """Every published artefact in the graph."""
    return sorted((ds for ds in graph.datasets if is_sink(ds)), key=str)


def by_kind(graph: Graph) -> dict[SinkKind, list[DatasetId]]:
    """Published artefacts grouped by what they are."""
    out: dict[SinkKind, list[DatasetId]] = {}
    for ds in sinks_in(graph):
        kind = kind_of(ds)
        assert kind is not None
        out.setdefault(kind, []).append(ds)
    return out


def riskiest(graph: Graph, *, limit: int = 10) -> list[tuple[DatasetId, int]]:
    """Datasets feeding the most published artefacts, worst first.

    Where review effort and quality expectations are worth concentrating, because
    these are the tables whose being wrong reaches furthest outside the building.
    """
    ranked: list[tuple[DatasetId, int]] = []
    for ds in graph.datasets:
        if is_sink(ds):
            continue
        found = sinks_of(graph, ds)
        if found:
            ranked.append((ds, len(found)))
    ranked.sort(key=lambda pair: (-pair[1], str(pair[0])))
    return ranked[:limit]


def notice_text(graph: Graph, ds: DatasetId, *, reason: str = "") -> str:
    """Draft the internal notice a restatement needs, from the graph.

    Names every artefact and the route to it, so the person writing the real notice
    starts from what is provably affected rather than from memory. It is a draft and
    says so — the graph knows what is downstream, not what was material.
    """
    impact = restatement_impact(graph, ds)
    lines = [f"Restatement notice (draft) — {ds}"]
    if reason:
        lines.append(f"Reason: {reason}")
    if not impact.is_published:
        lines.append(
            f"No published artefact is downstream. {len(impact.tables)} table(s) need "
            "rebuilding; nothing has been told to anyone."
        )
        return "\n".join(lines)

    lines.append("")
    lines.append("Affected published artefacts:")
    for sink in impact.sinks:
        routes = publication_paths(graph, ds, sink)
        via = " -> ".join(str(n) for n in routes[0]) if routes else "direct"
        lines.append(f"  - {describe(sink)}")
        lines.append(f"    via {via}")
    if impact.regulatory:
        lines.append("")
        lines.append(
            "Of these, the following are filings or signed reports and may require a "
            "formal amendment rather than a refresh:"
        )
        lines.extend(f"  - {describe(s)}" for s in impact.regulatory)
    lines.append("")
    lines.append(
        "Draft, generated from lineage. It states what is downstream, which is not the "
        "same as what is material — that judgement is not the graph's to make."
    )
    return "\n".join(lines)


def coverage(graph: Graph) -> float:
    """Fraction of non-sink datasets that reach a published artefact.

    Low is ambiguous in the same way `usage.read_ratio` is: either most of the
    warehouse is intermediate, or publications have not been recorded. Both are worth
    knowing and neither is distinguishable from here.
    """
    tables = [ds for ds in graph.datasets if not is_sink(ds)]
    if not tables:
        return 0.0
    return len(published_datasets(graph)) / len(tables)


def declare_many(
    graph: Graph, publications: Sequence[tuple[DatasetId, Sequence[DatasetId]]]
) -> list[Edge]:
    """Record several publications at once, from config or a BI tool's export."""
    made: list[Edge] = []
    for sink, inputs in publications:
        made.extend(record_publication(graph, sink, inputs))
    return made
