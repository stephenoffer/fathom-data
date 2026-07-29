"""Turning a drift alert into a diagnosis.

"`revenue` moved 8%" is an alert. Someone still has to find out why, and that search
is the expensive part — it is a person opening tables in descending order of hunch.

The graph shortens it. A column that moved has a bounded set of upstream columns
that could have moved it, they are ordered by distance, and the ones that also
show drift in their own profiles are the candidates. Everything else is not.

Scoring is deliberately simple and deliberately explainable:

- an upstream column with drift of its own scores highest
- nearer causes outrank distant ones, because a change three hops up usually shows
  at two hops as well and the nearer one is where to look first
- a dataset whose profile is missing is reported as *unchecked* rather than clean,
  because "we did not look" and "we looked and it was fine" are different answers
  and conflating them sends people down the wrong path

Nothing here claims causation. It claims a ranked list of things that could have
caused it, which is what a person needs at 3am and what a naive correlation search
does not give.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..core.types import ColumnRef, DatasetId
from ..core.util import markdown as md
from ..graph.model import Graph
from ..graph.query import ancestors, column_ancestors, distance
from ..observe.profile import Finding, Profile, Severity, drift

__all__ = [
    "Cause",
    "Diagnosis",
    "attribute",
    "attribute_column",
    "blame_report",
    "rank",
    "root_causes",
    "suspects",
    "unchecked",
]

# How much of the score each signal is worth. Drift in the upstream column itself
# dominates; proximity breaks ties among several drifting candidates.
_DRIFT_WEIGHT = 0.6
_PROXIMITY_WEIGHT = 0.3
_COLUMN_WEIGHT = 0.1


@dataclass
class Cause:
    """One upstream candidate for an observed change, with its evidence."""

    dataset: DatasetId
    column: str | None = None
    hops: int = 0
    findings: list[Finding] = field(default_factory=list)
    checked: bool = False
    score: float = 0.0

    @property
    def has_drift(self) -> bool:
        """True when any upstream column moved."""
        return bool(self.findings)

    @property
    def worst_severity(self) -> Severity | None:
        """The most severe finding across every attributed cause."""
        if not self.findings:
            return None
        order = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
        return min((f.severity for f in self.findings), key=lambda s: order[s])

    def explain(self) -> str:
        """One sentence a human can act on."""
        where = f"`{self.column}` in {self.dataset}" if self.column else str(self.dataset)
        if not self.checked:
            return f"{where} is {self.hops} hop(s) upstream and was not profiled — unchecked"
        if not self.findings:
            return f"{where} is {self.hops} hop(s) upstream and shows no drift"
        detail = "; ".join(f.detail for f in self.findings[:2])
        return f"{where} is {self.hops} hop(s) upstream and drifted: {detail}"

    def __str__(self) -> str:
        return f"[{self.score:.2f}] {self.explain()}"


@dataclass
class Diagnosis:
    """A ranked account of what could have caused an observed change."""

    target: DatasetId
    target_column: str | None = None
    causes: list[Cause] = field(default_factory=list)

    @property
    def best(self) -> Cause | None:
        """The single most likely cause, or None when nothing was attributed."""
        return self.causes[0] if self.causes else None

    @property
    def confirmed(self) -> list[Cause]:
        """Candidates with drift of their own."""
        return [c for c in self.causes if c.has_drift]

    @property
    def unchecked(self) -> list[Cause]:
        """Candidates nobody profiled. The gap in the diagnosis, stated."""
        return [c for c in self.causes if not c.checked]

    def summary(self) -> str:
        """The attribution as text, most likely cause first."""
        where = f"{self.target}#{self.target_column}" if self.target_column else str(self.target)
        if not self.causes:
            return f"{where}: no upstream candidates in the graph"
        lines = [f"{where}: {len(self.confirmed)} upstream candidate(s) with drift"]
        for cause in self.causes[:8]:
            lines.append(f"  {cause}")
        if self.unchecked:
            lines.append(
                f"  ({len(self.unchecked)} upstream dataset(s) were not profiled; "
                "this diagnosis is incomplete)"
            )
        return "\n".join(lines)


def suspects(graph: Graph, target: DatasetId, *, max_depth: int = 6) -> list[DatasetId]:
    """Upstream datasets that could account for a change in `target`."""
    return ancestors(graph, target, max_depth=max_depth)


def unchecked(
    graph: Graph, target: DatasetId, profiles: Mapping[DatasetId, Profile], *, max_depth: int = 6
) -> list[DatasetId]:
    """Upstream datasets with no profile — the blind spots in any attribution."""
    return [ds for ds in suspects(graph, target, max_depth=max_depth) if ds not in profiles]


def _score(hops: int, findings: Sequence[Finding], *, column_match: bool, max_depth: int) -> float:
    proximity = max(0.0, 1.0 - (hops - 1) / max(1, max_depth))
    severity_weight = 0.0
    if findings:
        worst = min(
            (f.severity for f in findings),
            key=lambda s: {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}[s],
        )
        severity_weight = {Severity.ERROR: 1.0, Severity.WARN: 0.7, Severity.INFO: 0.3}[worst]
    return round(
        _DRIFT_WEIGHT * severity_weight
        + _PROXIMITY_WEIGHT * proximity
        + _COLUMN_WEIGHT * (1.0 if column_match else 0.0),
        4,
    )


def attribute(
    graph: Graph,
    target: DatasetId,
    *,
    before: Mapping[DatasetId, Profile],
    after: Mapping[DatasetId, Profile],
    max_depth: int = 6,
) -> Diagnosis:
    """Rank upstream datasets by how likely each is to explain a change in `target`.

    `before` and `after` hold profiles keyed by dataset — typically the last stored
    profile and a fresh one. Datasets absent from either are reported as unchecked
    rather than silently dropped.
    """
    diagnosis = Diagnosis(target=target)

    for source in suspects(graph, target, max_depth=max_depth):
        hops = distance(graph, source, target) or 1
        b, a = before.get(source), after.get(source)
        if b is None or a is None:
            diagnosis.causes.append(
                Cause(
                    dataset=source,
                    hops=hops,
                    checked=False,
                    score=_score(hops, (), column_match=False, max_depth=max_depth),
                )
            )
            continue
        findings = drift(b, a)
        diagnosis.causes.append(
            Cause(
                dataset=source,
                hops=hops,
                findings=findings,
                checked=True,
                score=_score(hops, findings, column_match=False, max_depth=max_depth),
            )
        )
    diagnosis.causes = rank(diagnosis.causes)
    return diagnosis


def attribute_column(
    graph: Graph,
    target: ColumnRef,
    *,
    before: Mapping[DatasetId, Profile],
    after: Mapping[DatasetId, Profile],
    max_depth: int = 6,
) -> Diagnosis:
    """Column-level attribution: rank the upstream *columns* that feed `target`.

    Sharper than the dataset-level form and only available where column lineage was
    extracted. Where it is, this is the difference between "something in the orders
    pipeline changed" and "`fx_rate` in `raw.rates` changed".
    """
    diagnosis = Diagnosis(target=target.dataset, target_column=target.column)
    upstream = column_ancestors(graph, target, max_depth=max_depth)

    for ref in upstream:
        hops = distance(graph, ref.dataset, target.dataset) or 1
        b, a = before.get(ref.dataset), after.get(ref.dataset)
        if b is None or a is None:
            diagnosis.causes.append(
                Cause(
                    dataset=ref.dataset,
                    column=ref.column,
                    hops=hops,
                    checked=False,
                    score=_score(hops, (), column_match=True, max_depth=max_depth),
                )
            )
            continue
        findings = [f for f in drift(b, a) if f.column == ref.column or f.column is None]
        diagnosis.causes.append(
            Cause(
                dataset=ref.dataset,
                column=ref.column,
                hops=hops,
                findings=findings,
                checked=True,
                score=_score(hops, findings, column_match=True, max_depth=max_depth),
            )
        )
    diagnosis.causes = rank(diagnosis.causes)
    return diagnosis


def rank(causes: Sequence[Cause]) -> list[Cause]:
    """Order candidates: highest score first, nearest first on a tie."""
    return sorted(causes, key=lambda c: (-c.score, c.hops, str(c.dataset), c.column or ""))


def blame_report(diagnosis: Diagnosis, *, limit: int = 10) -> str:
    """A Markdown write-up of a diagnosis, suitable for pasting into an incident."""
    where = (
        f"`{diagnosis.target}#{diagnosis.target_column}`"
        if diagnosis.target_column
        else f"`{diagnosis.target}`"
    )
    lines = [f"### Attribution for {where}", ""]
    if not diagnosis.causes:
        lines.append("No upstream lineage recorded, so nothing can be attributed.")
        return "\n".join(lines)

    confirmed = diagnosis.confirmed
    if confirmed:
        lines.append(f"**Most likely cause:** {confirmed[0].explain()}")
    else:
        lines.append("**No upstream drift found.** The change likely originated at this dataset.")
    rows = [
        [
            f"{cause.score:.2f}",
            md.code(cause.dataset) + (f"#{md.code(cause.column)}" if cause.column else ""),
            cause.hops,
            "unchecked"
            if not cause.checked
            else (f"{len(cause.findings)} finding(s)" if cause.findings else "clean"),
        ]
        for cause in diagnosis.causes
    ]
    lines.extend(["", md.table(["Score", "Upstream", "Hops", "Status"], rows, limit=limit)])
    if diagnosis.unchecked:
        lines.extend(
            [
                "",
                f"> {len(diagnosis.unchecked)} upstream dataset(s) have no profile. "
                "This attribution is incomplete until they are profiled.",
            ]
        )
    return "\n".join(lines)


def root_causes(diagnosis: Diagnosis) -> list[Cause]:
    """The deepest drifting candidates — the origin rather than the relay.

    A change usually shows at every hop between its origin and the alert. The one
    worth fixing is the furthest upstream that still shows it.
    """
    drifting = [c for c in diagnosis.causes if c.has_drift]
    if not drifting:
        return []
    deepest = max(c.hops for c in drifting)
    return [c for c in drifting if c.hops == deepest]
