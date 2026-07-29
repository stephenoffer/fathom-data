"""Compliance artefacts, generated from lineage rather than maintained by hand.

Every regime that touches data asks the same four questions in different words:
what personal data do you hold, where did it come from, where did it go, and who can
you prove that to. Organizations answer them with a spreadsheet that is accurate on
the day it is written.

Everything in this module is derived from the graph, so it is accurate on the day it
is read. That is the entire argument. The formats differ — an Article 30 record, a
subject access response, a training-data summary — but the source is one traversal.

Two commitments the module keeps everywhere:

- **Gaps are reported, never smoothed.** `readiness` lists what the graph cannot
  currently answer. A generated record that looks complete because the generator
  omitted what it did not know is worse than no record, since somebody will sign it.
- **Nothing claims a legal conclusion.** These are evidence bundles for a person who
  makes that call, and the wording says so.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..ai.assets import is_ai_asset, is_model, kind_of
from ..ai.training import data_bill_of_materials, training_data_summary
from ..ai.unlearning import completeness_statement, exposures
from ..core.types import DatasetId
from ..core.util import markdown as md
from ..core.util import text as _text
from ..govern.consent import ConsentScope, permitted_purposes, region_of
from ..govern.licenses import License, effective_license
from ..govern.policy import LabelSet
from ..graph.metrics import coverage
from ..graph.model import Graph
from ..graph.query import ancestors, descendants

__all__ = [
    "ComplianceReadiness",
    "ProcessingRecord",
    "ai_act_record",
    "audit_bundle",
    "cross_border_summary",
    "personal_data_inventory",
    "processing_record",
    "erasure_attestation",
    "readiness",
    "records_for",
    "subject_access_report",
]


@dataclass
class ProcessingRecord:
    """One dataset's entry in a record of processing activities.

    Modelled on GDPR Article 30, which is the most widely copied shape even where it
    is not the applicable law. Every field is populated from the graph or explicitly
    marked unknown.
    """

    dataset: DatasetId
    categories: list[str] = field(default_factory=list)
    purposes: list[str] = field(default_factory=list)
    sources: list[DatasetId] = field(default_factory=list)
    recipients: list[DatasetId] = field(default_factory=list)
    region: str = ""
    lawful_basis: str = ""
    retention: str = ""
    unknowns: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True when the record has everything a filing needs."""
        return not self.unknowns

    def summary(self) -> str:
        """The record as text, gaps stated rather than omitted."""
        lines = [
            f"## {self.dataset}",
            "",
            f"- data categories: {', '.join(self.categories) or '_none identified_'}",
            f"- purposes: {', '.join(self.purposes) or '_not recorded_'}",
            f"- lawful basis: {self.lawful_basis or '_not recorded_'}",
            f"- storage region: {self.region or '_not determinable from the identity_'}",
            f"- retention: {self.retention or '_not recorded_'}",
            f"- sources: {len(self.sources)}",
            f"- recipients: {len(self.recipients)}",
        ]
        if self.unknowns:
            lines.extend(["", "Gaps in this record:", "", md.bullets(self.unknowns)])
        return "\n".join(lines)


def processing_record(
    graph: Graph,
    ds: DatasetId,
    *,
    labels: LabelSet | None = None,
    consent: Mapping[DatasetId, ConsentScope] | None = None,
) -> ProcessingRecord:
    """Build one dataset's Article 30-shaped record from the graph."""
    label_set = labels or {}
    scopes = dict(consent or {})

    categories = sorted(
        {
            label.name
            for ref, values in label_set.items()
            if ref.dataset == ds
            for label in values
            if label.name != "pii"
        }
    )
    purposes = sorted(p.value for p in permitted_purposes(graph, ds, scopes))
    scope = scopes.get(ds)

    record = ProcessingRecord(
        dataset=ds,
        categories=categories,
        purposes=purposes,
        sources=ancestors(graph, ds),
        recipients=descendants(graph, ds),
        region=region_of(ds),
        lawful_basis=scope.basis if scope else "",
        retention=str(scope.retention) if scope and scope.retention else "",
    )

    if not categories:
        record.unknowns.append("no data categories recorded; run label inference over a profile")
    if not purposes:
        record.unknowns.append("no permitted purposes resolve from consent scopes upstream")
    if scope is None:
        record.unknowns.append("no consent scope declared for this dataset")
    if not record.sources and not graph.in_edges(ds):
        record.unknowns.append("no upstream lineage; the origin of this data is unrecorded")
    return record


def personal_data_inventory(graph: Graph, labels: LabelSet) -> dict[str, list[DatasetId]]:
    """Every dataset carrying each personal-data label.

    The answer to "what personal data do we hold", assembled rather than surveyed.
    """
    out: dict[str, set[DatasetId]] = {}
    for ref, values in labels.items():
        for label in values:
            out.setdefault(label.name, set()).add(ref.dataset)
    return {name: sorted(items, key=str) for name, items in sorted(out.items())}


def subject_access_report(
    graph: Graph,
    origin: DatasetId,
    *,
    subject_digest: str = "",
    labels: LabelSet | None = None,
) -> str:
    """Where a subject's data went, written for the subject rather than the engineer.

    A data subject access request asks for more than a dump of rows: it asks what was
    done with the data and who received it. The graph answers both, and the AI section
    answers the one most responses quietly omit.
    """
    reach = descendants(graph, origin)
    ai_assets = [ds for ds in reach if is_ai_asset(ds)]
    label_set = labels or {}
    categories = sorted(
        {
            label.name
            for ref, values in label_set.items()
            if ref.dataset == origin
            for label in values
        }
    )

    lines = [
        "# Data subject access — lineage section",
        "",
        f"Generated {datetime.now(UTC).date().isoformat()} for subject "
        f"`{subject_digest[:12] or '(undisclosed)'}…`.",
        "",
        "## Where the data originates",
        "",
        f"- `{origin}`",
        "",
        "## Data categories held",
        "",
    ]
    lines.append(md.bullets(categories, empty="_none identified_"))
    lines.extend(
        [
            "",
            f"## Systems the data reached ({len(reach)})",
            "",
            md.bullets(
                _text.truncate([f"{md.code(ds)} ({kind_of(ds).value})" for ds in reach], 100)
            ),
        ]
    )

    if ai_assets:
        lines.extend(["", "## Automated processing", ""])
        lines.append(
            "The data reached the following models and AI systems. Where a model was "
            "trained on it, information derived from the data is retained in that "
            "model's parameters and is not removed by deleting the source records."
        )
        lines.append("")
        lines.append(
            md.bullets(f"{md.code(e.asset)} — {e.route.value}" for e in exposures(graph, origin))
        )
    return "\n".join(lines)


def ai_act_record(
    graph: Graph,
    model: DatasetId,
    *,
    intended_use: str = "",
    licenses: Mapping[DatasetId, License] | None = None,
    consent: Mapping[DatasetId, ConsentScope] | None = None,
) -> str:
    """A provider-side record for a model: what it was trained on, and under what terms.

    Assembles the training-data summary, the effective licence of the closure, and the
    purposes upstream consent actually permits. Each of the three is generated, and
    each carries its own gaps.
    """
    bom = data_bill_of_materials(graph, model)
    lines = [training_data_summary(graph, model), "", "## Intended purpose", ""]
    lines.append(intended_use or "_Not stated._")

    if licenses:
        effective = effective_license(graph, model, licenses)
        lines.extend(
            [
                "",
                "## Licensing of training data",
                "",
                f"Effective combined licence: `{effective}`.",
                "",
                f"- commercial use: {_tri(effective.commercial)}",
                f"- derivative works: {_tri(effective.derivatives)}",
                f"- text and data mining: {_tri(effective.text_and_data_mining)}",
                f"- attribution required: {'yes' if effective.attribution else 'no'}",
            ]
        )
        if effective.is_unknown:
            lines.append("")
            lines.append(
                "At least one upstream source has no recorded licence. This record cannot "
                "assert that training was permitted."
            )

    if consent:
        purposes = sorted(p.value for p in permitted_purposes(graph, model, dict(consent)))
        lines.extend(["", "## Purposes permitted by upstream consent", ""])
        lines.append(", ".join(purposes) if purposes else "_None resolve; consent is unrecorded._")

    if bom.gaps:
        lines.extend(["", "## Provenance gaps", ""])
        lines.extend(f"- {gap}" for gap in bom.gaps)
    return "\n".join(lines)


def _tri(value: bool | None) -> str:
    return "permitted" if value is True else "forbidden" if value is False else "undetermined"


def cross_border_summary(graph: Graph) -> dict[str, list[str]]:
    """Data flows that cross a storage region boundary.

    Read off the identities, so it reflects where the bytes are rather than where a
    diagram says they are.
    """
    out: dict[str, list[str]] = {}
    for edge in graph.edges:
        source_region, target_region = region_of(edge.src), region_of(edge.dst)
        if not source_region or not target_region or source_region == target_region:
            continue
        out.setdefault(f"{source_region} -> {target_region}", []).append(
            f"{edge.src} -> {edge.dst}"
        )
    return {k: sorted(v) for k, v in sorted(out.items())}


@dataclass
class ComplianceReadiness:
    """Whether the graph can support a compliance claim, and what is missing."""

    score: float = 0.0
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """True when this can be filed as it stands."""
        return not self.blockers

    def summary(self) -> str:
        """The record as text, gaps stated rather than omitted."""
        verdict = "READY" if self.is_ready else "NOT READY"
        lines = [f"compliance readiness: {verdict} ({self.score:.0%})"]
        lines.extend(f"  blocker: {note}" for note in self.blockers)
        lines.extend(f"  warning: {note}" for note in self.warnings)
        return "\n".join(lines)


def readiness(
    graph: Graph,
    *,
    labels: LabelSet | None = None,
    consent: Mapping[DatasetId, ConsentScope] | None = None,
    licenses: Mapping[DatasetId, License] | None = None,
) -> ComplianceReadiness:
    """Can this graph support the claims a compliance report makes?

    Deliberately hard to pass. A report generated from a graph with 20% column
    lineage will name datasets where it should name columns, and that difference is
    the one an auditor asks about.
    """
    out = ComplianceReadiness()
    stats = coverage(graph)
    label_set = labels or {}

    if not graph.datasets:
        out.blockers.append("the graph is empty")
        return out

    if not label_set:
        out.blockers.append("no labels recorded; personal data cannot be located without them")
    if stats.column_ratio < 0.3:
        out.warnings.append(
            f"only {stats.column_ratio:.0%} of edges carry column lineage; reports will "
            "name datasets where they should name columns"
        )
    if not consent:
        out.blockers.append("no consent scopes declared; purpose limitation is unverifiable")
    if not licenses:
        out.warnings.append("no licences declared; training permissions cannot be asserted")

    models = [ds for ds in graph.datasets if is_model(ds)]
    untracked = [ds for ds in models if not graph.in_edges(ds)]
    if untracked:
        out.blockers.append(
            f"{len(untracked)} model(s) have no recorded training inputs; their "
            "provenance cannot be stated"
        )

    checks = [
        bool(label_set),
        bool(consent),
        bool(licenses),
        stats.column_ratio >= 0.3,
        not untracked,
        stats.spec_ratio >= 0.5,
    ]
    out.score = round(sum(1 for check in checks if check) / len(checks), 4)
    return out


def audit_bundle(
    graph: Graph,
    *,
    labels: LabelSet | None = None,
    consent: Mapping[DatasetId, ConsentScope] | None = None,
    licenses: Mapping[DatasetId, License] | None = None,
    models: Iterable[DatasetId] = (),
) -> dict[str, Any]:
    """Every compliance artefact this module can generate, in one structure.

    Written to be serialized whole and handed over, so an audit becomes a file
    transfer rather than a fortnight of screenshots.
    """
    label_set = labels or {}
    targets = list(models) or [ds for ds in graph.datasets if is_model(ds)]
    check = readiness(graph, labels=label_set, consent=consent, licenses=licenses)

    return {
        "generated": datetime.now(UTC).isoformat(),
        "readiness": {
            "score": check.score,
            "ready": check.is_ready,
            "blockers": check.blockers,
            "warnings": check.warnings,
        },
        "coverage": {
            "datasets": len(graph.datasets),
            "edges": len(graph.edges),
            "spec_ratio": coverage(graph).spec_ratio,
            "column_ratio": coverage(graph).column_ratio,
        },
        "personal_data": {
            name: [str(ds) for ds in items]
            for name, items in personal_data_inventory(graph, label_set).items()
        },
        "cross_border": cross_border_summary(graph),
        "models": {
            str(model): {
                "training_inputs": [str(ds) for ds in data_bill_of_materials(graph, model).direct],
                "gaps": data_bill_of_materials(graph, model).gaps,
            }
            for model in targets
        },
    }


def erasure_attestation(graph: Graph, origin: DatasetId, *, subject_digest: str = "") -> str:
    """The statement accompanying a completed erasure request.

    Delegates the hard sentence to `fathom.ai.unlearning`, which is where the model
    exposure lives, rather than restating it and risking the two drifting apart.
    """
    return completeness_statement(graph, origin, subject_digest=subject_digest)


def records_for(
    graph: Graph,
    datasets: Sequence[DatasetId],
    *,
    labels: LabelSet | None = None,
    consent: Mapping[DatasetId, ConsentScope] | None = None,
) -> str:
    """A full record of processing activities across several datasets, as Markdown."""
    lines = [
        "# Record of processing activities",
        "",
        f"Generated from lineage on {datetime.now(UTC).date().isoformat()}. "
        "This is evidence for a determination, not a determination.",
        "",
    ]
    for ds in datasets:
        record = processing_record(graph, ds, labels=labels, consent=consent)
        lines.extend([record.summary(), ""])
    return "\n".join(lines)
