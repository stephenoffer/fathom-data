"""Erasure that reaches the model, and honesty about where it stops.

Deleting a person's rows from every table is the part of a right-to-be-forgotten
request that tooling handles well. It is also not the whole request. If those rows
fed a training run, the model retains them — diffusely, unextractably in the normal
case, and demonstrably in the bad case — and no amount of deletion downstream of the
weights changes that.

Most data tooling handles this by not mentioning it. That is the failure mode this
module exists to prevent: an erasure report reading `complete: true` while a model
trained on the subject continues to serve traffic.

So the contract here is deliberately uncomfortable:

- `exposures` names every model the subject's data reached, and by which route.
- `obligations` states, per model, what would actually discharge the request:
  retraining without the subject, shredding a key the data was encrypted under, or
  an approximate unlearning method whose guarantees are weaker than either.
- `completeness_statement` writes the paragraph a DPO has to sign, including the
  sentence about weights, because omitting it is what makes the rest a lie.

Nothing here claims to remove a subject from a model. It claims to tell you, exactly,
what you have not yet done.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..core.types import DatasetId, ErasureMode, KeyPredicate
from ..core.util import markdown as md
from ..govern.erasure import ErasurePlan, ErasureTarget
from ..graph.model import Graph
from ..graph.query import descendants, shortest_path
from .assets import AssetKind, is_model, kind_of

__all__ = [
    "Exposure",
    "ExposureRoute",
    "Obligation",
    "Remediation",
    "completeness_statement",
    "estimate_retraining_cost",
    "exposure_summary",
    "exposures",
    "models_exposed_to",
    "extend_plan",
    "is_deletion_sufficient",
    "obligations",
    "retraining_required",
    "unreachable_copies",
]


class ExposureRoute(StrEnum):
    """How a subject's data reached a model. Ordered by how hard it is to undo."""

    TRAINING = "training"  # in the weights; deletion downstream does nothing
    FINE_TUNE = "fine_tune"  # in an adapter's weights, which is smaller but the same problem
    EVALUATION = "evaluation"  # graded against, not learned from; usually only a record to purge
    CONTEXT = "context"  # passed at inference; gone unless logs retained it
    RETRIEVAL = "retrieval"  # reachable through an index; deletable, and often forgotten
    UNKNOWN = "unknown"


class Remediation(StrEnum):
    """What would actually discharge the obligation for one model."""

    RETRAIN = "retrain"  # rebuild without the subject. Complete, and expensive.
    CRYPTO_SHRED = "crypto_shred"  # destroy the key. Complete only if it was encrypted per subject.
    APPROXIMATE_UNLEARN = "approximate_unlearn"  # influence-based removal; a claim, not a proof
    DELETE_VECTORS = "delete_vectors"  # for indexes: genuinely complete
    PURGE_LOGS = "purge_logs"  # for context exposure: complete if logs are the only copy
    NONE = "none"  # nothing available. Say so.


# Which remediations are genuinely complete, versus which merely reduce exposure.
_COMPLETE = {Remediation.RETRAIN, Remediation.CRYPTO_SHRED, Remediation.DELETE_VECTORS}

_DEFAULT_REMEDIATION: dict[ExposureRoute, Remediation] = {
    ExposureRoute.TRAINING: Remediation.RETRAIN,
    ExposureRoute.FINE_TUNE: Remediation.RETRAIN,
    ExposureRoute.EVALUATION: Remediation.PURGE_LOGS,
    ExposureRoute.CONTEXT: Remediation.PURGE_LOGS,
    ExposureRoute.RETRIEVAL: Remediation.DELETE_VECTORS,
    ExposureRoute.UNKNOWN: Remediation.NONE,
}


def _route_from_evidence(evidence: str) -> ExposureRoute:
    head = evidence.split(":", 1)[0]
    return {
        "training": ExposureRoute.TRAINING,
        "eval": ExposureRoute.EVALUATION,
        "context": ExposureRoute.CONTEXT,
        "embedding": ExposureRoute.RETRIEVAL,
    }.get(head, ExposureRoute.UNKNOWN)


@dataclass(frozen=True)
class Exposure:
    """One AI asset the subject's data reached, and how."""

    asset: DatasetId
    route: ExposureRoute
    path: tuple[DatasetId, ...] = ()
    hops: int = 0

    @property
    def is_in_weights(self) -> bool:
        """True when the exposure is baked into parameters rather than sitting in storage."""
        return self.route in {ExposureRoute.TRAINING, ExposureRoute.FINE_TUNE}

    def __str__(self) -> str:
        route = " -> ".join(str(node) for node in self.path) or str(self.asset)
        return f"{self.asset} [{self.route.value}] via {route}"


def exposures(graph: Graph, origin: DatasetId) -> list[Exposure]:
    """Every AI asset a subject's data reached from `origin`.

    Walks the same edges invalidation does, so anything the planner would rebuild is
    something this finds. The route is read off the evidence string the recording
    module wrote, which is why those prefixes are stable.
    """
    found: list[Exposure] = []
    for asset in descendants(graph, origin):
        if kind_of(asset) is AssetKind.TABLE:
            continue
        path = shortest_path(graph, origin, asset) or [origin, asset]
        route = ExposureRoute.UNKNOWN
        for edge in graph.in_edges(asset):
            candidate = _route_from_evidence(edge.evidence)
            if candidate is not ExposureRoute.UNKNOWN:
                route = candidate
                break
        if route is ExposureRoute.UNKNOWN:
            # No recording module wrote this edge, so fall back on what the asset is.
            # An eval set reached by an ETL job is still an eval-set exposure.
            route = {
                AssetKind.ADAPTER: ExposureRoute.FINE_TUNE,
                AssetKind.MODEL: ExposureRoute.TRAINING,
                AssetKind.CHECKPOINT: ExposureRoute.TRAINING,
                AssetKind.VECTOR_INDEX: ExposureRoute.RETRIEVAL,
                AssetKind.EMBEDDING_SPACE: ExposureRoute.RETRIEVAL,
                AssetKind.EVAL_SET: ExposureRoute.EVALUATION,
                AssetKind.PROMPT: ExposureRoute.CONTEXT,
                AssetKind.AGENT: ExposureRoute.CONTEXT,
                AssetKind.TOOL: ExposureRoute.CONTEXT,
                AssetKind.CORPUS: ExposureRoute.RETRIEVAL,
            }.get(kind_of(asset), ExposureRoute.UNKNOWN)
        found.append(
            Exposure(asset=asset, route=route, path=tuple(path), hops=max(0, len(path) - 1))
        )
    return sorted(found, key=lambda e: (e.route.value, str(e.asset)))


@dataclass
class Obligation:
    """What remains to be done for one exposed asset, and whether it can be done."""

    asset: DatasetId
    route: ExposureRoute
    remediation: Remediation
    discharged: bool = False
    note: str = ""

    @property
    def is_complete_if_done(self) -> bool:
        """True when performing this remediation genuinely removes the subject."""
        return self.remediation in _COMPLETE

    def __str__(self) -> str:
        state = "done" if self.discharged else "outstanding"
        return f"{self.asset}: {self.remediation.value} ({state}) — {self.note}"


def obligations(
    graph: Graph,
    origin: DatasetId,
    *,
    encrypted_per_subject: Iterable[DatasetId] = (),
    supports_approximate_unlearning: Iterable[DatasetId] = (),
) -> list[Obligation]:
    """What discharging this erasure actually requires, asset by asset.

    Two capabilities change the answer and both must be declared rather than assumed.
    Per-subject encryption makes crypto-shredding complete; without it, shredding a
    shared key destroys everyone's data or nobody's. Approximate unlearning is
    reported as available where a team says it is, and never as complete, because its
    guarantees do not survive an auditor asking what "approximate" means.
    """
    shreddable = set(encrypted_per_subject)
    approximable = set(supports_approximate_unlearning)
    out: list[Obligation] = []

    for exposure in exposures(graph, origin):
        remediation = _DEFAULT_REMEDIATION[exposure.route]
        note = ""

        if exposure.is_in_weights:
            if exposure.asset in shreddable:
                remediation = Remediation.CRYPTO_SHRED
                note = "trained on per-subject encrypted data; destroying the key is sufficient"
            elif exposure.asset in approximable:
                remediation = Remediation.APPROXIMATE_UNLEARN
                note = (
                    "approximate unlearning is available but does not prove removal; "
                    "retraining is the only complete option"
                )
            else:
                note = (
                    "the subject's data is in this model's parameters; deleting the source "
                    "rows does not remove it"
                )
        elif exposure.route is ExposureRoute.RETRIEVAL:
            note = "vectors derived from the subject remain searchable until deleted from the index"
        elif exposure.route is ExposureRoute.CONTEXT:
            note = "the subject's data appeared in a model context; purge retained request logs"
        elif exposure.route is ExposureRoute.EVALUATION:
            note = "the subject appears in an eval set and in any retained result records"
        else:
            remediation = Remediation.NONE
            note = "no route recorded; the exposure is real but the remedy is undetermined"

        out.append(
            Obligation(
                asset=exposure.asset, route=exposure.route, remediation=remediation, note=note
            )
        )
    return out


def retraining_required(graph: Graph, origin: DatasetId) -> list[DatasetId]:
    """Models whose only complete remedy is a retrain."""
    return sorted(
        (
            o.asset
            for o in obligations(graph, origin)
            if o.remediation is Remediation.RETRAIN and not o.discharged
        ),
        key=str,
    )


def is_deletion_sufficient(graph: Graph, origin: DatasetId) -> bool:
    """True only when nothing the subject reached retains them after row deletion.

    False whenever a model was trained on the data, which is the common case and the
    reason an erasure report must never be generated without consulting this.
    """
    return not any(exposure.is_in_weights for exposure in exposures(graph, origin))


def unreachable_copies(graph: Graph, origin: DatasetId) -> list[str]:
    """Copies this tool cannot destroy, stated plainly.

    Backups, replicas, and third-party endpoints are outside every lineage graph.
    Saying so on every report is the only way a partial erasure does not get filed
    as a complete one.
    """
    notes = [
        "backups and snapshots taken before this request are out of scope",
        "read replicas and downstream exports outside this graph are out of scope",
    ]
    for exposure in exposures(graph, origin):
        if exposure.is_in_weights:
            notes.append(
                f"{exposure.asset}: parameters retain information derived from the subject "
                "until the model is retrained"
            )
        if exposure.route is ExposureRoute.CONTEXT:
            notes.append(
                f"{exposure.asset}: any third-party endpoint the context was sent to holds "
                "its own copy under its own retention policy"
            )
    return notes


def extend_plan(
    graph: Graph,
    plan: ErasurePlan,
    *,
    encrypted_per_subject: Iterable[DatasetId] = (),
) -> ErasurePlan:
    """Give a plan's AI targets the reason that actually applies to them.

    The base planner already walks to every downstream dataset, so a model usually
    appears in the plan already — blocked with the generic "no adapter configured".
    That message is true and useless: no adapter will ever be able to delete a row
    from a set of weights. This rewrites those entries with the real obligation, and
    appends any AI asset the base plan missed.

    Targets stay blocked unless a shred is possible, so the plan keeps reporting
    `is_complete = False`. That is the point: the plan must not read as finished
    while a model still holds the subject's data.
    """
    shreddable = set(encrypted_per_subject)
    by_dataset = {target.dataset: index for index, target in enumerate(plan.targets)}

    for obligation in obligations(graph, plan.request.origin, encrypted_per_subject=shreddable):
        mode = (
            ErasureMode.CRYPTO_SHRED
            if obligation.remediation is Remediation.CRYPTO_SHRED
            else ErasureMode.NONE
        )
        blocked = None if mode is ErasureMode.CRYPTO_SHRED else obligation.note
        existing = by_dataset.get(obligation.asset)
        if existing is None:
            plan.targets.append(
                ErasureTarget(
                    dataset=obligation.asset,
                    partitions=frozenset({KeyPredicate()}),
                    mode=mode,
                    blocked=blocked,
                )
            )
        else:
            previous = plan.targets[existing]
            if previous.blocked:
                plan.refusals = [note for note in plan.refusals if note != previous.blocked]
            plan.targets[existing] = ErasureTarget(
                dataset=previous.dataset,
                partitions=previous.partitions,
                mode=mode,
                files=previous.files,
                blocked=blocked,
                widened=previous.widened,
            )
        if blocked:
            plan.refusals.append(f"{obligation.asset}: {blocked}")
    return plan


def estimate_retraining_cost(
    models: Sequence[DatasetId], *, cost_per_model: Mapping[DatasetId, float] | None = None
) -> float:
    """What discharging the retraining obligations would cost.

    Usually the number that decides whether a team encrypts training data per subject
    before the next request arrives rather than after.
    """
    prices = dict(cost_per_model or {})
    return sum(prices.get(model, 0.0) for model in models)


def completeness_statement(
    graph: Graph,
    origin: DatasetId,
    *,
    subject_digest: str = "",
    encrypted_per_subject: Iterable[DatasetId] = (),
) -> str:
    """The paragraph that goes in the erasure record.

    Written for a data protection officer rather than an engineer, and written to be
    accurate when it is unflattering. If a model retains the subject, this says so in
    the first sentence rather than in a field called `complete`.
    """
    items = obligations(graph, origin, encrypted_per_subject=encrypted_per_subject)
    in_weights = [o for o in items if o.route in {ExposureRoute.TRAINING, ExposureRoute.FINE_TUNE}]
    outstanding = [o for o in items if not o.discharged]

    lines = [f"# Erasure completeness — subject {subject_digest[:12] or '(undisclosed)'}…", ""]

    if in_weights:
        lines.extend(
            [
                "**This erasure is not complete.**",
                "",
                f"{len(in_weights)} model(s) were trained on data derived from this subject. "
                "Deleting the subject's rows from storage does not remove their contribution "
                "to those models' parameters. Until each is retrained without the subject, or "
                "the data it was trained on is crypto-shredded, information derived from the "
                "subject remains in the deployed system.",
                "",
                "Models affected:",
                "",
            ]
        )
        lines.append(
            md.bullets(f"{md.code(o.asset)} — remedy: {o.remediation.value}" for o in in_weights)
        )
        lines.append("")
    else:
        lines.extend(
            [
                "No model in this graph was trained on data derived from this subject. "
                "Row deletion is sufficient for the assets recorded here.",
                "",
            ]
        )

    if outstanding:
        lines.extend(["## Outstanding actions", ""])
        lines.append(
            md.bullets(f"{md.code(o.asset)}: {o.remediation.value} — {o.note}" for o in outstanding)
        )
        lines.append("")

    lines.extend(["## Out of scope", "", md.bullets(unreachable_copies(graph, origin))])
    return "\n".join(lines)


def exposure_summary(graph: Graph, origin: DatasetId) -> dict[str, int]:
    """Counts of exposed assets by route. The one-line version for a dashboard."""
    counts: dict[str, int] = {}
    for exposure in exposures(graph, origin):
        counts[exposure.route.value] = counts.get(exposure.route.value, 0) + 1
    return dict(sorted(counts.items()))


def models_exposed_to(graph: Graph, origin: DatasetId) -> list[DatasetId]:
    """Just the models, for the case where that is the whole question."""
    return sorted((e.asset for e in exposures(graph, origin) if is_model(e.asset)), key=str)
