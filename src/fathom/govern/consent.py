"""Purpose limitation and data residency, propagated along lineage.

Consent is collected once, at one place, for a stated set of purposes. The data then
travels — into a warehouse, into a feature store, into a model — and at every hop the
original purpose becomes less visible, until somebody trains a recommender on records
whose subjects agreed only to fraud detection. Nobody decided to do that. It happened
because purpose is metadata on a source table and the model is nine joins away.

Purposes propagate downstream by *intersection*, which is the opposite of how labels
propagate. A dataset joining a fraud-only source with a marketing-only source may be
used for neither: the permitted set is what every input permits, not what any input
permits. Getting this backwards is the single most common modelling error in consent
tooling, and it fails open.

Residency works the same way and is checked differently. A dataset inherits the
strictest region constraint of anything upstream, and a violation is a dataset
*stored* outside a region its inputs are pinned to. That check needs the storage
location, which the identity already carries — an `s3://eu-lake` namespace is
evidence, not an annotation somebody has to maintain.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from ..core.types import DatasetId
from ..core.util.clock import as_utc
from ..core.util.clock import now as _now
from ..graph.model import Graph
from ..graph.query import ancestors, closure, fold_downstream, shortest_path

__all__ = [
    "ConsentScope",
    "Purpose",
    "Residency",
    "ConsentReport",
    "blocking_sources",
    "expired",
    "purposes_breakdown",
    "region_of",
    "report",
    "training_permitted_datasets",
    "permitted_purposes",
    "propagate_purposes",
    "purpose_allowed",
    "residency_violations",
    "retention_violations",
    "transfer_paths",
    "unconsented_uses",
]


class Purpose(StrEnum):
    """Why data may be processed. Deliberately coarse; these are the ones consent forms use."""

    SERVICE = "service"  # delivering the thing the user asked for
    ANALYTICS = "analytics"
    PERSONALIZATION = "personalization"
    MARKETING = "marketing"
    FRAUD = "fraud"
    TRAINING = "training"  # training a model, which several regimes treat separately
    RESEARCH = "research"
    LEGAL = "legal"  # retention obligations that survive a deletion request


@dataclass(frozen=True)
class ConsentScope:
    """What one dataset may be used for, and until when.

    `basis` is the lawful basis claimed — consent, contract, legitimate interest.
    Recorded rather than inferred, because the basis decides whether an objection has
    to be honoured and no amount of lineage can determine it.
    """

    dataset: DatasetId
    purposes: frozenset[Purpose] = frozenset()
    basis: str = "consent"
    collected: datetime | None = None
    retention: timedelta | None = None
    regions: frozenset[str] = frozenset()

    @property
    def expires(self) -> datetime | None:
        """When this scope lapses, or None when no retention was recorded."""
        if self.collected is None or self.retention is None:
            return None
        return as_utc(self.collected) + self.retention

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """True when the retention period has run out.

        A scope with no retention declared never expires here; that is a statement
        about what was recorded, not a claim that consent is perpetual.
        """
        stamp = self.expires
        return stamp is not None and stamp < _now(now)

    def allows(self, purpose: Purpose, *, now: datetime | None = None) -> bool:
        """Whether this scope permits a purpose *today*.

        Expiry is checked here rather than left to a separate report. Retention was
        recorded, the deadline passed, and answering "yes, you may train on it" from
        the purpose set alone is the failure the retention field exists to prevent.
        """
        return not self.is_expired(now=now) and purpose in self.purposes

    def __str__(self) -> str:
        names = ", ".join(sorted(p.value for p in self.purposes)) or "none"
        return f"{self.dataset}: {names} ({self.basis})"


@dataclass(frozen=True)
class Residency:
    """Where a dataset's contents are permitted to be stored."""

    dataset: DatasetId
    regions: frozenset[str] = frozenset()
    strict: bool = True  # False means "prefer", True means "must not leave"

    def permits(self, region: str) -> bool:
        """True when this region is allowed, or no restriction was declared."""
        return not self.regions or region in self.regions


def propagate_purposes(
    graph: Graph,
    declared: Mapping[DatasetId, ConsentScope],
    *,
    now: datetime | None = None,
) -> dict[DatasetId, frozenset[Purpose]]:
    """Flow permitted purposes downstream, intersecting at every join.

    A dataset with no declared scope and no declared ancestor gets the empty set,
    which reads as "nothing is permitted". That fails closed, which is correct: an
    unlabelled dataset is not one you have established permission to train on.
    """
    return fold_downstream(
        graph,
        {
            ds: frozenset() if scope.is_expired(now=now) else scope.purposes
            for ds, scope in declared.items()
        },
        combine=lambda a, b: a & b,
        default=frozenset(),
    )


def permitted_purposes(
    graph: Graph,
    ds: DatasetId,
    declared: Mapping[DatasetId, ConsentScope],
    *,
    now: datetime | None = None,
) -> frozenset[Purpose]:
    """What this dataset may be used for, its whole upstream taken into account.

    An expired scope anywhere in the closure contributes nothing, so the
    intersection empties and the dataset permits nothing. That is the same
    fail-closed rule an undeclared scope follows: retention running out is a
    withdrawal of permission, not a note for a separate report.
    """
    scopes = [
        frozenset() if declared[node].is_expired(now=now) else declared[node].purposes
        for node in closure(graph, ds)
        if node in declared
    ]
    if not scopes:
        return frozenset()
    permitted = scopes[0]
    for scope in scopes[1:]:
        permitted &= scope
    return permitted


def purpose_allowed(
    graph: Graph,
    ds: DatasetId,
    purpose: Purpose,
    declared: Mapping[DatasetId, ConsentScope],
    *,
    now: datetime | None = None,
) -> bool:
    """Whether one specific use of a dataset is permitted today."""
    return purpose in permitted_purposes(graph, ds, declared, now=now)


def blocking_sources(
    graph: Graph, ds: DatasetId, purpose: Purpose, declared: Mapping[DatasetId, ConsentScope]
) -> list[DatasetId]:
    """The upstream datasets whose consent scope forbids this purpose.

    The actionable half of a refusal: either drop these inputs or re-collect consent
    for them. A bare "not permitted" leaves a team with nowhere to start.
    """
    return sorted(
        (
            node
            for node in closure(graph, ds)
            if node in declared and purpose not in declared[node].purposes
        ),
        key=str,
    )


def unconsented_uses(
    graph: Graph,
    declared: Mapping[DatasetId, ConsentScope],
    *,
    intended: Mapping[DatasetId, Purpose],
) -> list[str]:
    """Datasets used for a purpose their upstream consent does not cover."""
    out: list[str] = []
    for ds, purpose in sorted(intended.items(), key=lambda kv: str(kv[0])):
        if purpose_allowed(graph, ds, purpose, declared):
            continue
        blockers = blocking_sources(graph, ds, purpose, declared)
        detail = ", ".join(str(node) for node in blockers[:3]) or "no consent recorded upstream"
        out.append(f"{ds}: used for `{purpose.value}`, not permitted by {detail}")
    return out


# -- residency -----------------------------------------------------------------

# Standard cloud region codes. Matched as whole tokens so `eu-west-1-lake` yields
# `eu-west-1` and a bucket merely called `useful-data` yields nothing.
_AWS_REGION = re.compile(
    r"\b(?:us-gov-|us-|eu-|ap-|ca-|cn-|sa-|me-|af-|il-)"
    r"(?:central|north|south|east|west|northeast|northwest|southeast|southwest)-\d\b"
)
_GCP_REGION = re.compile(
    r"\b(?:us|europe|asia|australia|southamerica|northamerica|me)-"
    r"(?:central|north|south|east|west|northeast|northwest|southeast|southwest)\d\b"
)
_AZURE_REGION = re.compile(
    r"\b(?:west|east|north|south|central)(?:us|europe|india|uk|asia)\d?\b"
    r"|\b(?:uk|japan|korea|france|germany|norway|switzerland|brazil|canada)"
    r"(?:south|west|east|north|central)\b"
)


def region_of(ds: DatasetId) -> str:
    """The region a dataset is stored in, read off its namespace when it is there.

    Object-storage namespaces routinely encode the region — `s3://eu-west-1-lake`,
    `gs://europe-west4-warehouse`. Only the standard cloud region codes are matched;
    anything else returns an empty string rather than a guess, because a wrong region
    turns a residency check into a false accusation.
    """
    lowered = ds.namespace.lower()
    for pattern in (_AWS_REGION, _GCP_REGION, _AZURE_REGION):
        match = pattern.search(lowered)
        if match:
            return match.group(0)
    return ""


def residency_violations(graph: Graph, constraints: Mapping[DatasetId, Residency]) -> list[str]:
    """Datasets stored outside a region their upstream constraints require.

    Checks the storage location against every constraint inherited from upstream, so
    a copy of EU-pinned data landing in a US bucket is caught at the copy rather than
    at the audit.
    """
    out: list[str] = []
    for ds in graph.datasets:
        region = region_of(ds)
        if not region:
            continue
        for node in closure(graph, ds):
            constraint = constraints.get(node)
            if constraint is None or not constraint.strict or constraint.permits(region):
                continue
            out.append(
                f"{ds} is stored in `{region}` but inherits a residency constraint from "
                f"{node} limiting it to {sorted(constraint.regions)}"
            )
    return sorted(set(out))


def transfer_paths(
    graph: Graph, constraints: Mapping[DatasetId, Residency]
) -> list[list[DatasetId]]:
    """Routes by which constrained data reaches a dataset outside its permitted regions."""
    found: list[list[DatasetId]] = []
    for source, constraint in sorted(constraints.items(), key=lambda kv: str(kv[0])):
        if not constraint.strict:
            continue
        for target in graph.datasets:
            region = region_of(target)
            if not region or constraint.permits(region):
                continue
            path = shortest_path(graph, source, target)
            if path:
                found.append(path)
    return found


# -- retention -----------------------------------------------------------------


def expired(
    declared: Mapping[DatasetId, ConsentScope], *, now: datetime | None = None
) -> list[DatasetId]:
    """Datasets whose retention period has run out."""
    reference = _now(now)
    out: list[DatasetId] = []
    for ds, scope in declared.items():
        stamp = scope.expires
        if stamp is not None and stamp < reference:
            out.append(ds)
    return sorted(out, key=str)


def retention_violations(
    graph: Graph, declared: Mapping[DatasetId, ConsentScope], *, now: datetime | None = None
) -> list[str]:
    """Expired data, and everything downstream that still contains it.

    The downstream half is what makes this useful. Deleting an expired source while
    leaving four derived tables built from it is the normal outcome of a retention
    policy that stops at the source.
    """
    out: list[str] = []
    for ds in expired(declared, now=now):
        scope = declared[ds]
        out.append(f"{ds}: retention expired {scope.expires.isoformat() if scope.expires else ''}")
        for downstream in graph.datasets:
            if downstream == ds:
                continue
            if ds in ancestors(graph, downstream):
                out.append(f"  {downstream} is derived from it and holds the same data")
    return out


@dataclass
class ConsentReport:
    """One dataset's consent position, everything upstream taken into account."""

    dataset: DatasetId
    permitted: frozenset[Purpose] = frozenset()
    denied: list[str] = field(default_factory=list)
    expired_sources: list[DatasetId] = field(default_factory=list)
    residency: list[str] = field(default_factory=list)

    @property
    def is_clear(self) -> bool:
        """True when no purpose-limitation breach was proven."""
        return not (self.denied or self.expired_sources or self.residency)

    def summary(self) -> str:
        """The report as text."""
        names = ", ".join(sorted(p.value for p in self.permitted)) or "nothing"
        lines = [f"{self.dataset}: permitted for {names}"]
        lines.extend(f"  denied: {note}" for note in self.denied)
        lines.extend(f"  expired upstream: {ds}" for ds in self.expired_sources)
        lines.extend(f"  residency: {note}" for note in self.residency)
        return "\n".join(lines)


def report(
    graph: Graph,
    ds: DatasetId,
    declared: Mapping[DatasetId, ConsentScope],
    *,
    intended: Iterable[Purpose] = (),
    constraints: Mapping[DatasetId, Residency] | None = None,
    now: datetime | None = None,
) -> ConsentReport:
    """Assemble one dataset's full consent position."""
    permitted = permitted_purposes(graph, ds, declared)
    out = ConsentReport(dataset=ds, permitted=permitted)

    for purpose in intended:
        if purpose not in permitted:
            blockers = blocking_sources(graph, ds, purpose, declared)
            out.denied.append(
                f"`{purpose.value}` — blocked by "
                + (", ".join(str(node) for node in blockers[:3]) or "no recorded consent")
            )

    upstream = {*ancestors(graph, ds), ds}
    out.expired_sources = [node for node in expired(declared, now=now) if node in upstream]

    if constraints:
        region = region_of(ds)
        for node in upstream:
            constraint = constraints.get(node)
            if constraint and constraint.strict and region and not constraint.permits(region):
                out.residency.append(
                    f"stored in `{region}`, constrained to {sorted(constraint.regions)} by {node}"
                )
    return out


def purposes_breakdown(declared: Mapping[DatasetId, ConsentScope]) -> dict[str, int]:
    """How many datasets permit each purpose."""
    counts: dict[str, int] = {}
    for scope in declared.values():
        for purpose in scope.purposes:
            counts[purpose.value] = counts.get(purpose.value, 0) + 1
    return dict(sorted(counts.items()))


def training_permitted_datasets(
    graph: Graph, declared: Mapping[DatasetId, ConsentScope]
) -> list[DatasetId]:
    """Datasets that may lawfully be used to train a model.

    The list a training pipeline should be restricted to, rather than the list it
    happens to have access to.
    """
    return sorted(
        (ds for ds in graph.datasets if purpose_allowed(graph, ds, Purpose.TRAINING, declared)),
        key=str,
    )
