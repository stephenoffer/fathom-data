"""Copies of the data that no adapter can see, declared so a proof can name them.

Every erasure artifact this library produces carries the same sentence: it covers what
the configured adapters can reach. That is honest and it is not enough. "There may be
copies elsewhere" is unfalsifiable — it cannot be reviewed, cannot be closed out, and
cannot be told apart from "we did not look".

The gap is specific and it is always the same list. A nightly snapshot in another
account. A read replica for the BI tool. A CSV a partner receives monthly. A Kafka
topic with seven-day retention. A vendor's system that ingested an export in 2023.
None of them appear in any lineage graph, because nothing about them is derivable —
they are facts about an organization, not about its SQL.

So they are **declared**. A declared replica turns an unfalsifiable caveat into a
checklist with owners and dates, which is the difference between a proof an auditor
accepts and one they return.

**Nothing here deletes anything.** Every function reports what exists, what it would
take, and what has been attested. The `erase` verb refuses to act outside what it can
verify, and this module does not widen that — it widens what the *report* is honest
about.

**Two directions the module errs, both toward saying more:**

- An undeclared replica is invisible, so `coverage` reports how much of the estate has
  been declared at all, and a low number reads as "we have not mapped this" rather
  than "we are clean".
- A replica whose retention has not elapsed and which nobody has attested is
  `OUTSTANDING`, never "assumed expired". Time passing is not evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from ..core.types import DatasetId
from ..core.util.clock import as_utc, now

__all__ = [
    "BEYOND_CONTROL",
    "Attestation",
    "CopyKind",
    "Disposition",
    "Replica",
    "ReplicaReport",
    "attested",
    "by_kind",
    "coverage",
    "declare",
    "discharged",
    "disposition_of",
    "expired_by",
    "for_dataset",
    "obligations_for",
    "outstanding",
    "proof_entries",
    "report",
    "undeclared_datasets",
    "unreachable_summary",
]


class CopyKind(StrEnum):
    """Where a copy lives. Ordered roughly by how hard it is to reach."""

    SNAPSHOT = "snapshot"  # a point-in-time backup, usually restorable but not editable
    REPLICA = "replica"  # a live mirror; deleting upstream usually propagates
    ARCHIVE = "archive"  # cold storage, often WORM, often the hardest
    EXPORT = "export"  # a file handed to somebody; cannot be recalled
    STREAM = "stream"  # a topic or log with a retention window
    THIRD_PARTY = "third_party"  # a vendor's system; only they can delete it
    CACHE = "cache"  # a derived store that repopulates itself


# Kinds where the copy has left the organization's control. Deleting the source does
# nothing to these, and no retention window closes them on its own.
BEYOND_CONTROL = frozenset({CopyKind.EXPORT, CopyKind.THIRD_PARTY})

# Kinds that follow the source without further action, because the copy is not
# independent of it.
FOLLOWS_SOURCE = frozenset({CopyKind.REPLICA, CopyKind.CACHE})


class Disposition(StrEnum):
    """What is true of one copy for one erasure, right now."""

    FOLLOWS_SOURCE = "follows_source"  # deleting upstream reaches it; nothing more to do
    EXPIRED = "expired"  # its retention window has demonstrably elapsed
    ATTESTED = "attested"  # somebody recorded having handled it
    OUTSTANDING = "outstanding"  # nothing has closed it out
    UNREACHABLE = "unreachable"  # beyond control; only the holder can act


@dataclass(frozen=True)
class Replica:
    """One declared copy of a dataset that lineage cannot see.

    `retention` is how long the copy lives after it is written. Absent means unbounded,
    which is the honest default: a backup nobody set a lifecycle rule on does not
    quietly expire, and assuming it does is how an erasure gets filed as complete.
    """

    dataset: DatasetId
    kind: CopyKind
    location: str  # free text: an account, a bucket, a vendor name
    owner: str = ""
    retention: timedelta | None = None
    note: str = ""

    @property
    def is_beyond_control(self) -> bool:
        return self.kind in BEYOND_CONTROL

    @property
    def follows_source(self) -> bool:
        return self.kind in FOLLOWS_SOURCE

    def __str__(self) -> str:
        where = f"{self.kind.value} at {self.location}"
        return f"{self.dataset}: {where}" + (f" ({self.owner})" if self.owner else "")


@dataclass(frozen=True)
class Attestation:
    """A record that somebody handled one copy for one subject."""

    location: str
    at: datetime
    by: str = ""
    note: str = ""


def declare(
    dataset: DatasetId,
    kind: CopyKind | str,
    location: str,
    *,
    owner: str = "",
    retention: timedelta | None = None,
    note: str = "",
) -> Replica:
    """Declare a copy, accepting the kind as a string so config can build one."""
    resolved = kind if isinstance(kind, CopyKind) else CopyKind(str(kind).lower())
    if not location.strip():
        raise ValueError(
            "a replica needs a location; 'somewhere else' is the caveat this module "
            "exists to replace"
        )
    return Replica(
        dataset=dataset,
        kind=resolved,
        location=location.strip(),
        owner=owner,
        retention=retention,
        note=note,
    )


def disposition_of(
    replica: Replica,
    *,
    erased_at: datetime,
    attestations: Mapping[str, Attestation] | None = None,
    at: datetime | None = None,
) -> Disposition:
    """What is true of one copy, given when the source was erased.

    Order matters and is deliberate. A copy beyond the organization's control is
    reported as such even if somebody attested to it, because an attestation about a
    third party is a claim about someone else's system — worth recording, and not the
    same as having done it.
    """
    if replica.is_beyond_control:
        return Disposition.UNREACHABLE
    if replica.follows_source:
        return Disposition.FOLLOWS_SOURCE

    moment = at or now()
    if replica.retention is not None and as_utc(moment) - as_utc(erased_at) >= replica.retention:
        return Disposition.EXPIRED
    if attestations and replica.location in attestations:
        return Disposition.ATTESTED
    return Disposition.OUTSTANDING


@dataclass
class ReplicaReport:
    """Every declared copy of one dataset, and what remains open."""

    dataset: DatasetId
    by_disposition: dict[Disposition, list[Replica]] = field(default_factory=dict)

    @property
    def outstanding(self) -> list[Replica]:
        """Copies nothing has closed out. The work list."""
        return self.by_disposition.get(Disposition.OUTSTANDING, [])

    @property
    def unreachable(self) -> list[Replica]:
        """Copies only their holder can act on. The letters to write."""
        return self.by_disposition.get(Disposition.UNREACHABLE, [])

    @property
    def is_discharged(self) -> bool:
        """True only when nothing is outstanding *and* nothing is beyond control.

        A dataset with an outstanding export is not discharged, however many internal
        copies were handled — which is why this is one property rather than two.
        """
        return not self.outstanding and not self.unreachable

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_disposition.values())

    def summary(self) -> str:
        if self.total == 0:
            return (
                f"{self.dataset}: no copies declared. That is not the same as none "
                "existing — snapshots, replicas, and exports are facts about an "
                "organization, not about its SQL, so nothing can derive them."
            )
        lines = [f"{self.dataset}: {self.total} declared copy(ies)"]
        for disposition in Disposition:
            found = self.by_disposition.get(disposition, [])
            if found:
                lines.append(f"    {disposition.value}: {len(found)}")
                lines.extend(f"        {r}" for r in found)
        if self.unreachable:
            lines.append(
                "    Copies beyond this organization's control cannot be discharged "
                "here. Only the holder can act, and a record of asking is not a record "
                "of it being done."
            )
        if self.outstanding:
            lines.append(
                f"    {len(self.outstanding)} copy(ies) outstanding. Time passing is "
                "not evidence; these need an attestation or an elapsed retention window."
            )
        return "\n".join(lines)


def report(
    dataset: DatasetId,
    replicas: Iterable[Replica],
    *,
    erased_at: datetime,
    attestations: Sequence[Attestation] = (),
    at: datetime | None = None,
) -> ReplicaReport:
    """Classify every declared copy of one dataset."""
    by_location = {a.location: a for a in attestations}
    result = ReplicaReport(dataset=dataset)
    for replica in replicas:
        if replica.dataset != dataset:
            continue
        disposition = disposition_of(replica, erased_at=erased_at, attestations=by_location, at=at)
        result.by_disposition.setdefault(disposition, []).append(replica)
    return result


def obligations_for(
    replicas: Iterable[Replica], *, erased_at: datetime, at: datetime | None = None
) -> list[tuple[Replica, str]]:
    """Every copy still needing action, with what that action is.

    Sorted so the ones somebody else has to perform come last: those need a letter and
    a follow-up, and mixing them into the internal work list is how they get lost.
    """
    out: list[tuple[Replica, str]] = []
    for replica in replicas:
        disposition = disposition_of(replica, erased_at=erased_at, at=at)
        if disposition is Disposition.OUTSTANDING:
            out.append(
                (
                    replica,
                    f"delete or overwrite the {replica.kind.value} at {replica.location}"
                    + (f", owned by {replica.owner}" if replica.owner else "")
                    + ", then record an attestation",
                )
            )
        elif disposition is Disposition.UNREACHABLE:
            out.append(
                (
                    replica,
                    f"only {replica.owner or 'the holder'} can delete the "
                    f"{replica.kind.value} at {replica.location}; request it in writing "
                    "and record the response",
                )
            )
    out.sort(key=lambda pair: (pair[0].is_beyond_control, str(pair[0].dataset), pair[0].location))
    return out


def outstanding(
    replicas: Iterable[Replica], *, erased_at: datetime, at: datetime | None = None
) -> list[Replica]:
    """Copies nothing has closed out."""
    return [
        r
        for r in replicas
        if disposition_of(r, erased_at=erased_at, at=at) is Disposition.OUTSTANDING
    ]


def attested(
    replicas: Iterable[Replica],
    attestations: Sequence[Attestation],
    *,
    erased_at: datetime,
    at: datetime | None = None,
) -> list[Replica]:
    """Copies somebody recorded having handled."""
    by_location = {a.location: a for a in attestations}
    return [
        r
        for r in replicas
        if disposition_of(r, erased_at=erased_at, attestations=by_location, at=at)
        is Disposition.ATTESTED
    ]


def expired_by(
    replicas: Iterable[Replica], *, erased_at: datetime, at: datetime | None = None
) -> list[Replica]:
    """Copies whose retention window has demonstrably elapsed."""
    return [
        r for r in replicas if disposition_of(r, erased_at=erased_at, at=at) is Disposition.EXPIRED
    ]


def coverage(datasets: Sequence[DatasetId], replicas: Iterable[Replica]) -> float:
    """Fraction of datasets with at least one declared copy.

    Deliberately ambiguous and worth stating as such: a low number means either that
    most datasets genuinely have no copies, or that nobody has mapped them. Those are
    not distinguishable from here, and only one of them is good news.
    """
    if not datasets:
        return 0.0
    declared = {r.dataset for r in replicas}
    return len([d for d in datasets if d in declared]) / len(datasets)


def undeclared_datasets(
    datasets: Sequence[DatasetId], replicas: Iterable[Replica]
) -> list[DatasetId]:
    """Datasets with no declared copy at all — the ones an erasure proof cannot bound."""
    declared = {r.dataset for r in replicas}
    return sorted((d for d in datasets if d not in declared), key=str)


def unreachable_summary(replicas: Iterable[Replica]) -> str:
    """The sentence an erasure proof should carry, built from what was declared.

    Replaces "there may be copies elsewhere" with a list. An auditor can close a list.
    """
    beyond = [r for r in replicas if r.is_beyond_control]
    if not beyond:
        return (
            "No copies beyond this organization's control were declared. Absence of a "
            "declaration is not evidence of absence — nothing can derive an export or "
            "a vendor's copy from lineage."
        )
    lines = [
        f"{len(beyond)} declared copy(ies) are beyond this organization's control and "
        "cannot be discharged by any action taken here:"
    ]
    for replica in sorted(beyond, key=lambda r: (str(r.dataset), r.location)):
        who = replica.owner or "holder unknown"
        lines.append(f"  - {replica.kind.value} at {replica.location} ({who})")
    return "\n".join(lines)


def by_kind(replicas: Iterable[Replica]) -> dict[CopyKind, list[Replica]]:
    """Declared copies grouped by what they are."""
    out: dict[CopyKind, list[Replica]] = {}
    for replica in replicas:
        out.setdefault(replica.kind, []).append(replica)
    return out


def for_dataset(replicas: Iterable[Replica], dataset: DatasetId) -> list[Replica]:
    """Every declared copy of one dataset."""
    return [r for r in replicas if r.dataset == dataset]


def proof_entries(
    replicas_for_plan: Iterable[Replica],
    *,
    erased_at: datetime,
    attestations: Sequence[Attestation] = (),
    at: datetime | None = None,
) -> list[dict[str, str]]:
    """Declared copies as entries an `ErasureProof` can carry.

    The point of the whole module: a proof that lists five named copies with owners
    and dispositions is one an auditor can close out, and "there may be copies
    elsewhere" is one they return.
    """
    by_location = {a.location: a for a in attestations}
    out: list[dict[str, str]] = []
    for replica in sorted(replicas_for_plan, key=lambda r: (str(r.dataset), r.location)):
        disposition = disposition_of(replica, erased_at=erased_at, attestations=by_location, at=at)
        entry = {
            "dataset": str(replica.dataset),
            "kind": replica.kind.value,
            "location": replica.location,
            "disposition": disposition.value,
        }
        if replica.owner:
            entry["owner"] = replica.owner
        found = by_location.get(replica.location)
        if found is not None:
            entry["attested_at"] = as_utc(found.at).isoformat()
            if found.by:
                entry["attested_by"] = found.by
        out.append(entry)
    return out


def discharged(
    replicas_for_plan: Iterable[Replica],
    *,
    erased_at: datetime,
    attestations: Sequence[Attestation] = (),
    at: datetime | None = None,
) -> bool:
    """True when every declared copy is closed out and none is beyond control.

    A proof should not report `complete` while this is false, however clean the
    adapter-visible part was. That is the whole gap this module exists to close.
    """
    by_location = {a.location: a for a in attestations}
    return all(
        disposition_of(r, erased_at=erased_at, attestations=by_location, at=at)
        in {Disposition.FOLLOWS_SOURCE, Disposition.EXPIRED, Disposition.ATTESTED}
        for r in replicas_for_plan
    )
