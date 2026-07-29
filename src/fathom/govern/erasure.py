"""Partition-scoped erasure — the `erase` verb.

Deleting one subject from a lakehouse is a rewrite-the-world operation until you
know which files in which derived tables actually hold their rows. That is the same
question invalidation answers, pointed at a subject instead of a change, so this
module reuses the planner rather than reimplementing traversal: where the subject's
rows went is exactly where dirtiness would have gone.

The invariant here runs the *opposite* way to the planner's. The planner may
over-invalidate because a wasted rebuild costs money. Erasure may under-delete and
refuse, because an over-broad delete destroys data that cannot be recovered. So:

- dry run by default, always
- an explicit refusal when the storage layer cannot honour a delete, never a
  no-op that reports success
- a proof artifact naming every dataset, partition, and file touched
- the subject identifier is hashed, never written into the proof in plaintext
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from ..core.types import Capabilities, DatasetId, ErasureMode, KeyPredicate
from ..graph.model import Graph

__all__ = [
    "Eraser",
    "ErasurePlan",
    "ErasureProof",
    "ErasureRequest",
    "ErasureTarget",
    "apply_erasure",
    "plan_erasure",
    "unerasable",
]


@runtime_checkable
class Eraser(Protocol):
    """What an adapter must offer to actually destroy rows."""

    def erase(
        self,
        dataset: DatasetId,
        *,
        key_column: str,
        subject: Any,
        partitions: Sequence[KeyPredicate],
    ) -> int:
        """Delete matching rows and return how many went. Must be transactional."""
        ...


@dataclass(frozen=True)
class ErasureRequest:
    """One subject's right-to-be-forgotten request."""

    subject: Any
    key_column: str
    origin: DatasetId
    partitions: frozenset[KeyPredicate] = frozenset()
    reference: str = ""  # your ticket id, carried into the proof

    def subject_digest(self, salt: str) -> str:
        """A stable, non-reversible handle for the subject.

        Proofs are retained for years and read by people who should not learn who
        the subject was. The salt must be per-organization and secret, and it has no
        default on purpose: subject identifiers are low-entropy — an email address, a
        customer id — so an unsalted SHA-256 is reversible by anyone willing to hash
        a customer list, and the digest then identifies the subject as well as the
        raw value would.
        """
        if not salt:
            raise ValueError(
                "subject_digest requires a secret salt; an unsalted digest of a "
                "low-entropy identifier is reversible by dictionary attack. Set "
                "FATHOM_SALT, or pass salt= explicitly."
            )
        payload = f"{salt}\x1f{self.key_column}\x1f{self.subject}".encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ErasureTarget:
    """One dataset holding the subject's data, and how it can be erased."""

    dataset: DatasetId
    partitions: frozenset[KeyPredicate]
    mode: ErasureMode
    files: tuple[str, ...] = ()
    blocked: str | None = None
    widened: bool = False

    @property
    def is_actionable(self) -> bool:
        """True when this target can actually be erased from."""
        return self.blocked is None and self.mode is not ErasureMode.NONE


@dataclass
class ErasurePlan:
    """Everywhere the subject's data reached, and what can be done about it."""

    request: ErasureRequest
    targets: list[ErasureTarget] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """False means some copy of the subject's data cannot be destroyed."""
        return not self.refusals and all(t.is_actionable for t in self.targets)

    @property
    def actionable(self) -> list[ErasureTarget]:
        """Targets that can be erased from."""
        return [t for t in self.targets if t.is_actionable]

    def summary(self) -> str:
        """The plan as text, with incompleteness stated rather than buried."""
        # No digest here. A summary goes to a terminal and a CI log, and the only
        # digest this could compute without the caller's secret salt would be a
        # reversible one — see `ErasureRequest.subject_digest`. The salted digest
        # belongs in the proof, which is written deliberately.
        reference = f" [{self.request.reference}]" if self.request.reference else ""
        lines = [
            f"erasure plan on {self.request.key_column}{reference} "
            f"across {len(self.targets)} dataset(s)"
        ]
        for target in self.targets:
            keys = ", ".join(sorted(str(k) for k in target.partitions)[:3]) or "whole dataset"
            status = target.blocked or f"{target.mode.value}"
            flag = " [WIDENED]" if target.widened else ""
            lines.append(f"  {target.dataset}: {status}{flag}  {keys}")
        for refusal in self.refusals:
            lines.append(f"  REFUSED: {refusal}")
        if not self.is_complete:
            lines.append("")
            lines.append(
                "This plan is INCOMPLETE. Some copies cannot be destroyed by this tool; "
                "do not report the request as fulfilled."
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ErasureProof:
    """The artifact an auditor reads. Content-addressed so tampering is detectable."""

    subject_digest: str
    reference: str
    generated: datetime
    executed: bool
    entries: tuple[dict[str, Any], ...]
    complete: bool

    def to_json(self) -> str:
        """The proof as indented JSON, with an integrity digest appended."""
        body = {
            "subject_digest": self.subject_digest,
            "reference": self.reference,
            "generated": self.generated.isoformat(),
            "executed": self.executed,
            "complete": self.complete,
            "entries": list(self.entries),
        }
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        # The digest covers the body, so appending it cannot change what it attests.
        return json.dumps({**body, "digest": digest}, sort_keys=True, indent=2)

    @property
    def digest(self) -> str:
        """SHA-256 over the proof body.

        An integrity check, not a signature: it detects corruption and accidental
        edits, and anyone can recompute it after altering the body.
        """
        return hashlib.sha256(
            json.dumps(
                {
                    "subject_digest": self.subject_digest,
                    "reference": self.reference,
                    "generated": self.generated.isoformat(),
                    "executed": self.executed,
                    "complete": self.complete,
                    "entries": list(self.entries),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def plan_erasure(
    graph: Graph,
    request: ErasureRequest,
    *,
    capabilities: Mapping[DatasetId, Capabilities] | None = None,
    files: Mapping[DatasetId, Sequence[str]] | None = None,
) -> ErasurePlan:
    """Locate the subject's data everywhere it flowed, and check it can be destroyed.

    Reuses the invalidation planner: seeding it with the partitions holding the
    subject gives, for every downstream dataset, the partitions that could contain
    derived copies. Over-approximating here is safe — it means scanning a few extra
    partitions when deleting, not deleting extra rows, because the delete itself is
    still keyed on the subject.
    """
    capabilities = capabilities or {}
    files = files or {}
    plan = ErasurePlan(request=request)

    seeds = request.partitions or frozenset({KeyPredicate.unbounded(graph.spec(request.origin))})
    reach = graph.invalidate({request.origin: seeds})

    # Topological order, never alphabetical. A derived table must be re-derived only
    # after its sources have been erased, or it rebuilds from data that still holds
    # the subject and the erasure silently fails.
    datasets = reach.order or [request.origin]
    for dataset in datasets:
        caps = capabilities.get(dataset)
        mode = caps.erasure if caps else ErasureMode.NONE
        blocked: str | None = None

        if caps is None:
            blocked = "no adapter configured; cannot verify the data can be destroyed"
        elif mode is ErasureMode.NONE:
            blocked = (
                "storage refuses deletion (Object Lock, WORM, or an adapter without "
                "erasure support); crypto-shredding is the only remaining option"
            )

        plan.targets.append(
            ErasureTarget(
                dataset=dataset,
                partitions=reach.partitions(dataset) or frozenset({KeyPredicate()}),
                mode=mode,
                files=tuple(files.get(dataset, ())),
                blocked=blocked,
                widened=dataset in reach.widened,
            )
        )

    # Backups and replicas are out of scope and must be said out loud, every time.
    plan.refusals.extend(t.blocked for t in plan.targets if t.blocked is not None)
    return plan


def apply_erasure(
    plan: ErasurePlan,
    erasers: Mapping[DatasetId, Eraser],
    *,
    salt: str,
    dry_run: bool = True,
) -> ErasureProof:
    """Execute an erasure plan, or describe what it would do.

    `salt` has no default. It is the secret that makes the proof's subject digest
    non-reversible, and defaulting it to the empty string produced a proof that
    named the subject to anyone holding a customer list — see
    `ErasureRequest.subject_digest`.

    `dry_run` defaults to True and must be turned off explicitly. A plan with any
    refusal in it will still execute the parts it can, but the proof records
    `complete: false` so nobody can mistake a partial erasure for a finished one.
    """
    entries: list[dict[str, Any]] = []
    executed_any = False

    for target in plan.targets:
        entry: dict[str, Any] = {
            "dataset": str(target.dataset),
            "mode": target.mode.value,
            "partitions": sorted(str(k) for k in target.partitions),
            "files": list(target.files),
            "widened": target.widened,
        }
        if target.blocked:
            entry["status"] = "blocked"
            entry["reason"] = target.blocked
            entries.append(entry)
            continue

        if dry_run:
            # A dry run is a plan. Demanding an executor to describe what would happen
            # would make the plan-only path report everything as blocked.
            entry["status"] = "planned"
            entries.append(entry)
            continue

        eraser = erasers.get(target.dataset)
        if eraser is None:
            entry["status"] = "blocked"
            entry["reason"] = "no eraser supplied for this dataset"
            entries.append(entry)
            continue

        deleted = eraser.erase(
            target.dataset,
            key_column=plan.request.key_column,
            subject=plan.request.subject,
            partitions=sorted(target.partitions, key=str),
        )
        executed_any = True
        entry["status"] = "erased"
        entry["rows_deleted"] = deleted
        entries.append(entry)

    complete = plan.is_complete and all(e.get("status") == "erased" for e in entries)

    return ErasureProof(
        subject_digest=plan.request.subject_digest(salt),
        reference=plan.request.reference,
        generated=datetime.now(UTC),
        executed=executed_any,
        entries=tuple(entries),
        complete=complete and not dry_run,
    )


def unerasable(plan: ErasurePlan) -> Iterable[DatasetId]:
    """Datasets where the subject's data will survive this request."""
    return (t.dataset for t in plan.targets if not t.is_actionable)
