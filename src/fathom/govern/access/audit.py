"""An append-only record of who did what, that can be shown to have not been edited.

The store records outcomes. It does not record actors, so "who ran the erasure" and
"who confirmed the label that let personal data into the training set" are both
unanswerable — and both are the first question asked after something goes wrong.

The log is hash-chained: each entry commits to the digest of the one before it, so
removing or altering any entry breaks every digest after it. That is tamper
*evidence*, not tamper prevention. Someone with write access can still truncate the
tail; what they cannot do is quietly change entry 40 of 900 and leave the rest
verifiable. `verify` reports the first index where the chain breaks, because knowing
a log was altered is much less useful than knowing where.

Two things deliberately do not go in an entry: the value of anything sensitive, and
free-text prose. Entries carry an action, a target, and structured detail, so the
log stays queryable and so an audit log never becomes the place personal data leaks
to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "AuditEntry",
    "AuditLog",
    "AuditOutcome",
    "ChainBreak",
    "GENESIS",
    "entries_for",
    "entries_since",
    "digest_of",
    "record",
    "replay",
    "summarize",
    "verify",
    "who_touched",
]

# The digest an empty chain commits to. Fixed so an empty log verifies rather than
# being a special case every caller has to remember.
GENESIS = "0" * 64


class AuditOutcome(StrEnum):
    """Whether the recorded action succeeded.

    Denials are recorded too. A log containing only successes cannot answer "did
    anyone try", which is most of what an access review is looking for.
    """

    SUCCESS = "success"
    DENIED = "denied"
    FAILED = "failed"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class AuditEntry:
    """One recorded action.

    `previous` and `digest` form the chain. `detail` is structured rather than prose
    so the log stays queryable and never becomes the place free text leaks into.
    """

    sequence: int
    at: datetime
    actor: str
    action: str
    target: str = ""
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    tenant: str = ""
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)
    previous: str = GENESIS
    digest: str = ""

    def body(self) -> dict[str, Any]:
        """The part the digest commits to. Deliberately excludes the digest itself."""
        return {
            "sequence": self.sequence,
            "at": self.at.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "outcome": self.outcome.value,
            "tenant": self.tenant,
            "reason": self.reason,
            "detail": dict(self.detail),
            "previous": self.previous,
        }

    def compute_digest(self) -> str:
        payload = json.dumps(self.body(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps({**self.body(), "digest": self.digest}, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, raw: str) -> AuditEntry:
        blob = json.loads(raw)
        return cls(
            sequence=int(blob["sequence"]),
            at=datetime.fromisoformat(blob["at"]),
            actor=str(blob["actor"]),
            action=str(blob["action"]),
            target=str(blob.get("target", "")),
            outcome=AuditOutcome(blob.get("outcome", "success")),
            tenant=str(blob.get("tenant", "")),
            reason=str(blob.get("reason", "")),
            detail=dict(blob.get("detail", {})),
            previous=str(blob.get("previous", GENESIS)),
            digest=str(blob.get("digest", "")),
        )


def digest_of(entry: AuditEntry) -> str:
    return entry.compute_digest()


@dataclass(frozen=True)
class ChainBreak:
    """Where verification failed, and how.

    The index matters more than the fact. A log known to be altered somewhere is
    nearly useless; a log altered at entry 40 of 900 leaves 39 entries trustworthy
    and points at when it happened.
    """

    index: int
    sequence: int
    problem: str

    def summary(self) -> str:
        return f"entry {self.index} (sequence {self.sequence}): {self.problem}"


@dataclass
class AuditLog:
    """An append-only, hash-chained log."""

    entries: list[AuditEntry] = field(default_factory=list)

    @property
    def head(self) -> str:
        return self.entries[-1].digest if self.entries else GENESIS

    def __len__(self) -> int:
        return len(self.entries)

    def append(
        self,
        *,
        actor: str,
        action: str,
        target: str = "",
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        tenant: str = "",
        reason: str = "",
        detail: Mapping[str, Any] | None = None,
        at: datetime | None = None,
    ) -> AuditEntry:
        """Add an entry, chaining it to the current head."""
        entry = AuditEntry(
            sequence=len(self.entries),
            at=at or datetime.now(UTC),
            actor=actor,
            action=action,
            target=target,
            outcome=outcome,
            tenant=tenant,
            reason=reason,
            detail=dict(detail or {}),
            previous=self.head,
        )
        sealed = AuditEntry(
            **{
                **entry.body(),
                "at": entry.at,
                "outcome": entry.outcome,
                "digest": entry.compute_digest(),
            }
        )
        self.entries.append(sealed)
        return sealed


def record(
    log: AuditLog,
    actor: str,
    action: str,
    *,
    target: str = "",
    outcome: AuditOutcome = AuditOutcome.SUCCESS,
    tenant: str = "",
    reason: str = "",
    detail: Mapping[str, Any] | None = None,
    at: datetime | None = None,
) -> AuditEntry:
    """Append an entry. The function form, for call sites that hold a log."""
    return log.append(
        actor=actor,
        action=action,
        target=target,
        outcome=outcome,
        tenant=tenant,
        reason=reason,
        detail=detail,
        at=at,
    )


def verify(log: AuditLog | Sequence[AuditEntry]) -> list[ChainBreak]:
    """Check the chain, reporting every break with its index.

    Four failure modes are distinguished, because they mean different things: a
    recomputed digest that differs means the entry was edited; a `previous` that
    does not match means an entry was removed or reordered; a sequence gap means a
    truncation; and an empty digest means the entry was never sealed.
    """
    entries = list(log.entries if isinstance(log, AuditLog) else log)
    breaks: list[ChainBreak] = []
    expected_previous = GENESIS

    for index, entry in enumerate(entries):
        if not entry.digest:
            breaks.append(ChainBreak(index, entry.sequence, "entry was never sealed"))
        elif entry.compute_digest() != entry.digest:
            breaks.append(ChainBreak(index, entry.sequence, "content was altered after sealing"))
        if entry.previous != expected_previous:
            breaks.append(
                ChainBreak(
                    index,
                    entry.sequence,
                    "chain does not link to the previous entry; one was removed or reordered",
                )
            )
        if entry.sequence != index:
            breaks.append(
                ChainBreak(index, entry.sequence, f"sequence {entry.sequence} at position {index}")
            )
        expected_previous = entry.digest

    return breaks


def replay(entries: Iterable[str]) -> AuditLog:
    """Rebuild a log from serialised entries, preserving digests for verification."""
    log = AuditLog()
    log.entries = [AuditEntry.from_json(raw) for raw in entries]
    return log


def entries_for(
    log: AuditLog, *, actor: str = "", action: str = "", target: str = ""
) -> list[AuditEntry]:
    """Filter. Every argument is optional and they conjoin."""
    return [
        e
        for e in log.entries
        if (not actor or e.actor == actor)
        and (not action or e.action == action)
        and (not target or e.target == target)
    ]


def entries_since(log: AuditLog, when: datetime) -> list[AuditEntry]:
    return [e for e in log.entries if e.at >= when]


def who_touched(log: AuditLog, target: str) -> list[str]:
    """Every actor who acted on a target. The question asked after an incident."""
    return sorted({e.actor for e in log.entries if e.target == target})


def summarize(log: AuditLog) -> dict[str, Any]:
    """Counts by actor, action, and outcome, plus the chain's state."""
    by_actor: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for entry in log.entries:
        by_actor[entry.actor] = by_actor.get(entry.actor, 0) + 1
        by_action[entry.action] = by_action.get(entry.action, 0) + 1
        by_outcome[entry.outcome.value] = by_outcome.get(entry.outcome.value, 0) + 1

    breaks = verify(log)
    return {
        "entries": len(log.entries),
        "head": log.head,
        "intact": not breaks,
        "first_break": breaks[0].index if breaks else None,
        "actors": by_actor,
        "actions": by_action,
        "outcomes": by_outcome,
        "denied": by_outcome.get(AuditOutcome.DENIED.value, 0),
    }
