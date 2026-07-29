"""Key registry and crypto-shredding, so erasure has an answer on immutable storage.

`govern/erasure.py` names crypto-shredding as the only reliable option where objects
are versioned or write-once — and then stops, because there was no key registry to
shred against. This is that registry.

The idea is old and the failure modes are specific. Encrypt each subject's rows
under a key that exists only in a registry; destroy the key and the ciphertext is
unrecoverable without deleting a byte of it. That works exactly as well as the
key hygiene around it, so:

**A key is destroyed once, and the record survives it.** `destroy` removes the
material and keeps the metadata, because a registry that forgets a key ever existed
cannot prove it was destroyed — and proving it is the entire purpose.

**Shared keys are refused, not warned about.** A key covering two subjects cannot be
destroyed for one of them. `destroy_for_subject` raises rather than shredding both,
which is the difference between an erasure and an outage.

**A destroyed key stays destroyed.** Re-registering the same identifier is refused,
because the alternative is silently resurrecting an erasure obligation someone has
already certified as discharged.

There is no cryptography here. This tracks which key covers what and whether it
still exists; encrypting and decrypting belongs to a KMS, and the `provider` field
is where you record which one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "Key",
    "KeyRegistry",
    "KeyState",
    "ShredProof",
    "ShredRefused",
    "covered_by",
    "destroy",
    "destroy_for_subject",
    "keys_for_subject",
    "register",
    "rotate",
    "shred_proof",
    "subjects_covered",
    "verify_destroyed",
]


class KeyState(StrEnum):
    ACTIVE = "active"
    ROTATED = "rotated"  # superseded, retained to decrypt old ciphertext
    DESTROYED = "destroyed"  # material gone; the record is not

    @property
    def can_decrypt(self) -> bool:
        return self is not KeyState.DESTROYED


@dataclass(frozen=True)
class Key:
    """One key's metadata. Never the material itself."""

    identifier: str
    state: KeyState = KeyState.ACTIVE
    subjects: frozenset[str] = frozenset()
    provider: str = ""
    algorithm: str = ""
    created: datetime | None = None
    destroyed: datetime | None = None
    supersedes: str = ""
    reason: str = ""

    @property
    def is_shared(self) -> bool:
        """Covering more than one subject makes per-subject shredding impossible."""
        return len(self.subjects) > 1


class ShredRefused(RuntimeError):
    """Raised when destroying a key would erase more than was asked for."""


@dataclass
class KeyRegistry:
    """Which key covers what, and whether it still exists."""

    keys: dict[str, Key] = field(default_factory=dict)
    salt: str = ""

    def subject_digest(self, subject: str) -> str:
        """The handle a proof carries. Never the subject itself."""
        payload = f"{self.salt}\x1f{subject}".encode()
        return hashlib.sha256(payload).hexdigest()

    def get(self, identifier: str) -> Key | None:
        return self.keys.get(identifier)


def register(
    registry: KeyRegistry,
    identifier: str,
    *,
    subjects: Iterable[str] = (),
    provider: str = "",
    algorithm: str = "",
    at: datetime | None = None,
) -> Key:
    """Add a key.

    Re-registering a destroyed identifier is refused. The alternative silently
    resurrects an erasure obligation somebody has already certified as discharged.
    """
    existing = registry.keys.get(identifier)
    if existing is not None and existing.state is KeyState.DESTROYED:
        raise ShredRefused(
            f"key {identifier!r} was destroyed at {existing.destroyed}; re-registering "
            "it would resurrect an erasure that has already been certified"
        )
    key = Key(
        identifier=identifier,
        subjects=frozenset(subjects),
        provider=provider,
        algorithm=algorithm,
        created=at or datetime.now(UTC),
    )
    registry.keys[identifier] = key
    return key


def rotate(
    registry: KeyRegistry, identifier: str, successor: str, *, at: datetime | None = None
) -> Key:
    """Supersede a key, retaining the old one so existing ciphertext stays readable.

    Rotation is not destruction. A rotated key that is discarded takes its ciphertext
    with it, which is an outage rather than an erasure.
    """
    old = registry.keys.get(identifier)
    if old is None:
        raise KeyError(f"no key {identifier!r} to rotate")
    if old.state is KeyState.DESTROYED:
        raise ShredRefused(f"key {identifier!r} is destroyed and cannot be rotated")

    registry.keys[identifier] = Key(
        identifier=old.identifier,
        state=KeyState.ROTATED,
        subjects=old.subjects,
        provider=old.provider,
        algorithm=old.algorithm,
        created=old.created,
        reason=f"superseded by {successor}",
    )
    return register(
        registry,
        successor,
        subjects=old.subjects,
        provider=old.provider,
        algorithm=old.algorithm,
        at=at,
    )


def destroy(
    registry: KeyRegistry, identifier: str, *, reason: str = "", at: datetime | None = None
) -> Key:
    """Destroy a key's material, keeping its record.

    The record is what proves the destruction happened. A registry that forgets the
    key existed cannot demonstrate anything.
    """
    key = registry.keys.get(identifier)
    if key is None:
        raise KeyError(f"no key {identifier!r} to destroy")
    if key.state is KeyState.DESTROYED:
        return key  # idempotent; destroying twice is not an error

    destroyed = Key(
        identifier=key.identifier,
        state=KeyState.DESTROYED,
        subjects=key.subjects,
        provider=key.provider,
        algorithm=key.algorithm,
        created=key.created,
        destroyed=at or datetime.now(UTC),
        supersedes=key.supersedes,
        reason=reason,
    )
    registry.keys[identifier] = destroyed
    return destroyed


def keys_for_subject(registry: KeyRegistry, subject: str) -> list[Key]:
    return [k for k in registry.keys.values() if subject in k.subjects]


def subjects_covered(registry: KeyRegistry, identifier: str) -> frozenset[str]:
    key = registry.keys.get(identifier)
    return key.subjects if key else frozenset()


def covered_by(registry: KeyRegistry, subject: str) -> list[str]:
    """Key identifiers still able to decrypt this subject's data."""
    return sorted(k.identifier for k in keys_for_subject(registry, subject) if k.state.can_decrypt)


def destroy_for_subject(
    registry: KeyRegistry, subject: str, *, reason: str = "", at: datetime | None = None
) -> list[Key]:
    """Shred every key covering one subject.

    Refuses when a key also covers someone else. Destroying it anyway would erase a
    person who did not ask to be erased, which is an outage dressed as compliance.
    """
    candidates = [k for k in keys_for_subject(registry, subject) if k.state.can_decrypt]
    shared = [k for k in candidates if k.is_shared]
    if shared:
        names = ", ".join(sorted(k.identifier for k in shared))
        raise ShredRefused(
            f"key(s) {names} also cover other subjects; destroying them would erase "
            "people who did not request it. Re-key those subjects first."
        )
    return [destroy(registry, k.identifier, reason=reason, at=at) for k in candidates]


def verify_destroyed(registry: KeyRegistry, subject: str) -> bool:
    """True when no surviving key can decrypt this subject's data."""
    return not covered_by(registry, subject)


@dataclass(frozen=True)
class ShredProof:
    """Evidence that a subject's keys were destroyed.

    Composes with the erasure proof in `govern/erasure.py`: that one says the rows
    were rewritten where they could be, this one says the remainder was made
    unreadable.
    """

    subject_digest: str
    reference: str
    generated: datetime
    keys: tuple[Mapping[str, Any], ...]
    complete: bool

    def to_json(self) -> str:
        body = {
            "subject_digest": self.subject_digest,
            "reference": self.reference,
            "generated": self.generated.isoformat(),
            "complete": self.complete,
            "keys": [dict(k) for k in self.keys],
        }
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return json.dumps({**body, "digest": digest}, sort_keys=True, indent=2)

    def summary(self) -> str:
        state = "complete" if self.complete else "INCOMPLETE"
        return (
            f"shred proof for {self.subject_digest[:12]}… ({self.reference}): "
            f"{len(self.keys)} key(s), {state}"
        )


def shred_proof(
    registry: KeyRegistry, subject: str, *, reference: str = "", at: datetime | None = None
) -> ShredProof:
    """Build the artefact. `complete` is false while any key can still decrypt."""
    involved = keys_for_subject(registry, subject)
    return ShredProof(
        subject_digest=registry.subject_digest(subject),
        reference=reference,
        generated=at or datetime.now(UTC),
        keys=tuple(
            {
                "identifier": k.identifier,
                "state": k.state.value,
                "provider": k.provider,
                "destroyed": k.destroyed.isoformat() if k.destroyed else None,
            }
            for k in sorted(involved, key=lambda k: k.identifier)
        ),
        complete=verify_destroyed(registry, subject),
    )
