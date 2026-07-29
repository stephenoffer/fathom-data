"""Who may see what, and what they may do with it.

Without this, anyone who can run the CLI can read every profile, every label, and
every erasure proof — including the subject digests. In a regulated environment that
is disqualifying on its own, and it is the first question a security review asks.

Three decisions worth stating, because each has a common alternative that is worse:

**Deny by default, and governance artefacts deny harder.** An unmatched request is
refused, and erasure proofs and subject digests require an explicit grant that a
broad `read` on the dataset does not confer. A proof names a person; the whole point
of hashing the subject was that reading it should be a deliberate act.

**Grants are scoped by selector, not by enumeration.** `+gold.monthly+` survives new
tables appearing; a list of names does not, and the failure mode of a stale list is
that someone quietly loses access to data they own.

**A decision explains itself.** `explain` returns the rule that matched, because
"permission denied" with no reason produces a ticket, and the ticket produces a
broader grant than anyone intended.

This is an authorization model, not an authentication one. Establishing *who* the
principal is belongs to whatever already does that — an identity provider, a
service mesh, a signed request. This decides what they may do once you know.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ...core.types import DatasetId

__all__ = [
    "Action",
    "Decision",
    "Effect",
    "Grant",
    "Principal",
    "Role",
    "Sensitivity",
    "AccessPolicy",
    "columns_visible_to",
    "can",
    "explain",
    "grant_for",
    "is_governance_action",
    "mask_profile",
    "merge_roles",
    "principals_who_can",
    "redact",
    "require",
    "role_with",
    "sensitivity_of",
]


class Action(StrEnum):
    """What a principal is trying to do.

    Deliberately finer than read/write. The distinction that matters is between
    reading a dataset's shape and reading the governance artefacts about it, because
    the second names people.
    """

    READ_GRAPH = "read:graph"
    READ_PROFILE = "read:profile"
    READ_LABEL = "read:label"
    READ_PROOF = "read:proof"  # erasure proofs; names a subject, even hashed
    READ_SUBJECT = "read:subject"  # the digest itself
    READ_AUDIT = "read:audit"
    PLAN = "plan"
    APPLY = "apply"  # actually rebuild
    LABEL = "label"
    CONFIRM_LABEL = "confirm:label"  # a human decision that outranks inference
    ERASE = "erase"
    ADMIN = "admin"

    @property
    def is_write(self) -> bool:
        return self in {
            Action.APPLY,
            Action.LABEL,
            Action.CONFIRM_LABEL,
            Action.ERASE,
            Action.ADMIN,
        }


# Actions that a broad dataset grant must never imply. Each of these either names a
# person or changes the world, and both deserve to be asked for by name.
GOVERNANCE_ACTIONS = frozenset(
    {
        Action.READ_PROOF,
        Action.READ_SUBJECT,
        Action.READ_AUDIT,
        Action.ERASE,
        Action.CONFIRM_LABEL,
        Action.ADMIN,
    }
)


def is_governance_action(action: Action) -> bool:
    return action in GOVERNANCE_ACTIONS


class Effect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class Sensitivity(StrEnum):
    """How much a dataset's contents matter. Ordered."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        return list(Sensitivity).index(self)


@dataclass(frozen=True)
class Grant:
    """Permission to take an action over a set of datasets.

    `selector` is a glob over the dataset's string form. A `deny` grant beats every
    `allow`, so an exception can be carved out of a broad permission without having
    to enumerate everything the broad permission covers.
    """

    actions: frozenset[Action]
    selector: str = "*"
    effect: Effect = Effect.ALLOW
    max_sensitivity: Sensitivity | None = None
    columns: frozenset[str] = frozenset()  # empty means every column
    reason: str = ""

    def matches(self, dataset: DatasetId, action: Action) -> bool:
        return action in self.actions and fnmatch.fnmatch(str(dataset), self.selector)


def grant_for(
    *actions: Action,
    selector: str = "*",
    effect: Effect = Effect.ALLOW,
    max_sensitivity: Sensitivity | None = None,
    columns: Iterable[str] = (),
    reason: str = "",
) -> Grant:
    return Grant(
        actions=frozenset(actions),
        selector=selector,
        effect=effect,
        max_sensitivity=max_sensitivity,
        columns=frozenset(columns),
        reason=reason,
    )


@dataclass(frozen=True)
class Role:
    """A named bundle of grants."""

    name: str
    grants: tuple[Grant, ...] = ()
    inherits: tuple[str, ...] = ()
    description: str = ""


def role_with(
    name: str, *grants: Grant, inherits: Iterable[str] = (), description: str = ""
) -> Role:
    return Role(name=name, grants=tuple(grants), inherits=tuple(inherits), description=description)


@dataclass(frozen=True)
class Principal:
    """Whoever is asking. Authentication established this; we only authorize."""

    identifier: str
    roles: frozenset[str] = frozenset()
    tenant: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """The answer, and the rule that produced it."""

    allowed: bool
    action: Action
    dataset: DatasetId | None = None
    principal: str = ""
    rule: str = ""
    reason: str = ""

    def summary(self) -> str:
        verdict = "allowed" if self.allowed else "denied"
        target = f" on {self.dataset}" if self.dataset else ""
        detail = f" ({self.reason})" if self.reason else ""
        return f"{self.principal}: {self.action.value}{target} {verdict} by {self.rule}{detail}"


@dataclass
class AccessPolicy:
    """Roles, sensitivity labels, and the evaluation order between them."""

    roles: dict[str, Role] = field(default_factory=dict)
    sensitivity: dict[DatasetId, Sensitivity] = field(default_factory=dict)
    default_sensitivity: Sensitivity = Sensitivity.INTERNAL

    def add_role(self, role: Role) -> None:
        self.roles[role.name] = role

    def classify(self, dataset: DatasetId, level: Sensitivity) -> None:
        self.sensitivity[dataset] = level

    def sensitivity_of(self, dataset: DatasetId) -> Sensitivity:
        return self.sensitivity.get(dataset, self.default_sensitivity)

    def grants_for(self, principal: Principal) -> list[tuple[str, Grant]]:
        """Every grant a principal holds, with the role that carried it.

        Inheritance is resolved depth-first and cycle-safe, because a role graph
        with a loop is a config people write and an infinite recursion is a poor way
        to find out.
        """
        found: list[tuple[str, Grant]] = []
        seen: set[str] = set()

        def walk(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            role = self.roles.get(name)
            if role is None:
                return
            for grant in role.grants:
                found.append((name, grant))
            for parent in role.inherits:
                walk(parent)

        for name in sorted(principal.roles):
            walk(name)
        return found


def sensitivity_of(policy: AccessPolicy, dataset: DatasetId) -> Sensitivity:
    return policy.sensitivity_of(dataset)


def explain(
    policy: AccessPolicy, principal: Principal, action: Action, dataset: DatasetId | None = None
) -> Decision:
    """Decide, and say which rule decided.

    Order: explicit deny, then sensitivity ceiling, then allow, then the default
    deny. Deny-first is what makes an exception expressible without enumerating the
    rule it is an exception to.
    """
    target = dataset or DatasetId("*", "*")

    matching = [
        (role, grant)
        for role, grant in policy.grants_for(principal)
        if grant.matches(target, action)
    ]

    for role, grant in matching:
        if grant.effect is Effect.DENY:
            return Decision(
                allowed=False,
                action=action,
                dataset=dataset,
                principal=principal.identifier,
                rule=f"{role}:deny",
                reason=grant.reason or "explicitly denied",
            )

    level = policy.sensitivity_of(target) if dataset else policy.default_sensitivity
    for role, grant in matching:
        if grant.max_sensitivity is not None and level.rank > grant.max_sensitivity.rank:
            return Decision(
                allowed=False,
                action=action,
                dataset=dataset,
                principal=principal.identifier,
                rule=f"{role}:sensitivity",
                reason=(
                    f"dataset is {level.value}, grant permits up to {grant.max_sensitivity.value}"
                ),
            )

    for role, grant in matching:
        if grant.effect is Effect.ALLOW:
            return Decision(
                allowed=True,
                action=action,
                dataset=dataset,
                principal=principal.identifier,
                rule=f"{role}:allow",
                reason=grant.reason,
            )

    return Decision(
        allowed=False,
        action=action,
        dataset=dataset,
        principal=principal.identifier,
        rule="default",
        reason=(
            "no grant matched. Governance actions need an explicit grant; a broad "
            "read does not confer them."
            if is_governance_action(action)
            else "no grant matched"
        ),
    )


def can(
    policy: AccessPolicy, principal: Principal, action: Action, dataset: DatasetId | None = None
) -> bool:
    return explain(policy, principal, action, dataset).allowed


class AccessDenied(PermissionError):
    """Raised by `require`. Carries the decision so a caller can log the reason."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(decision.summary())


def require(
    policy: AccessPolicy, principal: Principal, action: Action, dataset: DatasetId | None = None
) -> Decision:
    """Authorize or raise. The raising form, for call sites that cannot continue."""
    decision = explain(policy, principal, action, dataset)
    if not decision.allowed:
        raise AccessDenied(decision)
    return decision


def columns_visible_to(
    policy: AccessPolicy, principal: Principal, dataset: DatasetId, columns: Iterable[str]
) -> list[str]:
    """Column-level filtering.

    A grant with no column restriction sees everything; one with a restriction sees
    only what it names. Several grants union, because holding two roles should not
    see less than holding either.
    """
    every = list(columns)
    matching = [
        grant
        for _, grant in policy.grants_for(principal)
        if grant.effect is Effect.ALLOW and grant.matches(dataset, Action.READ_PROFILE)
    ]
    if not matching:
        return []
    if any(not grant.columns for grant in matching):
        return every
    permitted: set[str] = set()
    for grant in matching:
        permitted.update(grant.columns)
    return [c for c in every if c in permitted]


def redact(value: str, *, keep: int = 0) -> str:
    """Mask a value, optionally keeping a prefix for correlation."""
    if keep <= 0:
        return "***"
    return value[:keep] + "***" if len(value) > keep else "***"


def mask_profile(
    policy: AccessPolicy,
    principal: Principal,
    dataset: DatasetId,
    profile: Mapping[str, object],
    *,
    column_key: str = "columns",
) -> dict[str, object]:
    """Return a profile with columns the principal cannot see removed.

    Removed rather than masked. A masked column still discloses that it exists and
    what its null rate is, which is more than a denial should leak.
    """
    out = dict(profile)
    raw = out.get(column_key)
    if not isinstance(raw, (list, tuple)):
        return out

    names = [str(c.get("name", "")) if isinstance(c, Mapping) else str(c) for c in raw]
    visible = set(columns_visible_to(policy, principal, dataset, names))
    out[column_key] = [c for c, name in zip(raw, names, strict=True) if name in visible]
    return out


def merge_roles(*roles: Role) -> Role:
    """Combine roles into one. Used to flatten an inheritance chain for export."""
    return Role(
        name="+".join(r.name for r in roles),
        grants=tuple(g for r in roles for g in r.grants),
        description="merged",
    )


def principals_who_can(
    policy: AccessPolicy,
    principals: Sequence[Principal],
    action: Action,
    dataset: DatasetId | None = None,
) -> list[str]:
    """Reverse the question: who can do this?

    The audit question. Asked after an incident, and asked by every access review.
    """
    return sorted(p.identifier for p in principals if can(policy, p, action, dataset))
