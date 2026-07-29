"""Gating the changes that silently alter what every future rebuild covers.

Editing a partition spec from `day` to `month` does not fail anything. It changes
what a rebuild covers, forever, and the only symptom is that numbers stop matching
in a report nobody reconciles for a quarter. Same for widening a mapping, removing a
column-lineage edge, or relaxing a sink policy.

So the risk class here is derived from *what the change does to the graph*, not from
who made it or how large the diff is. `classify` reads the proposed change and
answers how bad it could be:

- **`ROUTINE`** — adding a dataset, tightening a policy. Nothing that already worked
  can break.
- **`SIGNIFICANT`** — a new edge, a spec on something previously unpartitioned.
- **`BREAKING`** — a spec change, a widened mapping, a removed edge. Something that
  was correct may now be wrong.
- **`GOVERNED`** — touching a contracted dataset, a policy, or anything with a
  consumer. Needs the consumer's approval, not just a reviewer's.

The rule that makes this worth having: **a proposer cannot approve their own
change.** Every gate that omits it degrades to a comment box within a month.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

__all__ = [
    "Approval",
    "ApprovalRule",
    "ChangeKind",
    "Proposal",
    "RiskClass",
    "Verdict",
    "approve",
    "classify",
    "decide",
    "default_rules",
    "outstanding",
    "reject",
    "required_approvals",
    "submit",
]


class ChangeKind(StrEnum):
    """What is being changed. The vocabulary risk is derived from."""

    ADD_DATASET = "add_dataset"
    REMOVE_DATASET = "remove_dataset"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    CHANGE_SPEC = "change_spec"  # the one that silently changes rebuild coverage
    WIDEN_MAPPING = "widen_mapping"
    NARROW_MAPPING = "narrow_mapping"
    ADD_POLICY = "add_policy"
    RELAX_POLICY = "relax_policy"
    REMOVE_POLICY = "remove_policy"
    CHANGE_CONTRACT = "change_contract"
    CONFIRM_LABEL = "confirm_label"


class RiskClass(StrEnum):
    """How bad a change could be. Ordered."""

    ROUTINE = "routine"
    SIGNIFICANT = "significant"
    BREAKING = "breaking"
    GOVERNED = "governed"

    @property
    def rank(self) -> int:
        return list(RiskClass).index(self)


# What each kind of change can do if it is wrong. Kept as data so the reasoning is
# inspectable rather than buried in branches.
_BASE_RISK: dict[ChangeKind, RiskClass] = {
    ChangeKind.ADD_DATASET: RiskClass.ROUTINE,
    ChangeKind.ADD_POLICY: RiskClass.ROUTINE,
    ChangeKind.NARROW_MAPPING: RiskClass.SIGNIFICANT,
    ChangeKind.ADD_EDGE: RiskClass.SIGNIFICANT,
    ChangeKind.CONFIRM_LABEL: RiskClass.SIGNIFICANT,
    ChangeKind.REMOVE_DATASET: RiskClass.BREAKING,
    ChangeKind.REMOVE_EDGE: RiskClass.BREAKING,
    ChangeKind.CHANGE_SPEC: RiskClass.BREAKING,
    ChangeKind.WIDEN_MAPPING: RiskClass.BREAKING,
    ChangeKind.RELAX_POLICY: RiskClass.GOVERNED,
    ChangeKind.REMOVE_POLICY: RiskClass.GOVERNED,
    ChangeKind.CHANGE_CONTRACT: RiskClass.GOVERNED,
}


@dataclass(frozen=True)
class Proposal:
    """A change somebody wants to make."""

    identifier: str
    kind: ChangeKind
    target: str
    proposer: str
    summary: str = ""
    before: str = ""
    after: str = ""
    consumers: tuple[str, ...] = ()  # teams downstream of the target
    contracted: bool = False
    submitted: datetime | None = None

    @property
    def is_spec_change(self) -> bool:
        return self.kind is ChangeKind.CHANGE_SPEC


def classify(proposal: Proposal) -> RiskClass:
    """How bad this change could be, from what it does rather than who proposed it.

    Escalates to `GOVERNED` when the target is contracted or has consumers, because
    a breaking change to something another team depends on is not the proposer's
    call to make alone.
    """
    base = _BASE_RISK.get(proposal.kind, RiskClass.SIGNIFICANT)
    depended_on = proposal.contracted or bool(proposal.consumers)
    if depended_on and base.rank >= RiskClass.BREAKING.rank:
        return RiskClass.GOVERNED
    return base


@dataclass(frozen=True)
class ApprovalRule:
    """How many approvals a risk class needs, and from whom."""

    risk: RiskClass
    approvals: int = 1
    from_roles: frozenset[str] = frozenset()
    require_consumer: bool = False
    reason: str = ""


def default_rules() -> list[ApprovalRule]:
    """A defensible starting point.

    `GOVERNED` requires a consumer's approval specifically. A reviewer from the
    producing team cannot consent on behalf of the team that will be broken.
    """
    return [
        ApprovalRule(RiskClass.ROUTINE, approvals=0, reason="nothing that worked can break"),
        ApprovalRule(RiskClass.SIGNIFICANT, approvals=1),
        ApprovalRule(
            RiskClass.BREAKING,
            approvals=1,
            from_roles=frozenset({"data-owner"}),
            reason="something correct may now be wrong",
        ),
        ApprovalRule(
            RiskClass.GOVERNED,
            approvals=2,
            from_roles=frozenset({"data-owner"}),
            require_consumer=True,
            reason="a downstream team is affected and has to agree",
        ),
    ]


@dataclass(frozen=True)
class Approval:
    """One person's sign-off."""

    approver: str
    roles: frozenset[str] = frozenset()
    team: str = ""
    at: datetime | None = None
    comment: str = ""
    rejected: bool = False


@dataclass(frozen=True)
class Verdict:
    """Whether a proposal may proceed, and what is missing if not."""

    proposal: str
    risk: RiskClass
    approved: bool
    missing: tuple[str, ...] = ()
    rejections: tuple[str, ...] = ()

    def summary(self) -> str:
        head = f"{self.proposal} [{self.risk.value}]: {'approved' if self.approved else 'blocked'}"
        lines = [head]
        lines.extend(f"  rejected by {who}" for who in self.rejections)
        lines.extend(f"  needs {what}" for what in self.missing)
        return "\n".join(lines)


@dataclass
class Submission:
    """A proposal and the approvals gathered against it."""

    proposal: Proposal
    approvals: list[Approval] = field(default_factory=list)


def submit(proposal: Proposal, *, at: datetime | None = None) -> Submission:
    stamped = Proposal(
        identifier=proposal.identifier,
        kind=proposal.kind,
        target=proposal.target,
        proposer=proposal.proposer,
        summary=proposal.summary,
        before=proposal.before,
        after=proposal.after,
        consumers=proposal.consumers,
        contracted=proposal.contracted,
        submitted=at or datetime.now(UTC),
    )
    return Submission(proposal=stamped)


def approve(
    submission: Submission,
    approver: str,
    *,
    roles: Iterable[str] = (),
    team: str = "",
    comment: str = "",
    at: datetime | None = None,
) -> Submission:
    """Record an approval.

    Self-approval is refused rather than ignored. Silently dropping it would let a
    proposer believe their change was signed off.
    """
    if approver == submission.proposal.proposer:
        raise ValueError(
            f"{approver} proposed this change and cannot approve it. Every gate that "
            "allows self-approval degrades to a comment box."
        )
    submission.approvals.append(
        Approval(
            approver=approver,
            roles=frozenset(roles),
            team=team,
            at=at or datetime.now(UTC),
            comment=comment,
        )
    )
    return submission


def reject(
    submission: Submission, approver: str, *, comment: str = "", at: datetime | None = None
) -> Submission:
    """Record a rejection. One rejection blocks regardless of how many approvals exist."""
    submission.approvals.append(
        Approval(approver=approver, at=at or datetime.now(UTC), comment=comment, rejected=True)
    )
    return submission


def required_approvals(
    proposal: Proposal, rules: Sequence[ApprovalRule] | None = None
) -> ApprovalRule:
    risk = classify(proposal)
    for rule in rules or default_rules():
        if rule.risk is risk:
            return rule
    return ApprovalRule(risk, approvals=1)


def decide(submission: Submission, rules: Sequence[ApprovalRule] | None = None) -> Verdict:
    """Whether the proposal may proceed."""
    proposal = submission.proposal
    risk = classify(proposal)
    rule = required_approvals(proposal, rules)

    rejections = tuple(a.approver for a in submission.approvals if a.rejected)
    if rejections:
        return Verdict(proposal.identifier, risk, approved=False, rejections=rejections)

    granted = [a for a in submission.approvals if not a.rejected]
    missing: list[str] = []

    if len(granted) < rule.approvals:
        missing.append(f"{rule.approvals - len(granted)} more approval(s)")

    if rule.from_roles and not any(a.roles & rule.from_roles for a in granted):
        missing.append(f"an approval from one of {sorted(rule.from_roles)}")

    if rule.require_consumer:
        consumer_teams = set(proposal.consumers)
        if consumer_teams and not any(a.team in consumer_teams for a in granted):
            missing.append(
                f"an approval from a consuming team ({sorted(consumer_teams)}); a "
                "reviewer from the producing team cannot consent on their behalf"
            )

    return Verdict(proposal.identifier, risk, approved=not missing, missing=tuple(missing))


def outstanding(
    submissions: Iterable[Submission], rules: Sequence[ApprovalRule] | None = None
) -> list[Verdict]:
    """Every proposal still blocked, worst risk first."""
    verdicts = [decide(s, rules) for s in submissions]
    return sorted((v for v in verdicts if not v.approved), key=lambda v: -v.risk.rank)
