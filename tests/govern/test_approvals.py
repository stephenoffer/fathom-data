"""Change approval.

Risk is derived from what a change does to the graph, not from who proposed it or
how large the diff is. The tests that matter check that a spec change is treated as
breaking, that a contracted dataset escalates, and that nobody approves their own
work.
"""

from __future__ import annotations

import pytest

from fathom.govern.approvals import (
    ApprovalRule,
    ChangeKind,
    Proposal,
    RiskClass,
    approve,
    classify,
    decide,
    default_rules,
    outstanding,
    reject,
    required_approvals,
    submit,
)


def proposal(kind: ChangeKind = ChangeKind.CHANGE_SPEC, **overrides) -> Proposal:
    base = {
        "identifier": "P-1",
        "kind": kind,
        "target": "gold.monthly",
        "proposer": "alice",
    }
    base.update(overrides)
    return Proposal(**base)


# -- risk ----------------------------------------------------------------------


def test_a_spec_change_is_breaking():
    """It does not fail anything. It changes what every future rebuild covers."""
    assert classify(proposal(ChangeKind.CHANGE_SPEC)) is RiskClass.BREAKING


def test_widening_a_mapping_is_breaking():
    assert classify(proposal(ChangeKind.WIDEN_MAPPING)) is RiskClass.BREAKING


def test_adding_a_dataset_is_routine():
    """Nothing that already worked can break."""
    assert classify(proposal(ChangeKind.ADD_DATASET)) is RiskClass.ROUTINE


def test_relaxing_a_policy_is_governed_regardless_of_consumers():
    assert classify(proposal(ChangeKind.RELAX_POLICY)) is RiskClass.GOVERNED


def test_a_contracted_target_escalates_a_breaking_change():
    """A breaking change to something another team depends on is not the proposer's
    call alone."""
    assert classify(proposal(ChangeKind.CHANGE_SPEC, contracted=True)) is RiskClass.GOVERNED


def test_consumers_escalate_a_breaking_change():
    assert classify(proposal(ChangeKind.CHANGE_SPEC, consumers=("finance",))) is RiskClass.GOVERNED


def test_consumers_do_not_escalate_a_routine_change():
    """Otherwise every addition to a popular table needs a committee."""
    assert classify(proposal(ChangeKind.ADD_DATASET, consumers=("finance",))) is RiskClass.ROUTINE


def test_risk_classes_are_ordered():
    assert RiskClass.GOVERNED.rank > RiskClass.BREAKING.rank > RiskClass.ROUTINE.rank


# -- decisions -----------------------------------------------------------------


def test_a_routine_change_needs_nobody():
    assert decide(submit(proposal(ChangeKind.ADD_DATASET))).approved


def test_a_breaking_change_needs_a_data_owner():
    submission = submit(proposal(ChangeKind.CHANGE_SPEC))
    approve(submission, "bob", team="platform")
    verdict = decide(submission)
    assert not verdict.approved
    assert any("data-owner" in m for m in verdict.missing)


def test_a_data_owner_approval_unblocks_a_breaking_change():
    submission = submit(proposal(ChangeKind.CHANGE_SPEC))
    approve(submission, "bob", roles=["data-owner"])
    assert decide(submission).approved


def test_a_governed_change_needs_the_consuming_team_specifically():
    """A reviewer from the producing team cannot consent on behalf of the team that
    will be broken."""
    submission = submit(proposal(ChangeKind.CHANGE_SPEC, contracted=True, consumers=("finance",)))
    approve(submission, "bob", roles=["data-owner"], team="platform")
    approve(submission, "dave", roles=["data-owner"], team="platform")

    verdict = decide(submission)
    assert not verdict.approved
    assert any("consuming team" in m for m in verdict.missing)


def test_a_consumer_approval_completes_a_governed_change():
    submission = submit(proposal(ChangeKind.CHANGE_SPEC, contracted=True, consumers=("finance",)))
    approve(submission, "bob", roles=["data-owner"], team="platform")
    approve(submission, "carol", team="finance")
    assert decide(submission).approved


def test_one_rejection_blocks_regardless_of_approvals():
    submission = submit(proposal(ChangeKind.CHANGE_SPEC))
    approve(submission, "bob", roles=["data-owner"])
    reject(submission, "carol", comment="breaks the close")

    verdict = decide(submission)
    assert not verdict.approved
    assert verdict.rejections == ("carol",)


def test_self_approval_is_refused_rather_than_ignored():
    """Silently dropping it would let a proposer believe their change was signed off,
    and every gate that allows it degrades to a comment box."""
    submission = submit(proposal())
    with pytest.raises(ValueError, match="cannot approve it"):
        approve(submission, "alice")


def test_a_verdict_explains_what_is_missing():
    summary = decide(submit(proposal(ChangeKind.CHANGE_SPEC))).summary()
    assert "blocked" in summary
    assert "needs" in summary


# -- rules ---------------------------------------------------------------------


def test_the_default_rules_cover_every_risk_class():
    covered = {rule.risk for rule in default_rules()}
    assert covered == set(RiskClass)


def test_rules_can_be_overridden():
    strict = [ApprovalRule(RiskClass.ROUTINE, approvals=1)]
    submission = submit(proposal(ChangeKind.ADD_DATASET))
    assert not decide(submission, strict).approved


def test_required_approvals_reports_the_matching_rule():
    rule = required_approvals(proposal(ChangeKind.CHANGE_SPEC, contracted=True))
    assert rule.risk is RiskClass.GOVERNED
    assert rule.require_consumer


def test_outstanding_sorts_worst_risk_first():
    blocked_governed = submit(proposal(ChangeKind.RELAX_POLICY, identifier="P-gov"))
    blocked_breaking = submit(proposal(ChangeKind.CHANGE_SPEC, identifier="P-break"))
    approved = submit(proposal(ChangeKind.ADD_DATASET, identifier="P-ok"))

    pending = outstanding([blocked_breaking, approved, blocked_governed])
    assert [v.proposal for v in pending] == ["P-gov", "P-break"]


def test_submission_stamps_a_time():
    assert submit(proposal()).proposal.submitted is not None
