"""Authorization, audit, tenancy, and key destruction.

Most of these test a refusal. Access control is only worth having if the default is
no, if a broad grant does not quietly confer a narrow privilege, and if destroying a
shared key fails instead of erasing someone who did not ask.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.types import DatasetId
from fathom.govern.access import (
    AccessDenied,
    AccessPolicy,
    Action,
    AuditEntry,
    AuditLog,
    AuditOutcome,
    CrossingKind,
    Effect,
    KeyRegistry,
    KeyState,
    Principal,
    Sensitivity,
    ShredRefused,
    Tenant,
    TenantMap,
    can,
    columns_visible_to,
    covered_by,
    destroy,
    destroy_for_subject,
    entries_for,
    entries_since,
    explain,
    grant_for,
    is_governance_action,
    keys_for_subject,
    leaks,
    mask_profile,
    principals_who_can,
    record,
    redact,
    register,
    replay,
    require,
    role_with,
    rotate,
    scope_graph,
    share,
    shred_proof,
    summarize,
    tenant_summary,
    unowned,
    verify,
    verify_destroyed,
    who_touched,
)

GOLD = DatasetId("duckdb", "gold.monthly")
USERS = DatasetId("duckdb", "raw.users")


def policy_with_roles() -> AccessPolicy:
    policy = AccessPolicy()
    policy.add_role(
        role_with(
            "analyst",
            grant_for(
                Action.READ_GRAPH,
                Action.READ_PROFILE,
                Action.PLAN,
                max_sensitivity=Sensitivity.CONFIDENTIAL,
            ),
        )
    )
    policy.add_role(
        role_with("dpo", grant_for(Action.READ_PROOF, Action.ERASE), inherits=["analyst"])
    )
    policy.classify(USERS, Sensitivity.RESTRICTED)
    return policy


ANALYST = Principal("alice", frozenset({"analyst"}))
DPO = Principal("bob", frozenset({"dpo"}))


# -- authorization -------------------------------------------------------------


def test_an_unmatched_request_is_denied():
    """Deny by default. An allow-by-default access model is not one."""
    nobody = Principal("mallory", frozenset())
    assert not can(policy_with_roles(), nobody, Action.READ_GRAPH, GOLD)


def test_a_matching_grant_allows():
    assert can(policy_with_roles(), ANALYST, Action.READ_PROFILE, GOLD)


def test_a_broad_read_does_not_confer_reading_a_proof():
    """A proof names a person; the point of hashing the subject was that reading it
    should be a deliberate act."""
    decision = explain(policy_with_roles(), ANALYST, Action.READ_PROOF, GOLD)
    assert not decision.allowed
    assert "explicit grant" in decision.reason


def test_governance_actions_are_enumerated():
    assert is_governance_action(Action.ERASE)
    assert is_governance_action(Action.READ_SUBJECT)
    assert not is_governance_action(Action.READ_PROFILE)


def test_a_sensitivity_ceiling_blocks_a_matching_grant():
    decision = explain(policy_with_roles(), ANALYST, Action.READ_PROFILE, USERS)
    assert not decision.allowed
    assert "restricted" in decision.reason


def test_a_decision_names_the_rule_that_made_it():
    """ "Permission denied" with no reason produces a ticket, and the ticket produces
    a broader grant than anyone intended."""
    assert explain(policy_with_roles(), ANALYST, Action.READ_PROFILE, GOLD).rule == "analyst:allow"


def test_deny_beats_allow_regardless_of_order():
    """So an exception is expressible without enumerating what it excepts."""
    policy = AccessPolicy()
    policy.add_role(
        role_with(
            "mixed",
            grant_for(Action.READ_PROFILE, selector="*"),
            grant_for(Action.READ_PROFILE, selector="*users*", effect=Effect.DENY, reason="pii"),
        )
    )
    principal = Principal("x", frozenset({"mixed"}))
    assert can(policy, principal, Action.READ_PROFILE, GOLD)
    assert not can(policy, principal, Action.READ_PROFILE, USERS)


def test_roles_inherit():
    assert can(policy_with_roles(), DPO, Action.READ_PROFILE, GOLD)


def test_a_role_inheritance_cycle_terminates():
    """A loop in a role graph is a config people write."""
    policy = AccessPolicy()
    policy.add_role(role_with("a", grant_for(Action.READ_GRAPH), inherits=["b"]))
    policy.add_role(role_with("b", inherits=["a"]))
    assert can(policy, Principal("x", frozenset({"a"})), Action.READ_GRAPH, GOLD)


def test_selectors_survive_new_tables_appearing():
    """A list of names goes stale, and the failure mode is silent lost access."""
    policy = AccessPolicy()
    policy.add_role(role_with("gold", grant_for(Action.READ_PROFILE, selector="*gold*")))
    principal = Principal("x", frozenset({"gold"}))
    assert can(policy, principal, Action.READ_PROFILE, DatasetId("duckdb", "gold.brand_new"))


def test_require_raises_with_the_decision_attached():
    with pytest.raises(AccessDenied) as caught:
        require(policy_with_roles(), ANALYST, Action.ERASE, GOLD)
    assert caught.value.decision.action is Action.ERASE
    assert not caught.value.decision.allowed


def test_require_returns_the_decision_when_allowed():
    assert require(policy_with_roles(), ANALYST, Action.READ_PROFILE, GOLD).allowed


def test_the_reverse_question_is_answerable():
    """Asked after an incident, and by every access review."""
    assert principals_who_can(policy_with_roles(), [ANALYST, DPO], Action.ERASE, GOLD) == ["bob"]


def test_write_actions_are_identified():
    assert Action.ERASE.is_write
    assert not Action.READ_GRAPH.is_write


# -- column visibility ---------------------------------------------------------


def test_an_unrestricted_grant_sees_every_column():
    columns = ["a", "b", "email"]
    assert columns_visible_to(policy_with_roles(), ANALYST, GOLD, columns) == columns


def test_a_column_restricted_grant_sees_only_what_it_names():
    policy = AccessPolicy()
    policy.add_role(role_with("narrow", grant_for(Action.READ_PROFILE, columns=["a", "b"])))
    principal = Principal("x", frozenset({"narrow"}))
    assert columns_visible_to(policy, principal, GOLD, ["a", "b", "email"]) == ["a", "b"]


def test_two_roles_union_rather_than_intersect():
    """Holding two roles should not see less than holding either."""
    policy = AccessPolicy()
    policy.add_role(role_with("one", grant_for(Action.READ_PROFILE, columns=["a"])))
    policy.add_role(role_with("two", grant_for(Action.READ_PROFILE, columns=["b"])))
    principal = Principal("x", frozenset({"one", "two"}))
    assert set(columns_visible_to(policy, principal, GOLD, ["a", "b", "c"])) == {"a", "b"}


def test_no_grant_sees_no_columns():
    assert columns_visible_to(policy_with_roles(), Principal("z", frozenset()), GOLD, ["a"]) == []


def test_masking_removes_columns_rather_than_blanking_them():
    """A masked column still discloses that it exists and what its null rate is,
    which is more than a denial should leak."""
    policy = AccessPolicy()
    policy.add_role(role_with("narrow", grant_for(Action.READ_PROFILE, columns=["a"])))
    principal = Principal("x", frozenset({"narrow"}))
    profile = {"columns": [{"name": "a", "nulls": 0}, {"name": "secret", "nulls": 5}]}

    masked = mask_profile(policy, principal, GOLD, profile)
    assert [c["name"] for c in masked["columns"]] == ["a"]


def test_redaction_can_keep_a_prefix_for_correlation():
    assert redact("abcdef", keep=2) == "ab***"
    assert redact("abcdef") == "***"


# -- audit ---------------------------------------------------------------------


def populated_log() -> AuditLog:
    log = AuditLog()
    record(log, "bob", "erase", target=str(USERS), detail={"subject_digest": "abc"})
    record(log, "alice", "read:profile", target=str(USERS), outcome=AuditOutcome.DENIED)
    record(log, "bob", "apply", target=str(GOLD))
    return log


def test_an_intact_chain_verifies():
    assert verify(populated_log()) == []


def test_an_empty_log_verifies():
    """Rather than being a special case every caller has to remember."""
    assert verify(AuditLog()) == []


def test_editing_an_entry_breaks_the_chain_at_a_known_index():
    """A log known to be altered somewhere is nearly useless; one altered at a known
    point leaves everything before it trustworthy."""
    log = populated_log()
    original = log.entries[0]
    log.entries[0] = AuditEntry(
        **{
            **original.body(),
            "at": original.at,
            "outcome": original.outcome,
            "actor": "mallory",
            "digest": original.digest,
        }
    )
    breaks = verify(log)
    assert breaks
    assert breaks[0].index == 0
    assert "altered" in breaks[0].problem


def test_removing_an_entry_breaks_the_link():
    log = populated_log()
    del log.entries[1]
    assert any("removed or reordered" in b.problem for b in verify(log))


def test_denials_are_recorded_too():
    """A log of only successes cannot answer "did anyone try", which is most of what
    an access review looks for."""
    assert summarize(populated_log())["denied"] == 1


def test_the_log_answers_who_touched_this():
    assert who_touched(populated_log(), str(USERS)) == ["alice", "bob"]


def test_entries_filter_conjunctively():
    log = populated_log()
    assert len(entries_for(log, actor="bob")) == 2
    assert len(entries_for(log, actor="bob", action="erase")) == 1


def test_entries_can_be_filtered_by_time():
    log = AuditLog()
    old = datetime.now(UTC) - timedelta(days=2)
    record(log, "a", "x", at=old)
    record(log, "b", "y")
    assert len(entries_since(log, datetime.now(UTC) - timedelta(days=1))) == 1


def test_a_log_round_trips_through_serialisation_and_still_verifies():
    log = populated_log()
    restored = replay([e.to_json() for e in log.entries])
    assert verify(restored) == []
    assert restored.head == log.head


def test_the_head_advances_with_each_entry():
    log = AuditLog()
    first = log.head
    record(log, "a", "x")
    assert log.head != first


# -- tenancy -------------------------------------------------------------------


def tenant_map() -> TenantMap:
    tenants = TenantMap()
    tenants.add(Tenant("team-a"))
    tenants.add(Tenant("team-b"))
    tenants.own(DatasetId("duckdb", "raw.events"), "team-a")
    tenants.own(DatasetId("duckdb", "a.mart"), "team-a")
    tenants.own(DatasetId("duckdb", "b.mart"), "team-b")
    return tenants


EDGES = [
    (DatasetId("duckdb", "raw.events"), DatasetId("duckdb", "a.mart")),
    (DatasetId("duckdb", "raw.events"), DatasetId("duckdb", "b.mart")),
]


def test_an_ungranted_crossing_is_a_violation():
    found = leaks(tenant_map(), EDGES)
    assert len(found) == 1
    assert found[0].target_tenant == "team-b"


def test_sharing_turns_a_violation_into_a_granted_crossing():
    tenants = tenant_map()
    share(tenants, DatasetId("duckdb", "raw.events"), with_tenants=["team-b"])
    report = tenant_summary(tenants, list(tenants.ownership), EDGES)
    assert report.ok
    assert len(report.shared) == 1


def test_a_declared_crossing_is_separated_from_a_violation():
    """A check that reports every legitimate shared dependency as a violation gets
    switched off, and then the real one is invisible too."""
    report = tenant_summary(tenant_map(), list(tenant_map().ownership), EDGES, declared=EDGES)
    assert not report.violations
    assert report.undeclared
    assert report.undeclared[0].kind is CrossingKind.UNDECLARED


def test_a_tenant_cannot_share_what_it_does_not_own():
    with pytest.raises(ValueError, match="no owner"):
        share(tenant_map(), DatasetId("duckdb", "nobody.table"), with_tenants=["team-b"])


def test_scoping_shows_own_plus_shared():
    tenants = tenant_map()
    share(tenants, DatasetId("duckdb", "raw.events"), with_tenants=["team-b"])
    visible = scope_graph(tenants, list(tenants.ownership), "team-b")
    assert DatasetId("duckdb", "b.mart") in visible
    assert DatasetId("duckdb", "raw.events") in visible
    assert DatasetId("duckdb", "a.mart") not in visible


def test_unowned_datasets_are_surfaced():
    """One has nobody to ask about a breach and nobody to bill for its storage."""
    orphan = DatasetId("duckdb", "orphan")
    assert unowned(tenant_map(), [orphan]) == [orphan]


def test_same_tenant_edges_are_not_crossings():
    tenants = tenant_map()
    internal = [(DatasetId("duckdb", "raw.events"), DatasetId("duckdb", "a.mart"))]
    assert leaks(tenants, internal) == []


# -- keys ----------------------------------------------------------------------


def test_destroying_a_subjects_key_makes_it_unreadable():
    registry = KeyRegistry(salt="s")
    register(registry, "k-alice", subjects=["alice"])
    assert covered_by(registry, "alice") == ["k-alice"]

    destroy_for_subject(registry, "alice", reason="DSR-1")
    assert verify_destroyed(registry, "alice")
    assert covered_by(registry, "alice") == []


def test_destroying_a_shared_key_is_refused():
    """Destroying it anyway erases a person who did not ask, which is an outage
    dressed as compliance."""
    registry = KeyRegistry()
    register(registry, "k-shared", subjects=["bob", "carol"])
    with pytest.raises(ShredRefused, match="also cover other subjects"):
        destroy_for_subject(registry, "bob")
    assert not verify_destroyed(registry, "bob")


def test_a_destroyed_key_cannot_be_re_registered():
    """The alternative silently resurrects an erasure already certified as done."""
    registry = KeyRegistry()
    register(registry, "k", subjects=["alice"])
    destroy(registry, "k")
    with pytest.raises(ShredRefused, match="already been certified"):
        register(registry, "k", subjects=["alice"])


def test_the_record_of_a_destroyed_key_survives_it():
    """A registry that forgets the key existed cannot prove it was destroyed."""
    registry = KeyRegistry()
    register(registry, "k", subjects=["alice"])
    destroyed = destroy(registry, "k", reason="DSR-2")

    assert registry.get("k") is not None
    assert destroyed.state is KeyState.DESTROYED
    assert destroyed.destroyed is not None
    assert destroyed.reason == "DSR-2"


def test_destroying_twice_is_idempotent():
    registry = KeyRegistry()
    register(registry, "k", subjects=["alice"])
    first = destroy(registry, "k")
    assert destroy(registry, "k").destroyed == first.destroyed


def test_rotation_retains_the_old_key_so_ciphertext_stays_readable():
    """A rotated key that is discarded takes its ciphertext with it, which is an
    outage rather than an erasure."""
    registry = KeyRegistry()
    register(registry, "k1", subjects=["alice"])
    rotate(registry, "k1", "k2")

    assert registry.get("k1").state is KeyState.ROTATED
    assert registry.get("k1").state.can_decrypt
    assert sorted(covered_by(registry, "alice")) == ["k1", "k2"]


def test_a_destroyed_key_cannot_be_rotated():
    registry = KeyRegistry()
    register(registry, "k", subjects=["alice"])
    destroy(registry, "k")
    with pytest.raises(ShredRefused, match="destroyed"):
        rotate(registry, "k", "k2")


def test_a_shred_proof_is_incomplete_while_any_key_survives():
    registry = KeyRegistry(salt="s")
    register(registry, "k1", subjects=["alice"])
    register(registry, "k2", subjects=["alice"])
    destroy(registry, "k1")

    proof = shred_proof(registry, "alice", reference="DSR-3")
    assert not proof.complete
    assert "INCOMPLETE" in proof.summary()


def test_a_shred_proof_never_contains_the_subject():
    registry = KeyRegistry(salt="org-secret")
    register(registry, "k", subjects=["alice@example.com"])
    destroy(registry, "k")
    body = shred_proof(registry, "alice@example.com", reference="DSR-4").to_json()
    assert "alice@example.com" not in body
    assert "digest" in body


def test_the_salt_changes_the_digest():
    assert KeyRegistry(salt="a").subject_digest("x") != KeyRegistry(salt="b").subject_digest("x")


def test_keys_for_a_subject_are_findable():
    registry = KeyRegistry()
    register(registry, "k1", subjects=["alice"])
    register(registry, "k2", subjects=["bob"])
    assert [k.identifier for k in keys_for_subject(registry, "alice")] == ["k1"]


def test_rotating_a_missing_key_raises():
    with pytest.raises(KeyError):
        rotate(KeyRegistry(), "nope", "k2")
