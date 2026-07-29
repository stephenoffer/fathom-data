"""Copies lineage cannot see, and the refusal to let time close them out."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.types import DatasetId
from fathom.govern import replicas

EVENTS = DatasetId("duckdb", "raw.events")
ORDERS = DatasetId("duckdb", "gold.orders")

ERASED = datetime(2026, 3, 1, tzinfo=UTC)
LATER = ERASED + timedelta(days=40)


def snapshot(**kwargs):
    base = {"dataset": EVENTS, "kind": replicas.CopyKind.SNAPSHOT, "location": "acct-2/backups"}
    return replicas.declare(**{**base, **kwargs})


# -- declaring -----------------------------------------------------------------


def test_a_replica_declares_its_kind_and_location():
    made = snapshot(owner="platform")
    assert made.kind is replicas.CopyKind.SNAPSHOT
    assert made.location == "acct-2/backups"
    assert made.owner == "platform"


def test_the_kind_may_be_a_string_so_config_can_build_one():
    assert replicas.declare(EVENTS, "archive", "glacier/2026").kind is replicas.CopyKind.ARCHIVE


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        replicas.declare(EVENTS, "sticky-note", "the fridge")


def test_a_replica_without_a_location_is_rejected():
    """'Somewhere else' is the caveat this module exists to replace."""
    with pytest.raises(ValueError, match="needs a location"):
        replicas.declare(EVENTS, replicas.CopyKind.SNAPSHOT, "   ")


def test_a_location_is_trimmed():
    assert replicas.declare(EVENTS, "snapshot", "  acct-2  ").location == "acct-2"


# -- disposition ---------------------------------------------------------------


def test_a_live_replica_follows_the_source():
    made = snapshot(kind=replicas.CopyKind.REPLICA, location="reader-1")
    assert (
        replicas.disposition_of(made, erased_at=ERASED, at=LATER)
        is replicas.Disposition.FOLLOWS_SOURCE
    )


def test_a_cache_follows_the_source_too():
    made = snapshot(kind=replicas.CopyKind.CACHE, location="redis")
    assert (
        replicas.disposition_of(made, erased_at=ERASED, at=LATER)
        is replicas.Disposition.FOLLOWS_SOURCE
    )


def test_an_export_is_unreachable_whatever_else_is_true():
    """Deleting the source does nothing to a CSV somebody already has."""
    made = snapshot(kind=replicas.CopyKind.EXPORT, location="partner-sftp")
    assert (
        replicas.disposition_of(made, erased_at=ERASED, at=LATER)
        is replicas.Disposition.UNREACHABLE
    )


def test_a_third_party_copy_is_unreachable_even_when_attested():
    """An attestation about someone else's system is a claim, not an action."""
    made = snapshot(kind=replicas.CopyKind.THIRD_PARTY, location="vendor-x")
    attested = {"vendor-x": replicas.Attestation("vendor-x", LATER, by="ana")}
    assert (
        replicas.disposition_of(made, erased_at=ERASED, attestations=attested, at=LATER)
        is replicas.Disposition.UNREACHABLE
    )


def test_an_elapsed_retention_window_expires_a_snapshot():
    made = snapshot(retention=timedelta(days=30))
    assert replicas.disposition_of(made, erased_at=ERASED, at=LATER) is replicas.Disposition.EXPIRED


def test_a_retention_window_still_running_stays_outstanding():
    made = snapshot(retention=timedelta(days=90))
    assert (
        replicas.disposition_of(made, erased_at=ERASED, at=LATER)
        is replicas.Disposition.OUTSTANDING
    )


def test_no_retention_means_unbounded_not_expired():
    """A backup nobody set a lifecycle rule on does not quietly go away."""
    assert (
        replicas.disposition_of(snapshot(), erased_at=ERASED, at=LATER)
        is replicas.Disposition.OUTSTANDING
    )


def test_an_attestation_closes_an_internal_copy():
    attested = {"acct-2/backups": replicas.Attestation("acct-2/backups", LATER, by="ana")}
    assert (
        replicas.disposition_of(snapshot(), erased_at=ERASED, attestations=attested, at=LATER)
        is replicas.Disposition.ATTESTED
    )


def test_an_attestation_for_a_different_location_does_not_close_this_one():
    attested = {"elsewhere": replicas.Attestation("elsewhere", LATER)}
    assert (
        replicas.disposition_of(snapshot(), erased_at=ERASED, attestations=attested, at=LATER)
        is replicas.Disposition.OUTSTANDING
    )


# -- the report ----------------------------------------------------------------


def test_a_report_groups_by_disposition():
    declared = [
        snapshot(),
        snapshot(kind=replicas.CopyKind.REPLICA, location="reader-1"),
        snapshot(kind=replicas.CopyKind.EXPORT, location="partner"),
    ]
    result = replicas.report(EVENTS, declared, erased_at=ERASED, at=LATER)
    assert result.total == 3
    assert len(result.outstanding) == 1
    assert len(result.unreachable) == 1


def test_a_report_ignores_copies_of_other_datasets():
    declared = [snapshot(), snapshot(dataset=ORDERS, location="other")]
    assert replicas.report(EVENTS, declared, erased_at=ERASED, at=LATER).total == 1


def test_nothing_declared_says_so_without_claiming_none_exist():
    result = replicas.report(EVENTS, [], erased_at=ERASED, at=LATER)
    assert "no copies declared" in result.summary()
    assert "not the same as none" in result.summary()


def test_an_outstanding_copy_blocks_discharge():
    result = replicas.report(EVENTS, [snapshot()], erased_at=ERASED, at=LATER)
    assert not result.is_discharged
    assert "Time passing is not evidence" in result.summary()


def test_an_unreachable_copy_blocks_discharge_however_much_else_was_handled():
    """A dataset with an outstanding export is not discharged, whatever else was done."""
    declared = [
        snapshot(kind=replicas.CopyKind.REPLICA, location="reader-1"),
        snapshot(kind=replicas.CopyKind.EXPORT, location="partner"),
    ]
    result = replicas.report(EVENTS, declared, erased_at=ERASED, at=LATER)
    assert result.outstanding == []
    assert not result.is_discharged


def test_everything_handled_discharges():
    declared = [snapshot(retention=timedelta(days=30))]
    result = replicas.report(EVENTS, declared, erased_at=ERASED, at=LATER)
    assert result.is_discharged


# -- the work list -------------------------------------------------------------


def test_obligations_say_what_to_do():
    declared = [snapshot(owner="platform")]
    (_, action) = replicas.obligations_for(declared, erased_at=ERASED, at=LATER)[0]
    assert "delete or overwrite the snapshot" in action
    assert "owned by platform" in action
    assert "record an attestation" in action


def test_an_unreachable_copy_asks_for_a_letter_not_a_delete():
    declared = [snapshot(kind=replicas.CopyKind.THIRD_PARTY, location="vendor-x", owner="Vendor X")]
    (_, action) = replicas.obligations_for(declared, erased_at=ERASED, at=LATER)[0]
    assert "only Vendor X can delete" in action
    assert "in writing" in action


def test_obligations_put_the_ones_somebody_else_owns_last():
    """Mixing them into the internal work list is how they get lost."""
    declared = [
        snapshot(kind=replicas.CopyKind.EXPORT, location="partner"),
        snapshot(location="acct-2/backups"),
    ]
    found = replicas.obligations_for(declared, erased_at=ERASED, at=LATER)
    assert found[0][0].kind is replicas.CopyKind.SNAPSHOT
    assert found[-1][0].kind is replicas.CopyKind.EXPORT


def test_a_handled_copy_produces_no_obligation():
    declared = [snapshot(retention=timedelta(days=1))]
    assert replicas.obligations_for(declared, erased_at=ERASED, at=LATER) == []


def test_the_filters_agree_with_the_dispositions():
    declared = [
        snapshot(location="a"),
        snapshot(location="b", retention=timedelta(days=1)),
    ]
    attestations = [replicas.Attestation("a", LATER)]
    assert [r.location for r in replicas.outstanding(declared, erased_at=ERASED, at=LATER)] == ["a"]
    assert [r.location for r in replicas.expired_by(declared, erased_at=ERASED, at=LATER)] == ["b"]
    assert [
        r.location for r in replicas.attested(declared, attestations, erased_at=ERASED, at=LATER)
    ] == ["a"]


# -- estate-level --------------------------------------------------------------


def test_coverage_is_the_declared_fraction():
    assert replicas.coverage([EVENTS, ORDERS], [snapshot()]) == pytest.approx(0.5)


def test_coverage_of_no_datasets_is_zero():
    assert replicas.coverage([], [snapshot()]) == 0.0


def test_undeclared_datasets_are_the_ones_a_proof_cannot_bound():
    assert replicas.undeclared_datasets([EVENTS, ORDERS], [snapshot()]) == [ORDERS]


def test_by_kind_groups_the_inventory():
    declared = [snapshot(), snapshot(kind=replicas.CopyKind.EXPORT, location="partner")]
    grouped = replicas.by_kind(declared)
    assert set(grouped) == {replicas.CopyKind.SNAPSHOT, replicas.CopyKind.EXPORT}


def test_for_dataset_filters():
    declared = [snapshot(), snapshot(dataset=ORDERS, location="other")]
    assert replicas.for_dataset(declared, ORDERS)[0].dataset == ORDERS


# -- the sentence a proof carries ----------------------------------------------


def test_the_summary_replaces_the_unfalsifiable_caveat_with_a_list():
    declared = [
        snapshot(kind=replicas.CopyKind.EXPORT, location="partner-sftp", owner="Partner Co"),
        snapshot(),
    ]
    text = replicas.unreachable_summary(declared)
    assert "1 declared copy(ies) are beyond" in text
    assert "partner-sftp" in text
    assert "Partner Co" in text
    assert "acct-2/backups" not in text  # internal copies are not in this list


def test_nothing_beyond_control_still_refuses_to_claim_completeness():
    text = replicas.unreachable_summary([snapshot()])
    assert "Absence of a declaration is not evidence of absence" in text


def test_an_unknown_holder_is_named_as_unknown():
    declared = [snapshot(kind=replicas.CopyKind.EXPORT, location="partner")]
    assert "holder unknown" in replicas.unreachable_summary(declared)


# -- reaching the proof --------------------------------------------------------


def test_proof_entries_name_every_copy_with_its_disposition():
    """A proof listing five named copies is one an auditor can close out."""
    declared = [
        snapshot(owner="platform"),
        snapshot(kind=replicas.CopyKind.EXPORT, location="partner", owner="Partner Co"),
    ]
    entries = replicas.proof_entries(declared, erased_at=ERASED, at=LATER)
    assert [e["disposition"] for e in entries] == ["outstanding", "unreachable"]
    assert entries[0]["location"] == "acct-2/backups"
    assert entries[1]["owner"] == "Partner Co"


def test_an_attestation_reaches_the_proof_entry():
    attestations = [replicas.Attestation("acct-2/backups", LATER, by="ana")]
    (entry,) = replicas.proof_entries(
        [snapshot()], erased_at=ERASED, attestations=attestations, at=LATER
    )
    assert entry["disposition"] == "attested"
    assert entry["attested_by"] == "ana"
    assert entry["attested_at"].startswith("2026-04-10")


def test_entries_are_ordered_so_a_proof_is_stable():
    declared = [snapshot(location="z"), snapshot(location="a")]
    entries = replicas.proof_entries(declared, erased_at=ERASED, at=LATER)
    assert [e["location"] for e in entries] == ["a", "z"]


def test_discharged_is_false_while_anything_is_outstanding():
    assert not replicas.discharged([snapshot()], erased_at=ERASED, at=LATER)


def test_discharged_is_false_for_anything_beyond_control():
    declared = [snapshot(kind=replicas.CopyKind.EXPORT, location="partner")]
    assert not replicas.discharged(declared, erased_at=ERASED, at=LATER)


def test_discharged_is_true_once_everything_is_closed():
    declared = [
        snapshot(location="a", retention=timedelta(days=1)),
        snapshot(kind=replicas.CopyKind.REPLICA, location="reader-1"),
    ]
    assert replicas.discharged(declared, erased_at=ERASED, at=LATER)


def test_nothing_declared_is_vacuously_discharged():
    """Which is exactly why `coverage` exists — declaring nothing is not being clean."""
    assert replicas.discharged([], erased_at=ERASED, at=LATER)
