"""Erasure planning, refusal, and execution.

The failure mode being defended against is a tool that reports success while the
subject's data survives somewhere. Every test here is about making that impossible
to do accidentally.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from fathom.adapters.engines.duckdb import DuckDBEngine
from fathom.core.grains import Grain
from fathom.core.ids import normalize_table
from fathom.core.types import (
    Capabilities,
    ChangeSource,
    ErasureMode,
    KeyPredicate,
    LineageSource,
    PartitionField,
    PartitionSpec,
)
from fathom.govern.erasure import ErasureRequest, apply_erasure, plan_erasure, unerasable
from fathom.ingest import ingest_engine

RAW = normalize_table("raw.events", system="duckdb")
SILVER = normalize_table("silver.events", system="duckdb")
GOLD = normalize_table("gold.monthly", system="duckdb")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MONTH = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))
SPECS = {RAW: DAY, SILVER: DAY, GOLD: MONTH}


def caps(mode: ErasureMode) -> Capabilities:
    return Capabilities(
        lineage=LineageSource.QUERY_LOG, change=ChangeSource.WATERMARK, erasure=mode
    )


ALL_REWRITABLE = {ds: caps(ErasureMode.REWRITE) for ds in (RAW, SILVER, GOLD)}


@pytest.fixture
def engine():
    eng = DuckDBEngine()
    conn = eng.connect()
    conn.execute("CREATE SCHEMA raw; CREATE SCHEMA silver; CREATE SCHEMA gold")
    conn.execute("CREATE TABLE raw.events (dt DATE, user_id VARCHAR, amount DOUBLE)")
    conn.executemany(
        "INSERT INTO raw.events VALUES (?, ?, ?)",
        [
            ("2026-03-14", "u1", 10.0),
            ("2026-03-14", "u2", 20.0),
            ("2026-03-15", "u1", 30.0),
            ("2026-04-01", "u2", 40.0),
        ],
    )
    eng.register_model(SILVER, "SELECT dt, user_id, amount FROM raw.events", DAY)
    eng.register_model(
        GOLD,
        "SELECT DATE_TRUNC('month', dt) AS dt, SUM(amount) AS revenue "
        "FROM silver.events GROUP BY 1",
        MONTH,
    )
    eng.full_rebuild(SILVER)
    eng.full_rebuild(GOLD)
    yield eng
    eng.close()


@pytest.fixture
def graph(engine):
    return ingest_engine(engine, specs=SPECS).graph


def request_for_u1() -> ErasureRequest:
    return ErasureRequest(
        subject="u1",
        key_column="user_id",
        origin=RAW,
        partitions=frozenset(
            {KeyPredicate.of(dt=datetime(2026, 3, 14)), KeyPredicate.of(dt=datetime(2026, 3, 15))}
        ),
        reference="DSR-1234",
    )


# -- planning ------------------------------------------------------------------


def test_plan_reaches_every_derived_copy(graph):
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    assert {t.dataset for t in plan.targets} == {RAW, SILVER, GOLD}


def test_targets_are_ordered_sources_before_derived(graph):
    """Re-deriving an aggregate before its source is erased silently does nothing."""
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    order = [t.dataset for t in plan.targets]
    assert order.index(RAW) < order.index(SILVER) < order.index(GOLD)


def test_plan_scopes_to_the_partitions_holding_the_subject(graph):
    """The reason this is affordable: March only, not the whole table."""
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    gold = next(t for t in plan.targets if t.dataset == GOLD)
    assert gold.partitions == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 1))})


def test_missing_adapter_blocks_rather_than_assuming_erasable(graph):
    plan = plan_erasure(graph, request_for_u1(), capabilities={RAW: caps(ErasureMode.REWRITE)})
    assert not plan.is_complete
    assert set(unerasable(plan)) == {SILVER, GOLD}


def test_worm_storage_is_refused_with_a_reason(graph):
    capabilities = {**ALL_REWRITABLE, GOLD: caps(ErasureMode.NONE)}
    plan = plan_erasure(graph, request_for_u1(), capabilities=capabilities)

    assert not plan.is_complete
    blocked = next(t for t in plan.targets if t.dataset == GOLD)
    assert "Object Lock" in (blocked.blocked or "")
    assert "crypto-shred" in (blocked.blocked or "")


def test_incomplete_plans_say_so_loudly(graph):
    capabilities = {**ALL_REWRITABLE, GOLD: caps(ErasureMode.NONE)}
    summary = plan_erasure(graph, request_for_u1(), capabilities=capabilities).summary()
    assert "INCOMPLETE" in summary
    assert "do not report the request as fulfilled" in summary


# -- proofs --------------------------------------------------------------------


def test_dry_run_is_the_default(graph, engine):
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    before = engine.rows(RAW)

    proof = apply_erasure(plan, {ds: engine for ds in (RAW, SILVER, GOLD)}, salt="org-secret")

    assert engine.rows(RAW) == before
    assert not proof.executed
    assert all(e["status"] == "planned" for e in proof.entries)


def test_proof_never_contains_the_subject_in_plaintext(graph, engine):
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    proof = apply_erasure(plan, {ds: engine for ds in (RAW, SILVER, GOLD)}, salt="org-secret")
    body = proof.to_json()

    assert "u1" not in json.loads(body)["subject_digest"]
    assert json.loads(body)["reference"] == "DSR-1234"
    assert len(proof.subject_digest) == 64


def test_salt_changes_the_digest(graph):
    request = request_for_u1()
    assert request.subject_digest("a") != request.subject_digest("b")
    assert request.subject_digest("a") == request.subject_digest("a")


def test_proof_digest_covers_the_body(graph, engine):
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    proof = apply_erasure(plan, {ds: engine for ds in (RAW, SILVER, GOLD)}, salt="org-secret")
    body = json.loads(proof.to_json())
    assert body["digest"] == proof.digest
    assert len(body["digest"]) == 64


def test_dry_run_proof_is_never_marked_complete(graph, engine):
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    proof = apply_erasure(
        plan, {ds: engine for ds in (RAW, SILVER, GOLD)}, salt="org-secret", dry_run=True
    )
    assert not proof.complete


# -- execution -----------------------------------------------------------------


def test_execution_removes_the_subject_everywhere(graph, engine):
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    erasers = {RAW: engine, SILVER: engine, GOLD: engine}

    # Order is load-bearing: sources first, then re-derive downstream.
    proof = apply_erasure(plan, erasers, salt="org-secret", dry_run=False)

    assert proof.executed and proof.complete
    assert all(row[1] != "u1" for row in engine.rows(RAW))
    assert all(row[1] != "u1" for row in engine.rows(SILVER))


def test_derived_aggregates_are_rebuilt_not_deleted(graph, engine):
    """gold has no user_id column; the only correct action is to re-derive it."""
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    apply_erasure(
        plan, {RAW: engine, SILVER: engine, GOLD: engine}, salt="org-secret", dry_run=False
    )

    revenue = {row[0]: row[1] for row in engine.rows(GOLD)}
    march = next(v for k, v in revenue.items() if k.month == 3)
    assert march == 20.0  # u2's 20.0 only; u1's 10.0 and 30.0 are gone


def test_untouched_partitions_are_left_alone(graph, engine):
    before = engine.fingerprints(GOLD)
    plan = plan_erasure(graph, request_for_u1(), capabilities=ALL_REWRITABLE)
    apply_erasure(
        plan, {RAW: engine, SILVER: engine, GOLD: engine}, salt="org-secret", dry_run=False
    )
    after = engine.fingerprints(GOLD)

    april = KeyPredicate.of(dt=datetime(2026, 4, 1))
    assert before[april] == after[april]


def test_blocked_targets_are_recorded_and_the_proof_stays_incomplete(graph, engine):
    capabilities = {**ALL_REWRITABLE, GOLD: caps(ErasureMode.NONE)}
    plan = plan_erasure(graph, request_for_u1(), capabilities=capabilities)
    proof = apply_erasure(plan, {RAW: engine, SILVER: engine}, salt="org-secret", dry_run=False)

    blocked = [e for e in proof.entries if e["status"] == "blocked"]
    assert [e["dataset"] for e in blocked] == [str(GOLD)]
    assert not proof.complete


def test_erasing_a_source_without_the_key_column_is_refused(engine):
    """Better to fail loudly than to delete by some other column and hope."""
    unmodelled = normalize_table("raw.other", system="duckdb")
    engine.connect().execute("CREATE TABLE raw.other AS SELECT 1 AS x")
    with pytest.raises(ValueError, match="no column 'user_id'"):
        engine.erase(unmodelled, key_column="user_id", subject="u1", partitions=[KeyPredicate()])
