"""End-to-end proof that the planner is worth trusting.

Three claims, in order of importance:

1. An incremental rebuild driven by the plan produces byte-identical results to a
   full rebuild. If this fails, nothing else about the project matters.
2. Shadow mode grades a real plan and reports zero missed partitions.
3. Shadow mode actually *catches* an unsound plan. A soundness checker that cannot
   fail is decoration.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from fathom import shadow
from fathom.adapters.engines.duckdb import DuckDBEngine, render_predicate
from fathom.core.grains import Grain
from fathom.core.ids import normalize_table
from fathom.core.types import ANY, KeyPredicate, PartitionField, PartitionSpec
from fathom.ingest import ingest_engine
from fathom.store import Store

RAW = normalize_table("raw.events", system="duckdb")
SILVER = normalize_table("silver.events", system="duckdb")
GOLD = normalize_table("gold.monthly", system="duckdb")

DAY_REGION = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
MONTH_REGION = PartitionSpec.of(
    PartitionField.time("dt", Grain.MONTH), PartitionField.value("region")
)
SPECS = {RAW: DAY_REGION, SILVER: DAY_REGION, GOLD: MONTH_REGION}

SEED_ROWS = [
    ("2026-03-14", "eu", 1.0),
    ("2026-03-14", "us", 2.0),
    ("2026-03-15", "eu", 3.0),
    ("2026-04-01", "eu", 4.0),
    ("2026-04-02", "us", 5.0),
]


@pytest.fixture
def engine() -> DuckDBEngine:
    eng = DuckDBEngine()
    conn = eng.connect()
    conn.execute("CREATE SCHEMA raw; CREATE SCHEMA silver; CREATE SCHEMA gold")
    conn.execute("CREATE TABLE raw.events (dt DATE, region VARCHAR, amount DOUBLE)")
    conn.executemany("INSERT INTO raw.events VALUES (?, ?, ?)", SEED_ROWS)

    eng.register_model(SILVER, "SELECT dt, region, amount FROM raw.events", DAY_REGION)
    eng.register_model(
        GOLD,
        "SELECT DATE_TRUNC('month', dt) AS dt, region, SUM(amount) AS revenue "
        "FROM silver.events GROUP BY 1, 2",
        MONTH_REGION,
    )
    eng.full_rebuild(SILVER)
    eng.full_rebuild(GOLD)
    yield eng
    eng.close()


def build_graph(engine: DuckDBEngine):
    return ingest_engine(engine, specs=SPECS).graph


def landed_partition() -> KeyPredicate:
    return KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")


def land_new_data(engine: DuckDBEngine) -> None:
    """New rows arrive for one day and region only."""
    engine.connect().execute("INSERT INTO raw.events VALUES ('2026-03-14', 'eu', 99.0)")


# -- the claim that matters ---------------------------------------------------


def test_incremental_rebuild_matches_full_rebuild(engine: DuckDBEngine):
    graph = build_graph(engine)
    land_new_data(engine)

    plan = graph.invalidate({RAW: [landed_partition()]})
    assert plan.order, "planner produced nothing to rebuild"

    for dataset in plan.order:
        if dataset in engine.models:
            engine.apply(dataset, plan.partitions(dataset))

    incremental = {ds: engine.fingerprints(ds) for ds in (SILVER, GOLD)}

    engine.full_rebuild(SILVER)
    engine.full_rebuild(GOLD)
    full = {ds: engine.fingerprints(ds) for ds in (SILVER, GOLD)}

    assert incremental == full, "incremental rebuild diverged from a full rebuild"


def test_incremental_rebuild_touches_only_the_affected_partitions(engine: DuckDBEngine):
    """The savings claim: untouched partitions must not be rewritten."""
    graph = build_graph(engine)
    before = engine.fingerprints(GOLD)
    land_new_data(engine)

    plan = graph.invalidate({RAW: [landed_partition()]})
    for dataset in plan.order:
        if dataset in engine.models:
            engine.apply(dataset, plan.partitions(dataset))

    after = engine.fingerprints(GOLD)
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}

    assert changed == {KeyPredicate.of(dt=datetime(2026, 3, 1), region="eu")}
    # April and the US region were never in the plan and must be untouched.
    assert (
        before[KeyPredicate.of(dt=datetime(2026, 4, 1), region="eu")]
        == after[KeyPredicate.of(dt=datetime(2026, 4, 1), region="eu")]
    )


# -- shadow mode ---------------------------------------------------------------


def test_shadow_grades_a_real_plan_as_sound(engine: DuckDBEngine):
    graph = build_graph(engine)
    land_new_data(engine)
    plan = graph.invalidate({RAW: [landed_partition()]})

    report = shadow.run(engine, plan, [SILVER, GOLD])

    assert report.is_sound, report.summary()
    assert report.missed_total == 0
    assert report.savings > 0.5, f"expected real savings, got {report.savings:.0%}"


def test_shadow_catches_an_unsound_plan(engine: DuckDBEngine):
    """Deliberately plan the wrong partition; the checker must notice."""
    graph = build_graph(engine)
    land_new_data(engine)

    plan = graph.invalidate({RAW: [landed_partition()]})
    # Point the plan at a partition that did not change, and drop the one that did.
    plan.dirty[SILVER] = frozenset({KeyPredicate.of(dt=datetime(2026, 4, 2), region="us")})
    plan.dirty[GOLD] = frozenset({KeyPredicate.of(dt=datetime(2026, 4, 1), region="us")})

    report = shadow.run(engine, plan, [SILVER, GOLD])

    assert not report.is_sound
    assert report.missed_total > 0
    assert "UNSOUND" in report.summary()


def test_shadow_observations_persist(engine: DuckDBEngine):
    graph = build_graph(engine)
    land_new_data(engine)
    plan = graph.invalidate({RAW: [landed_partition()]})

    with Store() as store:
        shadow.run(engine, plan, [SILVER, GOLD], store=store)
        summary = store.shadow_summary()
        assert summary["runs"] == 2
        assert summary["missed"] == 0
        assert len(store.shadow_history()) == 2


# -- rendering -----------------------------------------------------------------


def test_predicate_uses_a_half_open_range_for_time():
    sql = render_predicate(DAY_REGION, [KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")])
    assert "\"dt\" >= TIMESTAMP '2026-03-14 00:00:00'" in sql
    assert "\"dt\" < TIMESTAMP '2026-03-15 00:00:00'" in sql
    assert "\"region\" = 'eu'" in sql


def test_unbounded_key_collapses_the_whole_predicate():
    """One unbounded key means a full rebuild; ORing it with others would be wrong."""
    keys = [
        KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu"),
        KeyPredicate.of(dt=ANY, region=ANY),
    ]
    assert render_predicate(DAY_REGION, keys) == "TRUE"


def test_predicate_escapes_quotes():
    sql = render_predicate(DAY_REGION, [KeyPredicate.of(dt=ANY, region="o'brien")])
    assert "'o''brien'" in sql


def test_rebuild_statements_are_delete_then_insert(engine: DuckDBEngine):
    statements = engine.render_rebuild(SILVER, [landed_partition()])
    assert statements[0].startswith("DELETE FROM silver.events WHERE")
    assert statements[1].startswith("INSERT INTO silver.events SELECT * FROM (")


def test_apply_rolls_back_on_failure(engine: DuckDBEngine):
    """A partially applied rebuild would leave a partition half-deleted."""
    before = engine.fingerprints(SILVER)
    engine.register_model(SILVER, "SELECT * FROM does_not_exist", DAY_REGION)
    with pytest.raises(duckdb.Error):
        engine.apply(SILVER, [landed_partition()])
    assert engine.fingerprints(SILVER) == before
