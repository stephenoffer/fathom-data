"""Shared fixture for the examples: a small three-layer DuckDB warehouse."""

from __future__ import annotations

from fathom import Grain, PartitionField, PartitionSpec
from fathom.adapters import DuckDBEngine
from fathom.core.ids import normalize_table

RAW = normalize_table("raw.events", system="duckdb")
SILVER = normalize_table("silver.events", system="duckdb")
GOLD = normalize_table("gold.monthly", system="duckdb")

DAY_REGION = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
MONTH_REGION = PartitionSpec.of(
    PartitionField.time("dt", Grain.MONTH), PartitionField.value("region")
)
SPECS = {RAW: DAY_REGION, SILVER: DAY_REGION, GOLD: MONTH_REGION}

SEED_ROWS = [
    ("2026-03-14", "eu", "u1", 10.0),
    ("2026-03-14", "eu", "u2", 5.0),
    ("2026-03-14", "us", "u2", 20.0),
    ("2026-03-15", "eu", "u1", 30.0),
    ("2026-04-01", "eu", "u2", 40.0),
    ("2026-04-02", "us", "u3", 50.0),
]


def build_warehouse() -> DuckDBEngine:
    """raw.events -> silver.events (daily) -> gold.monthly (monthly rollup)."""
    engine = DuckDBEngine()
    conn = engine.connect()
    conn.execute("CREATE SCHEMA raw; CREATE SCHEMA silver; CREATE SCHEMA gold")
    conn.execute(
        "CREATE TABLE raw.events (dt DATE, region VARCHAR, user_id VARCHAR, amount DOUBLE)"
    )
    conn.executemany("INSERT INTO raw.events VALUES (?, ?, ?, ?)", SEED_ROWS)

    engine.register_model(SILVER, "SELECT dt, region, user_id, amount FROM raw.events", DAY_REGION)
    engine.register_model(
        GOLD,
        "SELECT DATE_TRUNC('month', dt) AS dt, region, SUM(amount) AS revenue "
        "FROM silver.events GROUP BY 1, 2",
        MONTH_REGION,
    )
    engine.full_rebuild(SILVER)
    engine.full_rebuild(GOLD)
    return engine
