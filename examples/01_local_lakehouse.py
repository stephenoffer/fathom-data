"""The whole loop on a local DuckDB warehouse.

Builds a three-layer pipeline, lands one day of new data, plans the rebuild, applies
it, and then proves the incremental result is byte-identical to a full rebuild.

That last assertion is the claim everything else in this project rests on.

    python examples/01_local_lakehouse.py
"""

from __future__ import annotations

from datetime import datetime

from fathom import Grain, KeyPredicate, PartitionField, PartitionSpec
from fathom.adapters import DuckDBEngine
from fathom.core.ids import normalize_table
from fathom.ingest import ingest_engine

RAW = normalize_table("raw.events", system="duckdb")
SILVER = normalize_table("silver.events", system="duckdb")
GOLD = normalize_table("gold.monthly", system="duckdb")

DAY_REGION = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
MONTH_REGION = PartitionSpec.of(
    PartitionField.time("dt", Grain.MONTH), PartitionField.value("region")
)
SPECS = {RAW: DAY_REGION, SILVER: DAY_REGION, GOLD: MONTH_REGION}


def build_warehouse() -> DuckDBEngine:
    engine = DuckDBEngine()
    conn = engine.connect()
    conn.execute("CREATE SCHEMA raw; CREATE SCHEMA silver; CREATE SCHEMA gold")
    conn.execute("CREATE TABLE raw.events (dt DATE, region VARCHAR, amount DOUBLE)")
    conn.executemany(
        "INSERT INTO raw.events VALUES (?, ?, ?)",
        [
            ("2026-03-14", "eu", 10.0),
            ("2026-03-14", "us", 20.0),
            ("2026-03-15", "eu", 30.0),
            ("2026-04-01", "eu", 40.0),
            ("2026-04-02", "us", 50.0),
        ],
    )

    engine.register_model(SILVER, "SELECT dt, region, amount FROM raw.events", DAY_REGION)
    engine.register_model(
        GOLD,
        "SELECT DATE_TRUNC('month', dt) AS dt, region, SUM(amount) AS revenue "
        "FROM silver.events GROUP BY 1, 2",
        MONTH_REGION,
    )
    engine.full_rebuild(SILVER)
    engine.full_rebuild(GOLD)
    return engine


def main() -> None:
    engine = build_warehouse()

    # 1. Build the dependency graph from the registered models.
    graph = ingest_engine(engine, specs=SPECS).graph
    print("Graph:")
    for edge in graph.edges:
        print(f"  {edge}")

    # 2. One day of new data arrives, for one region only.
    engine.connect().execute("INSERT INTO raw.events VALUES ('2026-03-14', 'eu', 99.0)")
    landed = KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
    print(f"\nNew data landed in {RAW} at {landed}")

    # 3. Plan. One dirty day becomes one dirty day downstream and one dirty *month*
    #    in the rollup, still scoped to a single region.
    plan = graph.invalidate({RAW: [landed]})
    print("\nPlan:")
    print("  " + plan.summary().replace("\n", "\n  "))

    # 4. Snapshot everything, then apply only the planned partitions.
    before = {ds: engine.fingerprints(ds) for ds in (SILVER, GOLD)}
    for dataset in plan.order:
        if dataset in engine.models:
            statements = engine.apply(dataset, plan.partitions(dataset))
            print(f"\nRebuilt {dataset}:")
            for statement in statements:
                print(f"  {statement[:100]}...")

    incremental = {ds: engine.fingerprints(ds) for ds in (SILVER, GOLD)}

    # 5. Which partitions actually moved? Only the ones we planned.
    touched = {
        key
        for key in set(before[GOLD]) | set(incremental[GOLD])
        if before[GOLD].get(key) != incremental[GOLD].get(key)
    }
    print(f"\nPartitions of {GOLD} that changed: {[str(k) for k in touched]}")
    assert touched == {KeyPredicate.of(dt=datetime(2026, 3, 1), region="eu")}

    # 6. The claim everything rests on: incremental == full.
    engine.full_rebuild(SILVER)
    engine.full_rebuild(GOLD)
    full = {ds: engine.fingerprints(ds) for ds in (SILVER, GOLD)}

    assert incremental == full, "incremental rebuild diverged from a full rebuild"
    print("\nIncremental rebuild is byte-identical to a full rebuild.")

    skipped = len(full[GOLD]) - len(touched)
    print(f"It rebuilt 1 of {len(full[GOLD])} partitions, skipping {skipped}.")

    engine.close()


if __name__ == "__main__":
    main()
