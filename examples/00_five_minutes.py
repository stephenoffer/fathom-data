"""The whole idea, in one file, with nothing installed but the base package.

Start here. There is no warehouse, no object store, and no config file — just a
graph built in memory and a plan computed over it, so you can see what this library
actually does before deciding whether to point it at anything real.

    python examples/00_five_minutes.py

Everything after this adds realism, not concepts:

    01_local_lakehouse.py  the same loop against a real DuckDB warehouse
    02_shadow_mode.py      how to decide whether to trust the planner
"""

from __future__ import annotations

from datetime import datetime

from fathom import DatasetId, Graph, KeyPredicate, PartitionSpec
from fathom.core.partitions import PartitionMapping, Passthrough, TimeWindow

# ---------------------------------------------------------------------------
# 1. Say how each table is partitioned.
#
# This is the one thing that cannot be inferred and has to be declared. Snowflake
# has no partitions to read; Delta records the column names but not the grain. A
# `dt` holding days and a `dt` holding months are indistinguishable in metadata,
# and guessing wrong makes every downstream answer wrong.
#
# `PartitionSpec.parse` is the compact form. In a project these live in fathom.yml.
# ---------------------------------------------------------------------------

DAILY = PartitionSpec.parse("dt:day, region")
MONTHLY = PartitionSpec.parse("dt:month, region")

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.events")
GOLD = DatasetId("duckdb", "gold.monthly")
ROLLING = DatasetId("duckdb", "gold.rolling_7d")


# ---------------------------------------------------------------------------
# 2. Say what each edge does to partitions.
#
# This is the part that makes a plan smaller than "everything downstream". Each
# edge answers one question: if this input partition is dirty, which output
# partitions are dirty?
# ---------------------------------------------------------------------------


def build_graph() -> Graph:
    graph = Graph()

    # raw -> silver: a straight transform. Same day in, same day out; the EU rows
    # produce EU rows and never touch the US ones.
    graph.connect(
        RAW,
        SILVER,
        evidence="sql",
        src_spec=DAILY,
        dst_spec=DAILY,
        mapping=PartitionMapping.identity(DAILY),
        columns=[("amount", "amount")],
    )

    # silver -> gold.monthly: a rollup. One dirty day dirties the month holding it,
    # and no other month.
    graph.connect(
        SILVER,
        GOLD,
        evidence="sql",
        dst_spec=MONTHLY,
        mapping=PartitionMapping.rollup(DAILY, MONTHLY),
        columns=[("amount", "revenue")],
    )

    # silver -> gold.rolling_7d: a trailing window. One dirty day dirties that day
    # and the six after it, because each of those windows read it.
    #
    # Written as a length rather than as offsets on purpose. `(0, 6)` typed by hand
    # is where the off-by-one lives, and an off-by-one here under-invalidates
    # silently — which is the one failure mode this library exists to prevent.
    graph.connect(
        SILVER,
        ROLLING,
        evidence="sql",
        dst_spec=DAILY,
        mapping=PartitionMapping.of(
            dt=TimeWindow.trailing("dt", 7, "day"),
            region=Passthrough("region"),
        ),
        columns=[("amount", "amount_7d")],
    )
    return graph


def main() -> None:
    graph = build_graph()

    print("The graph")
    print("---------")
    print(graph.describe())
    print()

    # Read an edge back in English. This is worth doing in review: the mapping is
    # the only part of the graph nobody can verify by reading the SQL that produced
    # it, so it is the part worth saying out loud.
    print("What the rolling-window edge claims")
    print("-----------------------------------")
    rolling_edge = next(e for e in graph.out_edges(SILVER) if e.dst == ROLLING)
    print(rolling_edge.explain())
    print()

    # -----------------------------------------------------------------------
    # 3. One day of one region is redelivered. What has to be rebuilt?
    # -----------------------------------------------------------------------

    dirty_day = KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
    plan = graph.invalidate({RAW: [dirty_day]})

    print("A vendor redelivers 2026-03-14, for the EU only")
    print("-----------------------------------------------")
    print(plan.summary())
    print()
    print(f"{plan.total_partitions} partitions across {len(plan)} datasets.")
    print()

    # The assertions are the point of the example. Each one is a claim the README
    # makes, checked rather than described.

    # The rollup gets exactly one month — March — and not the other 35.
    months = plan.partitions(GOLD)
    assert len(months) == 1, months
    assert next(iter(months)).get("dt") == datetime(2026, 3, 1)

    # The trailing window gets seven days: the dirty one and the six that read it.
    assert len(plan.partitions(ROLLING)) == 7

    # Every single one of them is still scoped to the EU. The US partitions were
    # never touched, and nothing rebuilds them.
    for dataset in plan:
        for key in plan.partitions(dataset):
            assert key.get("region") == "eu", (dataset, key)

    # Build order is dependency order, so a runner can just iterate it.
    assert list(plan) == [RAW, SILVER, GOLD, ROLLING] or list(plan) == [
        RAW,
        SILVER,
        ROLLING,
        GOLD,
    ]

    # -----------------------------------------------------------------------
    # 4. What happens when nothing proved a relationship.
    #
    # The honest answer is "the whole thing", and it is what every unparseable
    # query, opaque UDF, and undeclared spec falls back to. It costs compute and
    # never costs correctness.
    # -----------------------------------------------------------------------

    unproven = Graph()
    unproven.connect(RAW, SILVER, evidence="declared", src_spec=DAILY, dst_spec=DAILY)
    widened = unproven.invalidate({RAW: [dirty_day]})

    print("The same change, over an edge nothing could prove")
    print("-------------------------------------------------")
    print(widened.explain(SILVER))
    print()

    assert SILVER in widened.widened

    print("Next: 01_local_lakehouse.py runs this against a real warehouse and")
    print("      checks the incremental result matches a full rebuild exactly.")


if __name__ == "__main__":
    main()
