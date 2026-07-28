"""Grading the planner before trusting it.

Shadow mode runs a full rebuild alongside the plan and compares. Two numbers come
out: partitions skipped, and partitions wrongly called clean. The second must be
zero.

The second half deliberately feeds it a wrong plan, because a soundness checker
that cannot report a failure is decoration.

    python examples/02_shadow_mode.py
"""

from __future__ import annotations

from datetime import datetime

from examples_common import GOLD, RAW, SILVER, SPECS, build_warehouse

from fathom import KeyPredicate, Store, shadow
from fathom.ingest import ingest_engine


def main() -> None:
    # --- a sound plan ---------------------------------------------------------
    engine = build_warehouse()
    graph = ingest_engine(engine, specs=SPECS).graph

    engine.connect().execute("INSERT INTO raw.events VALUES ('2026-03-14', 'eu', 'u9', 99.0)")
    landed = KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
    plan = graph.invalidate({RAW: [landed]})

    # Ordering matters: the tables must still hold the pre-change build, so shadow
    # runs *before* anything is applied.
    with Store() as store:
        report = shadow.run(engine, plan, [SILVER, GOLD], store=store)

        print(report.summary())
        print()
        for result in report.results:
            print(f"  {result}")

        assert report.is_sound
        assert report.missed_total == 0

        summary = store.shadow_summary()
        print(f"\nRecorded: {summary['runs']} runs, {summary['missed']} missed")

    engine.close()

    # --- a deliberately wrong plan -------------------------------------------
    print("\n" + "=" * 70)
    print("Now the same check against a plan that is wrong on purpose.\n")

    engine = build_warehouse()
    graph = ingest_engine(engine, specs=SPECS).graph
    engine.connect().execute("INSERT INTO raw.events VALUES ('2026-03-14', 'eu', 'u9', 99.0)")

    plan = graph.invalidate({RAW: [landed]})
    # Point at partitions that did not change, and drop the ones that did.
    plan.dirty[SILVER] = frozenset({KeyPredicate.of(dt=datetime(2026, 4, 2), region="us")})
    plan.dirty[GOLD] = frozenset({KeyPredicate.of(dt=datetime(2026, 4, 1), region="us")})

    report = shadow.run(engine, plan, [SILVER, GOLD])
    print(report.summary())

    assert not report.is_sound, "the checker failed to notice a wrong plan"
    print("\nThe checker caught it. That is the property that makes the sound")
    print("result above worth anything.")

    engine.close()


if __name__ == "__main__":
    main()
