"""Locating and destroying a subject's data.

Two things this shows that are easy to get wrong:

1. A derived aggregate has no `user_id` column. There is nothing to delete there —
   the only correct action is to re-derive it from already-erased upstream.
2. Which makes ordering load-bearing. Rebuilding the aggregate before erasing the
   source reads data that still contains the subject, produces identical output,
   and reports success.

    python examples/04_erasure.py
"""

from __future__ import annotations

import json
from datetime import datetime

from examples_common import GOLD, RAW, SILVER, SPECS, build_warehouse

from fathom import Capabilities, ChangeSource, ErasureMode, KeyPredicate, LineageSource
from fathom.govern.erasure import ErasureRequest, apply_erasure, plan_erasure, unerasable
from fathom.ingest import ingest_engine


def caps(mode: ErasureMode) -> Capabilities:
    return Capabilities(
        lineage=LineageSource.QUERY_LOG, change=ChangeSource.WATERMARK, erasure=mode
    )


SALT = "org-secret"  # per-organization secret; keeps the proof digest non-reversible


def main() -> None:
    engine = build_warehouse()
    graph = ingest_engine(engine, specs=SPECS).graph

    request = ErasureRequest(
        subject="u1",
        key_column="user_id",
        origin=RAW,
        partitions=frozenset(
            {
                KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu"),
                KeyPredicate.of(dt=datetime(2026, 3, 15), region="eu"),
            }
        ),
        reference="DSR-77",
    )

    # --- 1. a target on WORM storage cannot be erased -------------------------
    blocked = {
        RAW: caps(ErasureMode.REWRITE),
        SILVER: caps(ErasureMode.REWRITE),
        GOLD: caps(ErasureMode.NONE),
    }
    plan = plan_erasure(graph, request, capabilities=blocked)

    print(plan.summary())
    print(f"\nWill survive this request: {[str(d) for d in unerasable(plan)]}")
    assert not plan.is_complete

    # --- 2. everything erasable ----------------------------------------------
    print("\n" + "=" * 70)
    everywhere = {ds: caps(ErasureMode.REWRITE) for ds in (RAW, SILVER, GOLD)}
    plan = plan_erasure(graph, request, capabilities=everywhere)

    print("\nTarget order (sources first, so aggregates re-derive from erased data):")
    for target in plan.targets:
        print(f"  {target.dataset}")

    before = {row[0:2]: row[2] for row in engine.rows(GOLD)}
    print(f"\nBefore: March/eu revenue = {before[(datetime(2026, 3, 1), 'eu')]}")

    # --- 3. dry run changes nothing ------------------------------------------
    proof = apply_erasure(plan, {RAW: engine, SILVER: engine, GOLD: engine}, salt=SALT)
    assert not proof.executed
    print("Dry run (the default) changed nothing.")

    # --- 4. execute ----------------------------------------------------------
    proof = apply_erasure(
        plan,
        {RAW: engine, SILVER: engine, GOLD: engine},
        dry_run=False,
        salt=SALT,
    )

    after = {row[0:2]: row[2] for row in engine.rows(GOLD)}
    print(f"After:  March/eu revenue = {after[(datetime(2026, 3, 1), 'eu')]}")
    print("  (u1's 10.0 and 30.0 are gone; u2's 5.0 survives. The aggregate was")
    print("   re-derived from erased upstream, not deleted from directly.)")

    assert all(row[2] != "u1" for row in engine.rows(SILVER))

    # Untouched partitions stay untouched.
    april = (datetime(2026, 4, 1), "eu")
    assert before[april] == after[april]
    print(f"April/eu is untouched at {after[april]}.")

    # --- 5. the proof ---------------------------------------------------------
    body = json.loads(proof.to_json())
    print("\nProof artifact:")
    print(f"  subject_digest {body['subject_digest'][:24]}…  (never the raw value)")
    print(f"  reference      {body['reference']}")
    print(f"  complete       {body['complete']}")
    print(f"  digest         {body['digest'][:24]}…  (SHA-256 over the body)")
    for entry in body["entries"]:
        print(f"    {entry['dataset']}: {entry['status']}")

    assert "u1" not in proof.to_json()
    assert proof.complete

    engine.close()


if __name__ == "__main__":
    main()
