"""The three questions a lineage graph alone cannot answer.

Every other example stays inside the warehouse. This one is about the boundary:

    1. Which partitions should exist and never arrived?
    2. Who actually reads this table, and what has it cost?
    3. What have we already published from it?

None of the three is answerable from the graph, because all three are about
observed history — arrivals, reads, and runs — which nothing records unless you
record it. The store keeps those three logs; this example writes to them and then
asks the questions.

The last section is the point of the whole thing: it deliberately shows a table that
is expensive and unread and a table that is quiet but feeds a regulatory filing, and
the tool declines to recommend deleting either.

    python examples/06_worth_keeping.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from examples_common import DAY_REGION, GOLD, RAW, SILVER, SPECS, build_warehouse

from fathom import Grain, KeyPredicate, Store, sinks
from fathom.graph.plan import lifetime
from fathom.graph.plan.cost import CostModel
from fathom.ingest import ingest_engine
from fathom.observe import completeness, usage

NOW = datetime.now(UTC)


def main() -> None:
    engine = build_warehouse()
    graph = ingest_engine(engine, specs=SPECS).graph
    engine.close()

    with Store() as store:
        store.save_graph(graph)

        # --- 1. what should have landed --------------------------------------
        #
        # The source delivers one partition per region per day. Two days in the
        # middle of March never arrived for `eu`, and nothing downstream can see
        # that: a missing partition has no profile to drift and no rows to fail an
        # expectation. It is indistinguishable from a day with no sales.
        for day in (13, 14, 15, 16, 17):
            for region in ("eu", "us"):
                if region == "eu" and day in (14, 15):
                    continue  # the outage
                store.record_arrival(
                    completeness.Arrival(
                        dataset=RAW,
                        key=KeyPredicate.of(dt=datetime(2026, 3, day), region=region),
                        observed=datetime(2026, 3, day, 2, tzinfo=UTC),
                        digest=f"sha256:{day}{region}",
                    )
                )

        gaps = completeness.report(
            RAW,
            DAY_REGION,
            store.present_partitions(RAW),
            start=datetime(2026, 3, 13),
            end=datetime(2026, 3, 17),
        )
        print("=== 1. partitions that should exist and do not")
        print(gaps.summary())
        assert not gaps.is_complete
        assert len(gaps.runs) == 1, "the two missing eu days are one incident, not two"
        assert gaps.runs[0].count == 2

        # A partition rewritten with different contents is a restatement, not a
        # replay, and that is what silently double-counts revenue downstream.
        late = completeness.Arrival(
            dataset=RAW,
            key=KeyPredicate.of(dt=datetime(2026, 3, 13), region="eu"),
            observed=datetime(2026, 3, 20, 6, tzinfo=UTC),
            digest="sha256:corrected",
        )
        store.record_arrival(late)
        restated = completeness.restatements(store.arrivals(RAW))
        print(f"\nrestatements: {len(restated)} partition(s) rewritten with new contents")
        assert len(restated) == 1

        lag = completeness.arrival_lag(late, field_name="dt", grain=Grain.DAY)
        assert lag is not None and lag > timedelta(days=6)

        # --- 2. who reads it, and what it cost -------------------------------
        #
        # gold.monthly is read by people. silver.events is touched only by the job
        # that maintains it, which is being maintained rather than used.
        store.record_reads(
            [
                usage.ReadEvent(GOLD, "ana", NOW - timedelta(days=1), kind="dashboard"),
                usage.ReadEvent(GOLD, "finance_team", NOW - timedelta(days=3)),
                usage.ReadEvent(SILVER, "airflow_worker", NOW - timedelta(days=1)),
            ]
        )
        for dataset, partitions in ((RAW, 40), (SILVER, 40), (GOLD, 400)):
            for back in range(1, 30):
                store.record_run(
                    lifetime.RunRecord(dataset, NOW - timedelta(days=back), partitions=partitions)
                )

        window = timedelta(days=90)
        stats = store.usage(window=window)
        print("\n=== 2. who reads each dataset")
        for _, found in sorted(stats.items(), key=lambda kv: str(kv[0])):
            print(f"  {found.summary()}")
        assert stats[SILVER].human_principals == set(), "only a scheduler touches silver"

        totals = lifetime.accumulate(store.runs(), CostModel(price_per_partition=0.05))
        print("\n  lifetime cost:")
        for total in lifetime.most_expensive_lifetime(totals):
            print(f"    {total.summary()}")

        # --- 3. what has already been published ------------------------------
        #
        # Declared, not discovered: no BI tool exposes its queries uniformly, and a
        # guessed dependency names the wrong people in a restatement notice.
        sinks.record_publication(graph, sinks.dashboard("revenue/exec", tool="looker"), [GOLD])
        sinks.record_publication(graph, sinks.filing("10-K/2026", regulator="sec"), [GOLD])

        print("\n=== 3. what a restatement of raw.events would touch")
        print(sinks.notice_text(graph, RAW, reason="the eu outage above"))
        assert sinks.has_regulatory_exposure(graph, RAW)

        # --- and what the tool refuses to conclude ---------------------------
        #
        # silver.events is expensive and no person has read it in 90 days. It is
        # still not a delete candidate, because gold.monthly is downstream and is
        # read — and even if it were not, "no reads observed" is not "no reads".
        # `people_only` matters here: silver.events is touched by its own scheduler
        # and by nobody else. Counting that as a read would make an intermediate
        # table look used by the very job that maintains it.
        reads = usage.read_counts(stats, people_only=True)
        findings = lifetime.value(totals, reads, threshold=50.0, window=window)
        print("\n=== what this does and does not conclude")
        print(lifetime.summarize(findings))

        candidates = {c.dataset for c in usage.retirement_candidates(graph, stats, window=window)}
        assert SILVER not in candidates, "one hop from something read is not unused"
        assert GOLD not in candidates

        verdicts = {f.dataset: f.verdict for f in findings}
        assert verdicts[GOLD] is lifetime.Verdict.EARNING
        assert verdicts[SILVER] is lifetime.Verdict.REVIEW
        print(
            "\nsilver.events is flagged for review, not deletion — and the summary "
            "above states why that distinction is not pedantry."
        )


if __name__ == "__main__":
    main()
