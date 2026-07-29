# What this is for

A catalog of the business problems `fathom` exists to solve, with an honest status on
each one. It is written for three readers: someone deciding whether this is worth
adopting, someone deciding what to build next, and someone six months from now
wondering why a module exists.

**Status means one thing only:**

| | |
|---|---|
| **Solved** | a documented entry point answers it end to end, and a test covers it |
| **Partial** | answered under stated conditions, or answered with a known weakness |
| **Gap** | no capability today; listed because the problem is real, not because we handle it |

A `Partial` or `Gap` here is a commitment to say so out loud rather than a thing to
be embarrassed about. The gaps are the roadmap; see [improvements.md](improvements.md)
for what is being done about them.

Problems are numbered `P1`…`P128` and referenced by number elsewhere in the docs.

---

## Part I — The rebuild bill

Every organization running a warehouse rebuilds far more than changed, because
proving what *didn't* change is harder than rebuilding it. This is the single
largest recurring cost in most data platforms and the one nobody can itemize.

### P1 — Nightly full rebuilds of tables where nothing changed
**Who feels it** Whoever signs the warehouse invoice.
**Today** The DAG runs every model every night because there is no proof that a model's
inputs are unchanged. A 400-model project rebuilds 400 models to propagate two.
**fathom** `plan` seeds from what adapters report changed, propagates through partition
mappings, and returns only reachable partitions. Datasets not reached are not in the plan.
**Status** Solved — `graph.Graph.invalidate`, `cli plan`.

### P2 — One late-arriving source day triggers a full-history backfill
**Who feels it** Data engineering, at 3am.
**Today** A vendor redelivers 2024-03-14. Nobody can bound what that touches, so the
safe move is reprocessing all history.
**fathom** A dirty key propagates through `TimeWindow` mappings to exactly the downstream
buckets that read it — one day downstream, one month in a rollup, seven days through a
trailing window.
**Status** Solved — `core.partitions.apply`, `compose`.

### P3 — Rollups recomputed in full when one day changed
**Who feels it** Anyone with daily→monthly aggregation.
**Today** The monthly table is rebuilt for all 36 months because the grain relationship
is not written down anywhere a machine can read.
**fathom** `TimeWindow(src, 0, 0, DAY, MONTH)` states it. One dirty day resolves to one
dirty month.
**Status** Solved — `core.partitions.TimeWindow`.

### P4 — Trailing-window aggregates rebuilt wholesale
**Who feels it** Anyone computing 7-day or 28-day rolling metrics.
**Today** A rolling window means "any day could affect any other day", so the whole
table is rebuilt.
**fathom** `TimeWindow("dt", 0, 6, DAY, DAY)` states the reach exactly: one dirty day
taints the six days after it and nothing else.
**Status** Solved — property-tested in `tests/test_partitions.py`.

### P5 — Blast radius of a source change is guessed
**Who feels it** Whoever is asked "if we reload this table, what breaks?"
**Today** Somebody greps the repo and hopes.
**fathom** `graph.query.blast_radius` and `descendants` give the reachable set; the plan
gives the partition-level answer.
**Status** Solved — `graph.query`.

### P6 — Backfills are over-scoped and run for days
**Who feels it** Analytics engineering, and everyone waiting on the cluster.
**Today** Backfill windows are chosen by superstition and padded for safety.
**fathom** Seed the planner with the true backfill range; the plan bounds every
downstream dataset's affected partitions rather than inheriting the padding.
**Status** Solved.

### P7 — Reprocessing after a bug fix has no bounded scope
**Who feels it** Whoever shipped the bug.
**Today** "The transform was wrong since March" becomes an unbounded rebuild because
nobody can enumerate what consumed the bad output.
**fathom** Seed with the affected partitions of the corrected dataset and plan forward.
**Status** Solved.

### P8 — Warehouse spend cannot be attributed to a dataset
**Who feels it** FinOps, platform leadership.
**Today** The bill is one number. Per-model cost requires manual query-tag hygiene
nobody maintains.
**fathom** `graph.plan.cost` distributes plan cost across datasets from declared
rates, and `graph.plan.billing` checks those rates against the actual invoice.
`reconcile` compares modelled against billed over the same window; `calibrate` returns
a corrected model.
**Status** Solved for the aggregate, and explicit about the rest. Warehouse billing is
per-warehouse or per-job, not per-dataset, so per-dataset figures are produced **only
for rows the bill actually attributes** — apportioning an unattributed total would
hand the cost model's own assumptions back as evidence for themselves. `calibrate`
refuses below fourteen periods, because a bias factor from three days is a rounding
artifact with a decimal point.

### P9 — Nobody can state what incremental processing saved
**Who feels it** The person who has to justify the platform team's headcount.
**Today** "We think it helps."
**fathom** `cost.savings` and `estimate_full_rebuild` give the counterfactual: what a
full rebuild would have cost against what the plan costs.
**Status** Solved — `graph.plan.cost`.

### P10 — Orchestrator DAGs are hand-maintained and drift from reality
**Who feels it** Whoever debugs the 2am failure caused by a missing edge.
**Today** Airflow dependencies are written by hand and diverge from the SQL.
**fathom** `report.orchestrators` generates the DAG file itself — Airflow, Dagster,
or Prefect — via `fathom dag --flavor airflow --out dags/rebuild.py`.
**Status** Solved. Nothing imports the orchestrator: the output is a file you commit,
not a runtime binding. Wave boundaries are emitted as real dependencies rather than
ordering hints, and partition keys travel with each task — a generated DAG that loses
either is worse than none, since the first rebuilds from unwritten inputs and the
second quietly full-rebuilds. Intervals, retries, and alerting are deliberately absent;
a generator guessing at them would be wrong in a way that looks authoritative.

### P11 — Rebuild ordering is wrong, so downstream runs before upstream
**Who feels it** Everyone, as intermittent wrong numbers.
**Today** Ordering is whatever the DAG author wrote.
**fathom** `InvalidationPlan.order` is a topological order over the affected subgraph;
`schedule.waves` groups it into levels that can run in parallel.
**Status** Solved.

### P12 — No parallelism plan, so rebuilds run serially
**Who feels it** Anyone whose backfill takes 14 hours.
**Today** Parallelism is a guess set once in a config file.
**fathom** `schedule.waves` gives the maximum safe parallelism per level;
`Schedule.max_parallelism` reports it.
**Status** Solved.

### P13 — Batch sizes are arbitrary: queries either time out or are too small to matter
**Who feels it** Whoever tunes the cluster.
**Today** A magic number in a config.
**fathom** `schedule.batch_partitions` groups contiguous partitions into batches, and
`rebalance` re-splits to a maximum size.
**Status** Solved.

### P14 — Carbon reporting for data pipelines is unanswerable
**Who feels it** Sustainability reporting, increasingly mandatory.
**Today** Estimated from total cloud spend with a blanket coefficient.
**fathom** `cost.carbon` derives an estimate from the same partition and byte counts the
plan already produces, so avoided compute shows up as avoided emissions.
**Status** Partial — coefficients are user-supplied; the value is the itemization, not
the constant.

### P15 — The cost of a plan is unknown until after it runs
**Who feels it** Anyone who has kicked off a backfill and watched the bill.
**Today** Run it and find out.
**fathom** `cost.estimate_plan` prices the plan before execution; `budget_exceeded`
gates it.
**Status** Solved.

### P16 — Budget overruns are discovered on the invoice
**Who feels it** Finance, a month late.
**Today** Monthly surprise.
**fathom** `cost.budget_exceeded` fails a plan that exceeds a declared budget, before
the compute is spent.
**Status** Solved.

### P17 — Continuous profiling costs more than the pipeline it watches
**Who feels it** Anyone who tried to run a data-observability tool at full scan.
**Today** Nightly full-table scans to compute distributions. This is why continuous
profiling gets switched off.
**fathom** Two compounding reductions: profile only the partitions the graph says
changed, and read Parquet footers rather than data pages.
**Status** Solved — `observe.profile.profile_parquet`.

### P18 — Full-scan profiling on a warehouse pays egress and compute twice
**Who feels it** Lakehouse teams.
**Today** Pull the data out to profile it.
**fathom** `Pushdown` capabilities let an adapter return sketches, quantiles, or approx
distinct counts computed in place.
**Status** Partial — the capability is declared and honoured in the type system; only
some adapters implement real pushdown, and the rest fall back to footers.

### P19 — Cost models cannot be compared before committing to one
**Who feels it** Platform teams choosing a chargeback scheme.
**fathom** `cost.compare_models` evaluates the same plan under several cost models.
**Status** Solved.

### P20 — Annual savings from a one-night measurement are extrapolated by hand
**Who feels it** Whoever writes the business case.
**fathom** `cost.annualized` projects a measured per-run saving across a run frequency.
**Status** Solved.

---

## Part II — Trust, correctness, and incident response

The second-largest cost after compute is the hour-by-hour cost of not being sure the
numbers are right, and the cost of finding out why when they aren't.

### P21 — A drift alert has no cause attached
**Who feels it** The on-call analytics engineer.
**Today** "`revenue` moved 8%" arrives at 7am with no next step. Diagnosis is a manual
walk upstream through SQL nobody wrote.
**fathom** `ai.attribution.attribute` walks the column graph upstream, checks which
ancestors also drifted in the same window, and ranks them.
**Status** Solved — `ai.attribution.Diagnosis`, `blame_report`.

### P22 — Alert fatigue: per-column thresholds fire constantly
**Who feels it** Everyone, until they mute the channel.
**Today** Static thresholds on 40,000 columns produce noise, so alerts get ignored, so
the one real incident is missed.
**fathom** Attribution collapses a storm of correlated alerts into one root cause;
`attribution.root_causes` returns only causes with no drifted ancestor of their own.
**Status** Solved.

### P23 — Data quality expectations are hand-written and rot
**Who feels it** Whoever inherits the test suite.
**Today** Someone writes 200 expectations during onboarding. Nobody updates them. They
fail for legitimate reasons and get disabled one by one.
**fathom** `observe.quality.learn` generates a suite from observed profile history, so
the baseline is what the data actually did rather than what someone assumed in 2023.
**Status** Solved.

### P24 — A schema change breaks downstream consumers silently
**Who feels it** The consumer, in production.
**Today** A column is renamed; a downstream dashboard shows nulls for a week.
**fathom** `observe.schema.diff_schemas` and `breaking_schema_changes` classify adds,
removes, and retypes, and `is_breaking` gates a merge.
**Status** Solved.

### P25 — Freshness is measured per table, not transitively
**Who feels it** Anyone who has trusted a "last updated 5 minutes ago" badge.
**Today** A table rebuilt five minutes ago is reported fresh even though its input is
four days stale, so the badge is worse than nothing — it is confidently wrong.
**fathom** `observe.freshness.effective_age` takes the maximum age across the upstream
closure. A dataset is exactly as fresh as its stalest ancestor.
**Status** Solved — this is the module's whole reason to exist.

### P26 — SLA violations are reported by consumers, not producers
**Who feels it** The producing team's credibility.
**fathom** `freshness.sla_violations` and `worst_offenders` evaluate declared SLAs
against the transitive age.
**Status** Solved.

### P27 — "Why is this table late?" takes an hour to answer
**Who feels it** On-call.
**fathom** `freshness.blame` names the specific upstream dataset responsible for the
effective age, and `freshness_path` gives the chain.
**Status** Solved.

### P28 — Row-count collapse and null-rate spikes go undetected
**Who feels it** Downstream consumers.
**fathom** `observe.profile.drift` compares against profile history; `quality.learn`
derives the bounds.
**Status** Solved.

### P29 — Nobody can trust a new incremental tool's decisions
**Who feels it** The team asked to adopt one.
**Today** Adopting an incremental planner means betting correctness on a vendor's
claim. Most teams reasonably refuse.
**fathom** Shadow mode: run the planner alongside the existing full rebuild and grade
it. `missed` — partitions called clean that a full rebuild proved dirty — must be zero,
and the CLI exits non-zero the moment it isn't. Zero risk, because the full rebuild
happens either way.
**Status** Solved — `observe.shadow`, and the strongest adoption argument in the product.

### P30 — A planner's miss rate is never measured, only asserted
**Who feels it** Whoever gets blamed for the stale dashboard.
**fathom** `ShadowReport.missed_total` is persisted per run, so the evidence accumulates
over weeks rather than being re-argued.
**Status** Solved.

### P31 — The same table has three names across three systems
**Who feels it** Anyone joining Spark lineage to Trino lineage.
**Today** `s3a://bucket/x`, `s3://bucket/x`, `/dbfs/mnt/x`, and `catalog.schema.x` are
four nodes in four tools and one table in reality.
**fathom** `core.ids.normalize` collapses them to one `DatasetId` following the
OpenLineage naming convention, with `AliasRegistry` for the cases convention cannot
reach.
**Status** Solved.

### P32 — No baseline for "is this normal for a Tuesday"
**Who feels it** Anyone with weekly seasonality.
**Today** A Monday-vs-Sunday comparison fires every week.
**fathom** Profile history is stored per partition over time.
**fathom** `observe.seasonal.learn_seasonal` buckets observations by a cycle —
day-of-week, hour-of-day, day-of-month, month-of-year — and learns a band per bucket, so
1,000 rows is normal on Monday and an anomaly on Saturday.
**Status** Solved — `fathom seasonal --dataset X --cycle day_of_week`, learned from
the profile history already in the store and bucketed by the *partition's* own moment
so a backfill does not fail its own checks. It refuses to model a bucket with too few
observations rather than inventing a band from two Sundays. `strength` reports whether the cycle explains
anything at all, so seasonal modelling stays a decision rather than a default.

### P33 — Silent data loss: a partition that was never written
**Who feels it** Whoever notices the gap three months later.
**Today** A missing partition looks identical to a partition with no data.
**fathom** `observe.completeness.report` enumerates what should exist from the
partition spec and a range, compares it against what does, and collapses the difference
into contiguous runs — seven consecutive missing days is one incident, not seven alerts.
**Status** Solved — the only check in `observe` that can see a partition which never
arrived, since it has no profile to drift and no rows to fail an expectation. Runs are
computed per value slice, so `region=eu` missing three days and `region=us` missing one
are reported as the two separate incidents they are.

### P34 — Late and duplicate data is double-counted
**Who feels it** Finance, in a restatement.
**fathom** `observe.completeness` records `Arrival`s and splits repeats into `replays`
(identical digest, harmless) and `restatements` (contents changed). `late_arrivals`
measures lag from where the bucket closed, so a daily partition written at 02:00 the next
morning is two hours late rather than twenty-six.
**Status** Solved — a repeat arrival with an *unknown* digest is classified as a
restatement, never a replay, because treating "cannot tell" as "unchanged" is exactly
what lets a silent double-count through.

### P35 — Cardinality explosion in a join key degrades a pipeline for weeks
**Who feels it** Whoever eventually profiles the join.
**Who feels it** Finance, in a restatement, or nobody until the monthly close.
**Today** A join is the only operation that can make a table *larger* than its inputs
and the only one that silently drops rows. Both come from the key's shape changing,
and neither raises. `drift` sees the symptom — row count moved — and reports it as
thirty findings with no cause.
**fathom** `observe.joins` names the cause from the same profiles at no extra cost.
`uniqueness_lost` catches the key that stopped being unique, which is what doubles a
revenue total. `orphan_floor` proves how many keys *cannot* match. `amplification`
reports the arithmetic signature of fan-out.
**Status** Solved for what a profile can prove, and explicit about the rest. Key
*overlap* needs a scan, so nothing here estimates output rows — `orphan_floor` gives a
lower bound that can prove rows will be dropped and can never prove none will be. A
clear result says "no risk proven", never "safe".

### P36 — Refactoring a pipeline silently changes its output
**Who feels it** The reviewer who approved it.
**Today** `diff` compares the graph, and the graph is identical. `drift` compares
against yesterday, and yesterday is now the new version. Shadow mode compares the
planner's *decisions*. Nobody has the one comparison that matters: this build against
the build the old code would have produced.
**fathom** `observe.regression.compare_outputs` compares fingerprints partition by
partition, and `explain` turns a digest mismatch into row counts, null rates, and
ranges — a digest difference alone is the least actionable finding a review can get.
**Status** Solved, with two refusals. It reports **changed**, never *wrong*: a refactor
that fixes a bug is supposed to change the output, and deciding which is which is not
the tool's call — `intended` lets a reviewer accept changes, and accepted ones are
counted rather than hidden. And a clean result is refused below a coverage floor,
because comparing three partitions out of four hundred proves nothing about the rest.
`is_regression` is deliberately not the negation of `is_clean`: an inconclusive report
is neither.

---

## Part III — Change management, discovery, and impact

### P37 — A pull request's lineage impact is invisible in review
**Who feels it** The reviewer, who approves it anyway.
**Today** A diff shows SQL. It does not show that this SQL feeds 40 downstream tables
and one regulatory report.
**fathom** `graph.diff.diff_graphs` compares the graph before and after, and
`review_comment` renders it for a PR.
**Status** Solved.

### P38 — Narrowing a dependency edge serves stale data forever
**Who feels it** Everyone, silently.
**Today** Someone tightens a window from 7 days to 1 as an optimization. Six days of
downstream data stop being invalidated. Nothing fails; the numbers are just wrong.
**fathom** `GraphDiff.is_safe` is false when an edge is removed or a mapping narrows.
Widening always passes; narrowing must be justified. This is the single most valuable
merge gate in the product, because the failure it prevents is otherwise undetectable.
**Status** Solved — `graph.diff.mapping_narrowed`.

### P39 — Dropping a column breaks consumers nobody could enumerate
**fathom** `graph.query.column_descendants` walks column-level edges downstream.
**Status** Partial — only as complete as column lineage coverage, which is native on
Snowflake and Databricks, parsed where SQL parses, and absent for Ray/Dask/Beam.
`metrics.coverage().column_ratio` reports honestly how much is covered.

### P40 — Deprecating a dataset requires knowing every consumer
**fathom** `graph.query.descendants`, plus `metrics.most_depended_on`.
**Status** Solved.

### P41 — Migration impact analysis is a two-week manual project
**fathom** `graph.query.upstream_subgraph` / `downstream_subgraph` scope it; `selectors`
expresses the scope in one string.
**Status** Solved.

### P42 — Orphaned datasets accumulate and nobody dares delete them
**fathom** `graph.query.isolated` finds structural orphans, which are rare because a
dead chain still has edges. `observe.usage.retirement_candidates` finds the real case: a
dataset nothing read, whose descendants nothing read either.
**Status** Solved, with a caveat the module refuses to drop — nothing returns "unused",
only "no reads observed", and every answer carries the window it observed over. A table
read once a year for a filing looks identical to a dead one over thirty days, and
deleting it is the one mistake here that a rebuild cannot undo.

### P43 — Circular dependencies are discovered at runtime
**fathom** `graph.query.cycles` (Tarjan) and `has_cycle`.
**Status** Solved.

### P44 — Nobody knows how much of the lineage graph is trustworthy
**Who feels it** Whoever is asked to bet a rebuild strategy on it.
**Today** A lineage catalog shows a picture. It does not distinguish an edge it proved
from an edge someone typed.
**fathom** `metrics.coverage` reports four ratios, and `Edge.evidence` records how each
edge was learned. `field_ratio` is close to the ceiling on how much of a rebuild any
plan can skip, which makes it the number to publish and the number to move.
**Status** Solved.

### P45 — Onboarding: what does this warehouse even contain
**fathom** `render.tree`, `graph_to_mermaid`, `metrics.graph_stats`, `namespaces`.
**Status** Solved.

### P46 — Every tool has a different selection syntax
**fathom** `graph.selectors` implements dbt's syntax — `+model+`, `2+model`, `@model`,
`tag:pii`, `ns:snowflake` — so the muscle memory transfers.
**Status** Solved.

### P47 — A dependency exists but contributes nothing to planning
**Who feels it** Whoever wonders why savings are low.
**Today** An edge with an unprovable mapping is in the picture and is worthless for
planning, but looks identical to a precise one.
**fathom** `metrics.bounded_edge_ratio` and `precision_ceiling` separate the two.
**Status** Solved.

### P48 — Graph health degrades invisibly over time
**fathom** `metrics.health_report` and `health_score`.
**Status** Solved.

### P49 — Which change caused the plan to get bigger?
**fathom** `graph.diff.diff_plans` compares two plans.
**Status** Solved.

---

## Part IV — Privacy and subject rights

### P50 — A subject access request takes a week of manual work
**Who feels it** Privacy operations, against a statutory clock.
**Today** Someone emails six teams and assembles a spreadsheet. The answer is a
best effort nobody can verify.
**fathom** `report.compliance.subject_access_report` walks the graph from the origin
dataset and reports every derived location, with its own gaps stated.
**Status** Solved.

### P51 — Erasing one subject is a rewrite-the-world operation
**Who feels it** Whoever owns the lakehouse.
**Today** Deleting one person's rows from a partitioned lake means rewriting every file
that might contain them, because nothing knows which files actually do.
**fathom** Partition scoping reduces the candidate set to the partitions the graph
proves can hold the subject; `govern.erasure.plan_erasure` enumerates targets per
dataset with the erasure mode each supports.
**Status** Solved — the economics of erasure are the reason partition mappings pay for
themselves twice.

### P52 — Object storage has no delete vector, so erasure means rewriting objects
**fathom** `ErasureMode.REWRITE` is modelled explicitly, distinct from
`DELETE_VECTOR` (Iceberg/Delta positional deletes) and `CRYPTO_SHRED`.
**Status** Solved — the mode is chosen per adapter from declared capability.

### P53 — WORM storage makes deletion physically impossible, and tools claim success anyway
**Who feels it** Whoever signs the attestation.
**Today** A deletion pipeline reports success over Object Lock storage where nothing was
deleted. That is a false attestation.
**fathom** `ErasureMode.NONE` is a refusal. The plan is incomplete, `is_complete` is
false, and `unerasable` names the datasets.
**Status** Solved — refusing loudly is the feature.

### P54 — Over-deletion is unrecoverable and nobody plans for it
**Who feels it** Whoever deleted the wrong subject.
**fathom** Erasure carries the mirrored invariant to planning: it may under-delete and
refuse, but must never over-delete. Hence dry-run by default and explicit opt-in to
apply.
**Status** Solved — `apply_erasure` requires deliberate confirmation.

### P55 — No proof that an erasure happened
**fathom** `ErasureProof` with `to_json` and a content `digest`.
**Status** Solved.

### P56 — Backups, replicas, and snapshots silently fall outside the erasure claim
**Who feels it** The regulator, later.
**Who feels it** The regulator, later.
**Today** "There may be copies elsewhere" is unfalsifiable. It cannot be reviewed,
cannot be closed out, and cannot be told apart from "we did not look".
**fathom** `govern.replicas` makes them **declared**. A snapshot in another account, a
read replica, a CSV a partner receives, a Kafka topic, a vendor's system — none is
derivable from lineage, because they are facts about an organization rather than about
its SQL. Declaring one turns the caveat into a checklist with owners and dates.
`proof_entries` puts them in the erasure proof by name and disposition.
**Status** Solved for *statement*, which is the honest ceiling — nothing here deletes
anything. A copy beyond the organization's control is `UNREACHABLE` even when somebody
attested to it, because an attestation about a vendor's system is a claim rather than
an action. A copy whose retention has not elapsed and which nobody attested is
`OUTSTANDING`, never assumed expired: time passing is not evidence. And `coverage`
reports how much of the estate was declared at all, so a low number reads as "we have
not mapped this" rather than "we are clean".

### P57 — Discovering PII across 40,000 columns is a project nobody finishes
**Who feels it** Governance, forever.
**Today** Hand-labelling. It never completes and is stale where it did.
**fathom** `govern.policy.infer` labels from profile evidence — value shape, ranges,
cardinality — not just column names.
**Status** Solved.

### P58 — Name-based PII detection is wrong in both directions
**Who feels it** Whoever tunes the false positives.
**Today** A column called `latitude` is assumed to be a coordinate; a column called
`c_47` is assumed to be nothing.
**fathom** Profile evidence overrides the name guess. A `latitude` whose values top out
at 4,000 is rejected before a human sees it.
**Status** Solved — the reason inference and profiling belong in one tool.

### P59 — A label applied upstream does not follow the data downstream
**fathom** `govern.policy.propagate` pushes labels along graph edges with confidence
damping per hop.
**Status** Solved.

### P60 — Personal data reaches an external sink and nobody notices
**fathom** `govern.policy.enforce` against a `SinkPolicy`; `Violation.is_unattributed`
separates "we know this is PII" from "we could not tell".
**Status** Solved.

### P61 — Consent purposes are collected and then never enforced
**Who feels it** The subject, and eventually the regulator.
**Today** Consent lives in a CRM. The data lake has no idea.
**fathom** `govern.consent.propagate_purposes` intersects purposes along edges —
restrictively, so a derived dataset permits only what every input permits —
and `unconsented_uses` finds violations.
**Status** Solved. The restrictive combination is a package-level invariant: getting it
backwards fails open.

### P62 — Data residency constraints are violated by a pipeline nobody reviewed
**fathom** `govern.consent.residency_violations` and `transfer_paths`.
**Status** Solved.

### P63 — Retention limits pass unnoticed
**fathom** `govern.consent.retention_violations` and `expired`.
**Status** Solved.

### P64 — Re-identification through joining two non-identifying datasets
**Who feels it** The privacy team, in a review they had no tooling for.
**fathom** `govern.reidentification.assess` bounds the average group size over a
dataset's quasi-identifiers from per-column distinct counts, and `linkage_risks` finds
pairs of datasets that are each defensible and jointly identifying because a shared
ancestor lets them be joined.
**Status** Partial, and precisely so. It can **prove a dataset is risky** and can never
prove one is safe: the bound needs the distinct count of the *combination*, which is a
scan this does not do. `is_clear` is named for the absence of proven risk, and the
summary says so in its own text rather than reading as a clean bill.

### P65 — Nobody can produce the list of systems holding personal data
**fathom** `report.compliance.personal_data_inventory`.
**Status** Solved.

---

## Part V — Regulatory evidence

The common shape: a regulator asks a question, the organization answers with a
document that was accurate on the day it was written. Everything in `report.compliance`
is derived from the graph, so it is accurate on the day it is read.

### P66 — Article 30 records of processing are maintained by hand
**fathom** `report.compliance.processing_record`, with `is_complete` reporting its gaps.
**Status** Solved.

### P67 — Assembling an audit evidence bundle takes weeks
**fathom** `report.compliance.audit_bundle`.
**Status** Solved.

### P68 — Cross-border transfer inventory does not exist
**fathom** `report.compliance.cross_border_summary`.
**Status** Solved.

### P69 — Nobody knows whether the organization could answer an audit today
**fathom** `report.compliance.readiness` lists what the graph cannot currently answer.
**Status** Solved — the value is that it enumerates what is missing rather than
scoring what is present.

### P70 — EU AI Act training data summaries must be written and kept current
**Who feels it** Whoever files them.
**Today** Written once by a human from interviews, stale within a sprint.
**fathom** `ai.training.training_data_summary` generates the prose from lineage, so it
cannot go stale independently of the system it describes.
**Status** Solved.

### P71 — Model cards are written once and never updated
**fathom** `ai.training.model_card` generated from the recorded run.
**Status** Solved.

### P72 — Source data licences are unknown by the time data reaches a model
**Who feels it** Legal, after training.
**fathom** `govern.licenses.propagate` combines licences most-restrictive-first along
edges; `effective_license` gives the answer at any node.
**Status** Solved.

### P73 — "May we use this commercially?" cannot be answered
**fathom** `licenses.commercial_use_allowed`.
**Status** Solved.

### P74 — "May we train on this?" cannot be answered
**fathom** `licenses.training_permitted`, and `consent.training_permitted_datasets` for
the consent side of the same question.
**Status** Solved.

### P75 — Attribution requirements are inherited and forgotten
**fathom** `licenses.attribution_required` and `attribution_manifest`.
**Status** Solved.

### P76 — Unlicensed data enters a training set undetected
**fathom** `licenses.unlicensed` treats unknown as restrictive rather than permissive.
**Status** Solved.

### P77 — Provenance gaps are smoothed over in generated documents
**Who feels it** Whoever signs a document that turns out to be wrong.
**fathom** `ai.training.provenance_gaps` and `BillOfMaterials.is_complete` state the
gaps in the artefact itself.
**Status** Solved — a generated record that looks complete because the generator omitted
what it did not know is worse than no record.

### P78 — No audit trail of who changed the dependency graph
**fathom** `graph.history` records an authored chain of revisions, each keeping the
`GraphDiff` from its predecessor. `narrowings_of(history, src, dst)` answers the question
an incident actually asks: six days of downstream data stopped being invalidated, so when
did that window shrink and who shrank it.
**Status** Solved for attribution, with one stated limit — revisions store diffs rather
than snapshots, so history can tell you exactly when and how an edge changed but cannot
hand you the graph as it stood last March. `digest_at` verifies a graph you still hold is
the one a revision described. `record` refuses a revision computed against a stale graph,
since that would attribute one person's change to whoever committed next.

---

## Part VI — AI and ML systems

The premise: a model, a feature view, a vector index, a prompt, and an eval set are
datasets. They are produced from named inputs, in slices that rebuild independently,
with a version history someone will need explained, and they can contain a person's
data. Give them `DatasetId`s and the existing machinery applies. There is no second
graph for ML, which matters because a second graph is a graph that disagrees with the
first one.

### P79 — "What was this model trained on?" takes a person and a week
**fathom** `ai.training.data_bill_of_materials` walks the closure and states its gaps.
**Status** Solved.

### P80 — Eval contamination invalidates a benchmark score and nobody checks
**Who feels it** Everyone downstream of the decision that score justified.
**Today** If the eval set and the training data share an ancestor, the score measures
memorization. Nobody has the lineage to check.
**fathom** Contamination is a reachability property. `ai.evals.contamination` checks it
in the graph and reports `clean`, `suspect`, or `contaminated`.
**Status** Solved — and it never rounds `suspect` up to `clean`.

### P81 — A corpus is re-embedded nightly whether or not it changed
**Who feels it** Whoever pays the embedding bill.
**fathom** `ai.vectors.reindex_plan` compares content digests and prices the difference.
**Status** Solved.

### P82 — An embedding model version change silently corrupts a vector store
**Who feels it** Users getting confidently wrong neighbours.
**Today** A partial reindex leaves vectors from two spaces in one index. Distances
between them are meaningless, and nothing errors.
**fathom** `vectors.requires_full_reindex` short-circuits a version change to a full
reindex, because vectors from two spaces are not comparable.
**Status** Solved.

### P83 — Deleted documents leave orphan vectors that keep being retrieved
**fathom** `vectors.orphan_vectors` and `deletion_targets`.
**Status** Solved.

### P84 — An erased subject remains retrievable through the vector store
**Who feels it** The subject, and the regulator.
**Today** Rows are deleted from the warehouse; the embeddings of those rows are not.
**fathom** `vectors.retrievable_after_erasure` names what survives.
**Status** Solved — one of the highest-value checks in the library, because the failure
is invisible from the warehouse side.

### P85 — Training/serving skew is discovered from degraded production metrics
**fathom** `ai.features.skew` compares offline and online profiles.
**Status** Solved.

### P86 — Target leakage into a feature view invalidates a model
**fathom** `ai.features.leaky_features` and `label_reaches_features`.
**Status** Solved.

### P87 — Point-in-time correctness violations in feature backfills
**fathom** `ai.features.point_in_time_violations`.
**Status** Partial — detects declared violations from recorded timestamps; it cannot see
a violation inside an opaque transform.

### P88 — Feature views serve stale values with no staleness signal
**fathom** `ai.features.is_stale`, `stale_views`, `serving_risks`.
**Status** Solved.

### P89 — After a data change, which models need retraining?
**fathom** `ai.training.retraining_plan` and `stale_models`.
**Status** Solved — `plan` over the model graph is the same verb as over tables.

### P90 — A training run cannot be reproduced because its inputs were not pinned
**fathom** `ai.training.InputPin`, `is_reproducible`, `unpinned_inputs`.
**Status** Solved.

### P91 — Prompts are edited in production with no version history
**fathom** `ai.prompts.PromptTemplate` treats prompts as versioned datasets with content
digests, `history`, and `rollback`.
**Status** Solved.

### P92 — A prompt variable interpolates personal data into a third-party endpoint
**Who feels it** The DPO, on discovery.
**Today** Prompt templates are strings in a repo. Nobody models the variables as data
flows, so PII reaching an external model through interpolation is invisible.
**fathom** `prompts.labels_reaching` binds variables to datasets and reports the labels
that arrive. Personal data reaching a model through a prompt variable is still a
transfer.
**Status** Solved.

### P93 — Nobody records what was in the context window
**fathom** `ai.rag.ContextManifest` records the retrievals, and `enforce_context` checks
what reached a third-party endpoint against the same sink policy the `label` verb uses.
**Status** Solved.

### P94 — RAG systems retrieve and pay for context the model never uses
**fathom** `rag.unused_context`, `wasted_cost`, `citation_coverage`.
**Status** Partial by design — the manifest records what was *retrieved*, not what the
model *used*, so `unused_context` is a cost signal and explicitly not an attribution
claim.

### P95 — No audit of what an autonomous agent actually read and wrote
**fathom** `ai.agents.AgentRun`, `datasets_read`, `datasets_written`, `record_agent_run`
puts the run in the same graph as everything else.
**Status** Solved.

### P96 — Agent data exfiltration paths are unmapped
**fathom** `agents.exfiltration_paths` and `egress_points`.
**Status** Solved.

### P97 — Agents accumulate permissions far beyond what they use
**fathom** `agents.least_privilege_gap`.
**Status** Solved.

### P98 — An agent touching a dataset for the first time is not flagged
**fathom** `agents.first_time_access` against run history.
**Status** Solved.

### P99 — "Which models still hold this person?"
**Who feels it** Whoever must answer a deletion request honestly.
**Today** Rows are deleted and the request is marked complete, over models trained on
those rows that are still serving traffic.
**fathom** `ai.unlearning.exposures` names every model and the route by which it
retains the subject.
**Status** Solved.

### P100 — Deleting rows is treated as discharging a deletion obligation
**fathom** `unlearning.is_deletion_sufficient` and `obligations` state what would
actually discharge it — retraining, or crypto-shredding data encrypted per subject.
**Status** Solved.

### P101 — Unlearning tools claim completeness they cannot deliver
**fathom** `unlearning.completeness_statement` says so in its first sentence rather than
emitting `complete: true` over a model still serving traffic. Approximate unlearning is
reported where a team says it is available, and never as complete.
**Status** Solved — this library does not claim to remove a subject from a model. It
claims to tell you exactly what you have not yet done.

### P102 — The cost of discharging an unlearning obligation is unknown
**fathom** `unlearning.estimate_retraining_cost`.
**Status** Solved.

### P103 — Eval regressions between model versions go unnoticed
**fathom** `ai.evals.regressions` and `compare_results`.
**Status** Solved.

### P104 — Eval results are reported against a model version they did not test
**fathom** `evals.stale_results`.
**Status** Solved.

### P105 — Holdout sets leak into training over time
**fathom** `evals.holdout_integrity`.
**Status** Solved.

### P106 — Embedding drift is invisible until retrieval quality collapses
**fathom** `vectors.embedding_drift`, `centroid_shift`, `norm_shift`.
**Status** Solved.

### P107 — Which AI assets does a data change actually affect?
**fathom** `ai.assets` gives every asset kind an identity and a spec, so `plan` reaches
models, indexes, prompts, and eval sets through the same traversal.
**Status** Solved.

### P108 — Vector index shards are rebuilt whole
**fathom** `vectors.stale_shards` returns shard-level partition keys.
**Status** Solved.

---

## Part VII — Platform operations

### P109 — Lineage stops at each system's boundary
**Who feels it** Anyone whose stack is more than one product.
**fathom** Three adapter surfaces — engines, catalogs, storage — feeding one graph over
one identity scheme.
**Status** Solved.

### P110 — Adding a new system means implementing a full lineage integration
**Today** All-or-nothing integrations are why the long tail is never covered.
**fathom** Adapters declare capabilities rather than implement everything. One reporting
`LIST_DIFF` and `Pushdown.NONE` still works — slower and coarser, and the planner
degrades instead of failing.
**Status** Solved — a new system starts as a `DeclaredCatalog` and earns precision later.

### P111 — Metadata lag silently skips rows on incremental ingest
**Who feels it** Whoever finds the gap months later.
**Today** Snowflake's ACCOUNT_USAGE lags up to three hours; Databricks system tables
about two. Advancing a resume token past the lag permanently skips rows that had not
landed yet. This is discovered in production or not at all.
**fathom** `Capabilities.freshness_lag` states it per adapter and the ingest path
respects it.
**Status** Solved — declaring it is the fix; the bug is silent otherwise.

### P112 — Incremental ingest cannot resume after a failure
**fathom** Resume tokens per dataset and adapter in the store.
**Status** Solved.

### P113 — Metadata is locked inside one vendor's catalog
**fathom** `report.emit` produces OpenLineage, DataHub, Atlas, OpenMetadata, and Marquez
payloads as pure functions with no clients, so nothing is coupled to a vendor SDK.
**Status** Solved.

### P114 — dbt projects have lineage that no other tool can read
**fathom** `ingest.dbt` reads the manifest and compiled SQL.
**Status** Solved.

### P115 — Spark, Flink, Trino, Airflow, and Dagster each need bespoke integration
**fathom** `ingest.openlineage` consumes the events all of them already emit.
**Status** Solved.

### P116 — Evaluating a data tool requires a warehouse and a procurement cycle
**fathom** DuckDB engine and local storage adapters run the whole product on a laptop;
`examples/` are executed by the test suite.
**Status** Solved.

### P117 — Documentation drifts from the tool it documents
**fathom** `tests/test_docs.py` checks that every documented command, adapter, and
config key exists, and the examples are executed by the suite.
**Status** Solved.

### P118 — Partition specs drift across invocations
**fathom** `fathom.yml` holds them in one place.
**Status** Solved.

### P119 — Nothing tells you what is silently making plans worse
**fathom** `fathom doctor`.
**Status** Solved.

### P120 — A tool that writes to your data cannot be safely trialled
**fathom** Nothing writes by default. `plan` prints, `erase` dry-runs, and applying
requires an engine binding supplied deliberately.
**Status** Solved.

### P121 — SQL dialects that cannot be parsed produce wrong lineage
**Today** A parser that guesses produces edges that are confidently wrong, which is
worse than no edge.
**fathom** Anything unprovable — an opaque UDF, an unparseable dialect, a `MERGE`, a
spec mismatch, a cycle — widens to `UNBOUNDED`.
**Status** Solved.

### P122 — Self-referencing incremental models make a planner loop forever
**fathom** Bounded enlargements, then widen. Only a dataset that can actually reach
itself (Tarjan, cached) is charged against the cycle budget; a wide join gets the
acyclic safety valve instead.
**Status** Solved — and the distinction matters commercially. Charging fan-in against a
cycle budget turns any hub table with more parents than the budget into a spurious full
rebuild, which silently destroys the saving the planner exists to produce.

### P123 — A graph too large to enumerate crashes the planner
**fathom** `MAX_ENUMERATED_KEYS` collapses the widest dimensions rather than enumerating.
**Status** Solved.

### P124 — Restating a metric requires knowing every published number derived from it
**Who feels it** Whoever has to write the notice, from memory.
**Today** Lineage stops at the edge of the warehouse. `descendants` returns tables,
every one of which an engineer can rebuild — which is the wrong boundary, because the
expensive question is what has already been *told to people*.
**fathom** `graph.sinks` gives a dashboard, board pack, filing, export, or served
endpoint a `DatasetId`, so the existing traversal reaches them. `restatement_impact`
names them, and `notice_text` drafts the notice from lineage.
**Status** Solved. Filings and signed reports are reported on their own line and
`has_regulatory_exposure` is a separate question, because a wrong dashboard is
embarrassing and a wrong filing is a legal event — one count hides the other. The
draft says in its own text that downstream is not the same as material, since that
judgement is not the graph's to make.

### P125 — Two teams compute the same metric differently
**Status** **Gap** — no semantic layer or metric definition registry.

### P126 — Data contracts between teams are prose in a wiki
**fathom** `govern.contracts.Contract` binds a producer, named consumers, and the
promises between them; `verify` dispatches to the existing quality, schema, and freshness
checks and collects what they say.
**Status** Solved. The machinery already existed and was unattributed — what a contract
adds is *who*. A breach names who promised what to whom, and severity follows the blast
radius: the same removed column is a warning with no consumer and an error with three.
Anything that could not be checked is listed as unchecked rather than passing silently.

### P127 — Cost of a query is known; cost of a *dataset* over its life is not
**fathom** `graph.plan.lifetime.accumulate` totals what each dataset has cost across
the runs that actually happened, under the same `CostModel` that prices plans.
**Status** Solved — and cost is accumulated from recorded runs, never modelled from a
per-run figure times an assumed age, which would always flatter recently-created tables.
A dataset with no recorded runs has `None`, not zero: unmeasured and free are different
facts, and only one justifies keeping a table.

### P128 — Nobody can tell which datasets are worth the money they cost
**fathom** `graph.plan.lifetime.value` divides the two: lifetime spend against
observed reads, returning `earning`, `review`, `cheap_and_quiet`, or `unmeasured`.
**Status** Solved, with the asymmetry stated in the output. Cost is *measured* and usage
is *observed*, so a table read once a year for a regulatory filing looks identical here
to a dead one. `value` returns a verdict and never a decision, the review list is sorted
with the money at the top, and `threshold` has no default because the right number is a
fraction of a budget this library cannot see.

---

## Where the gaps are

Nine entries above are `Gap` or a `Partial` with real substance behind it. Grouped:

**Closed since this catalog was written** — P8, P10, P32, P33, P34, P35, P36, P42,
P56, P78, P124, P126, P127 and P128 moved to `Solved`, and P64 to a `Partial` that states exactly what it can and
cannot prove. Eight new modules, ten renderers, three persisted event streams, four CLI
commands, 284 tests. See [improvements.md](improvements.md).

**Still open:**

| Theme | Problems | Why it is still open |
|---|---|---|
| Proving safety, not only risk | P64 | the group-size bound needs a scan; today it can only prove risk |
| Reconstructing a past graph | P78 | revisions store diffs, not snapshots, by deliberate tradeoff |
| Semantics | P125 | not planned — see improvements.md for why |
| Planner precision | P122 | fixed in `graph.model`; the catalog entry predates the fix |
