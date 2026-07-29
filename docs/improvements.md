# Improvements

What was done about the `Gap` and weak `Partial` entries in
[problems.md](problems.md). Each section names the problems it closes, so the two
documents stay in step.

Twelve modules — **165 public entry points** — plus eleven renderers, sixteen store
methods over three schema migrations, nine CLI commands, two config blocks, six
`doctor` checks, an emit facet, three orchestrator generators, four guides, and a
runnable example.

**581 tests**, of which 53 are property tests over generated inputs. The suite went
852 → 1627.

| | |
|---|---|
| Public entry points (12 modules) | 165 |
| Renderers, store methods, CLI commands | 36 |
| Config keys, `doctor` checks, emit facet, generators | 20 |
| Fixes to pre-existing code | 4 |
| **New behaviours** | **225** |
| **Tests, each pinning one** | **581** |
| **Total** | **806** |

Every one exists because the catalog said the problem was real and the library could
not answer it.

---

## `observe.completeness` — P33, P34

The only check in `observe` that can see a partition which never arrived.

Everything else in the package reads data that showed up and asks whether it looks
right. A partition that was never written has no profile to drift, no rows to fail an
expectation, and no signal at all — it is indistinguishable downstream from a
partition that legitimately holds nothing. The gap gets found weeks later by somebody
noticing a dip in a chart.

- `expected_keys` enumerates what should exist from the partition spec and a range;
  `missing` and `unexpected` compare it against what does
- `gaps` collapses absences into contiguous runs, computed **within each value slice
  separately** — `region=eu` missing three days and `region=us` missing one are two
  incidents, and merging them misreports both
- `report` states which value domains it had to infer, because an inferred domain
  cannot report a value that has never once appeared
- `expected_keys` raises past `max_keys` rather than truncating: a shortened expected
  set makes an incomplete dataset look complete, which is the failure being prevented
- `Arrival`, `late_arrivals`, `replays`, `restatements` cover P34. A repeat arrival
  with an unknown digest is classified as a restatement, never a replay, because
  treating "cannot tell" as "unchanged" is what lets a silent double-count through
- lag is measured from where the bucket *closes*, so a daily partition written at
  02:00 the next morning is two hours late rather than twenty-six

## `observe.seasonal` — P32

Baselines that know Tuesday from Sunday.

`quality.learn` derives one flat band from all observations, which is right for most
data and wrong for anything cyclical. A B2B events table does a fifth of its Tuesday
volume on Sunday; one band across both is wide enough to admit Tuesday's floor and
Sunday's ceiling, which is to say wide enough to catch nothing. Narrow it to Tuesday
and it fires every weekend. Teams resolve this by muting the check, and the real
failure mode is not a wrong alert but an absent one.

- `learn_seasonal` buckets by `Cycle` — hour-of-day, day-of-week, day-of-month,
  month-of-year — and learns a band per bucket
- a bucket below `min_observations` goes to `unmodelled` and is **not checked**. Two
  Sundays is not a baseline for Sunday, and an invented band carries the same
  authority as a real one
- `strength` measures whether the cycle explains anything before one is assumed, so
  reaching for this module over `quality.learn` stays a decision. It returns `None`
  rather than `0.0` when there is no spread to measure, because no answer and no
  seasonality are different facts
- `Observation.when` is the partition's own moment, not when profiling ran —
  otherwise a Monday partition backfilled on Saturday lands in the Saturday band

## `govern.reidentification` — P64

Columns that identify nobody alone and everybody together.

`policy` finds direct identifiers and catches the easy half. The hard half is that a
birth date identifies nobody, a postcode identifies nobody, and together they identify
most of a population. Per-column labelling is structurally blind to it, because the
property is not a property of any column.

It over-reports risk and never under-reports it — the mirror of the planner, for the
same reason: a false alarm costs a review, a missed one is a disclosure that cannot be
recalled.

- `k_upper_bound` bounds the average group size at `N / max(dᵢ)` from per-column
  distinct counts, which needs no scan
- so `assess` can **prove a dataset risky** and can never prove one safe. `is_clear`
  is named for the absence of proven risk, and the summary states that in its own text
- `singling_out` catches the near-unique column that carries no label and identifies
  perfectly — a salted hash, an order reference
- `linkage_risks` finds pairs of datasets that are each defensible and jointly
  identifying because a shared ancestor lets them be joined. Compared by label kind
  rather than column name, so `dob` joined to `birth_date` correctly adds nothing

## `observe.usage` — P42, and half of P128

Who actually reads a dataset.

The graph answers what *could* consume a dataset, never what does, and the difference
is where a large part of a warehouse bill goes: tables built nightly for a dashboard
decommissioned in 2023, kept because the structural question keeps answering yes about
a consumer that is itself dead.

- `retirement_candidates` finds datasets nothing read whose descendants nothing read
  either — a dataset one hop from something used is not unused
- **nothing here returns "unused."** Everything returns "no reads observed", carries
  the `window` it observed over, and puts that window in its own summary text. Query
  logs have retention limits and a table read once a quarter looks dead over thirty
  days. Deleting a table read annually for a filing is the one mistake in this area a
  rebuild cannot undo
- `human_principals` discounts principals that look like schedulers, so a table read
  only by the job maintaining its own downstream still surfaces. Named as the
  heuristic it is

## `graph.history` — P78

The graph's own revision history.

`diff` compares two graphs you happen to be holding, which answers "what changed in
this pull request" and nothing else. Every question that arrives after an incident is
about time instead.

- `record` appends a revision holding the `GraphDiff` from its predecessor plus a
  content digest of the whole graph
- `narrowings_of(history, src, dst)` answers the incident question directly: six days
  of downstream data stopped being invalidated — when did that window shrink, and who
  shrank it
- `record` refuses a revision computed against a stale graph, since that attributes one
  person's change to whoever committed next
- recording an unchanged graph is a no-op, so a nightly ingest that found nothing does
  not fill the history with noise
- **stated tradeoff:** revisions store diffs, not snapshots. This can say exactly when
  and how an edge changed; it cannot hand you the graph as it stood last March.
  `digest_at` verifies a graph you still hold is the one a revision described, which is
  the part an audit needs

## `govern.contracts` — P126

A promise one team makes to another.

Everything needed to enforce a data contract already existed and none of it was bound
to a promise. `quality` checks a suite without saying whose. `schema` finds a breaking
change without saying who it breaks. `diff` gates a narrowing without knowing which
consumer it hurts. So the contract lives in a wiki and is discovered to have been
violated by the consumer, in production, on a Sunday.

- `Contract` binds one producer, named consumers, and the promises between them
- `verify` adds no checking machinery — it dispatches to the modules above. What it
  adds is *attribution*, which is the difference between an alert and an escalation
- **severity follows the blast radius**, the one judgement the module makes on its
  own: the same removed column is a warning with no consumer and an error with three
- a promise with no evidence supplied is listed `unchecked` rather than passing. A
  report that looks met because the caller forgot to pass a profile is the bug

---

## `graph.plan.lifetime` — P127, and the other half of P128

What a dataset has cost since it existed, against whether anyone reads it.

`cost` prices a plan and answers "should I run this backfill". It cannot answer the
question a platform owner faces quarterly: should this table exist at all. A table
costing $40 a night is invisible; three hundred of them is the budget. Nobody culls
them because the two facts needed live in different systems and are never divided by
one another.

- `accumulate` totals the runs that **actually happened**, under the caller's existing
  `CostModel`. A lifetime figure derived from a per-run cost times an assumed age would
  be a guess wearing a number's clothing, and would always flatter new tables
- a dataset with no recorded runs gets `None`, not zero — inventing a zero would make
  the unmeasured table look like the cheapest thing in the warehouse
- `value` divides lifetime spend by observed reads into `earning`, `review`,
  `cheap_and_quiet`, or `unmeasured`, sorted with the money at the top
- `threshold` has no default. The right number is a fraction of a budget this library
  cannot see, and a made-up default is one people leave alone
- the summary states the asymmetry once: cost is *measured*, usage is *observed*, so a
  table read once a year for a filing looks identical here to a dead one

## `graph.sinks` — P124

The last hop, where a number stops being data and becomes a published claim.

Lineage conventionally stops at the edge of the warehouse, which is the wrong boundary
for the most expensive question anybody asks of it: we restated this metric, so what
have we already told people? A dashboard, a board pack, a filing, an export, and a
served endpoint are all downstream and none of them is a table.

- sinks are datasets, so the existing traversal, policy propagation, and erasure
  machinery already reach them. No second graph
- sinks are **terminal** — `record_publication` refuses an edge out of one, because a
  filing that feeds a table would extend every restatement cone through it forever
- filings and signed reports are reported separately, and `has_regulatory_exposure` is
  its own question: a wrong dashboard is embarrassing, a wrong filing is a legal event
- `notice_text` drafts the restatement notice from lineage, and states in its own text
  that downstream is not the same as material

## Making them usable

A module nobody can reach is a module nobody uses. Alongside the eight:

- **Ten renderers** in `report.render` — `completeness_to_markdown`, `usage_to_markdown`,
  `retirement_to_markdown`, `risk_to_markdown`, `contract_report_to_markdown`,
  `lifetime_to_markdown`, `value_to_markdown`, `history_to_markdown`,
  `restatement_to_markdown`. Each carries the qualifying sentence its module carries;
  a Markdown table that drops it turns "no reads observed in 90 days" into "unused"
- **Three persisted event streams** in `store` (schema 2) — arrivals, reads, and runs.
  Every new module answers a question about history and previously had nowhere to get
  it. `present_partitions` reads the arrival log rather than a listing, so completeness
  still answers after a partition has been deleted
- **Four CLI commands** — `fathom completeness`, `usage`, `value`, `impact`.
  `completeness` and `impact` exit non-zero on a finding; `usage` and `value` do not,
  because failing a build over a review list trains people to append `|| true`
- **A `publications` block** in `fathom.yml`, so `fathom impact` answers without any
  Python. Re-applied from config on every load rather than persisted, so deleting a
  declaration removes it — an artefact that outlived its declaration would keep
  appearing in notices, and a notice is only worth reading if it is current
- **Two guides** — [completeness](guide/completeness.md) and
  [usage, value, impact](guide/value.md) — plus `examples/06_worth_keeping.py`, which
  the suite executes. It ends by *declining* to recommend deleting a table that is
  expensive and unread, which is the more useful half of that feature

### Persistence, and the rest of the last mile

- **`store` schema 2** — arrivals, reads, and runs. Every new module answers a
  question about history and previously had nowhere to get it. `present_partitions`
  reads the arrival log rather than a listing, so completeness still answers after a
  partition has been deleted
- **`store` schema 3** — graph revisions and the edges each one touched. A history
  that does not survive the process is not a history, and every question it answers is
  asked weeks later. Idempotent on digest, so a replayed ingest does not fork the chain
- **`contracts:` and `publications:` in `fathom.yml`**, with a duration parser that
  rejects a bare number — every config format that accepts one ends up with two
  readers silently disagreeing about seconds versus hours
- **Six CLI commands** — `completeness`, `usage`, `value`, `impact`, `risk`,
  `contracts`. The first four and `contracts` exit non-zero on a finding; `usage` and
  `value` do not, because failing a build over a review list trains people to append
  `|| true`
- **`doctor` checks for declared blocks**, which fail by being silently vacuous: a
  contract on a never-profiled dataset reports "unchecked" forever, and a contract
  with no consumers escalates to nobody
- **`emit.sink_facet`**, so the artefact/table distinction survives export. A catalog
  that receives a filing and a staging table as two identical nodes has lost the one
  property that made the filing worth tracking
- **`report.orchestrators` and `fathom dag`** — P10. `to_task_list` handed the plan
  over as data and left one step to the user, which in practice is the step where a
  good plan stops being used. Airflow, Dagster, and Prefect files are generated
  directly, importing none of them. Writing this found that task ids were only
  stripped of dots, so a dataset named `weird."name` emitted a DAG that did not parse;
  `to_task_list` now sanitizes to an identifier and is typed with a `TaskEntry`
  TypedDict instead of `dict[str, object]`
- **Revisions are recorded on `fathom ingest`**, with `--author` and `--note`, and read
  back by `fathom history --edge 'src->dst'`. A history nobody remembers to write is
  empty on the day it is needed. Concurrent writers are handled by rebuilding the chain
  from the store and skipping rather than appending a revision whose parent neither
  process saw

### Properties, not just examples

41 property tests over generated inputs, checking the claims the docstrings make:
missing partitions and present ones exactly partition the expected set; runs are
contiguous and never straddle a value slice; lifetime cost is order-independent (every
orchestrator that retries delivers runs out of order); a graph digest ignores edge
insertion order; `k_upper_bound` is monotone; and neither `assess` nor `verify` ever
reports clear while holding a finding.

Four of these found real problems. The `sinks` property caught a generator that
disagreed with the constructor's contract about what counts as a name — Python treats
the separator controls `\x1c`-`\x1f` as whitespace. The `doctor` test caught something
worse: applying a publication *registers its endpoints*, so a mistyped input silently
becomes a real node in the graph, and the check that was supposed to catch the typo
was reading the graph the typo had already created. It now tests against the ingested
graph instead.

The orchestrator generators found a third: `to_task_list` built task ids by replacing
dots and nothing else, so a dataset named `weird."name` emitted an Airflow file that
did not parse. Ids are now reduced to identifiers, and every generated file is checked
with `ast.parse` in the tests, because the alternative is finding out in someone's
scheduler at 3am. A fourth turned up while testing `fathom history`: a SQL change that
joins a second table does *not* change the existing edge, because the parser cannot
attribute unqualified columns across two sources — the test now asserts what the
ingest actually claimed rather than what seemed obvious.

Writing the example found a third inconsistency: `retirement_candidates`
discounts scheduled principals and `lifetime.value` did not, so an intermediate table
maintained by its own pipeline looked read. Rather than pick a default,
`usage.read_counts(stats, people_only=...)` now makes the choice visible at the call
site, and both docstrings name the trap.

---

## `graph.plan.billing` — P8

What the warehouse actually charged, against what the model said it would.

Every savings figure in this library rests on declared rates that nothing checked. A
model three times off does not fail — it produces confident numbers in the wrong
currency, and the first person to notice compares a quarterly saving against the
invoice and finds they disagree.

- `reconcile` compares modelled against billed over the same window; `calibrate`
  returns a corrected `CostModel` rather than mutating one
- **per-dataset reconciliation only where the bill attributes.** Warehouse billing is
  per-warehouse or per-job; apportioning an unattributed total across datasets would
  hand the cost model's own assumptions back as evidence for themselves. A dataset the
  bill never named is absent from the result, not shown against zero
- `attributed_share` reports how much of the bill can be assigned at all, so a low
  number reads as "we do not tag queries" rather than as a finding
- `calibrate` refuses below fourteen periods, on a zero bill, and on a model already
  inside tolerance — three cases where a correction is worse than none
- scaling every rate by one factor is deliberately blunt: the bill does not say which
  basis was wrong, so splitting the correction across per-partition and per-byte rates
  would be a guess dressed as a calibration
- it queries nothing. `WAREHOUSE_METERING_HISTORY` and friends are an adapter's job;
  a library that needs credentials to be imported is one people vendor around

## `observe.joins` — P35

Join keys, and the two ways one quietly ruins a table.

A join is the only operation in a warehouse that can make a table *larger* than its
inputs, and the only one that silently drops rows. Both come from the key's shape
changing, and neither raises. `drift` sees the symptom and reports it as one finding
per column, which is how a real join failure arrives as thirty alerts with no cause.

- `uniqueness_lost` catches the key that was unique and is not any more — the single
  most common cause of a revenue total silently doubling
- `orphan_floor` is the only cross-dataset claim, and only in the provable direction:
  if the left has more distinct keys than the right holds in total, the difference
  cannot match whatever the overlap is. It proves rows will be dropped and can never
  prove none will be
- nothing estimates output rows, because key *overlap* needs a scan. `amplification`
  reports the arithmetic signature of fan-out and returns a number rather than a
  verdict, since a legitimate unnest looks identical
- a falling fan-out is a warning and a rising one an error, because fewer duplicates
  does not duplicate anything downstream

## `govern.replicas` — P56

Copies of the data no adapter can see, declared so a proof can name them.

Every erasure artifact carried the same caveat: it covers what the configured adapters
can reach. That is honest and it is not enough. "There may be copies elsewhere" is
unfalsifiable — it cannot be reviewed, cannot be closed out, and cannot be told apart
from "we did not look".

The gap is specific and always the same list: a snapshot in another account, a read
replica, a CSV a partner receives monthly, a Kafka topic, a vendor's system that
ingested an export in 2023. None appears in any lineage graph, because none is
derivable — they are facts about an organization, not about its SQL.

- `declare` turns the caveat into a checklist with owners and dates; `proof_entries`
  puts them in the proof by name and disposition, which is the difference between a
  proof an auditor closes and one they return
- a copy beyond the organization's control is `UNREACHABLE` **even when attested**,
  because an attestation about a vendor's system is a claim rather than an action
- a copy whose retention has not elapsed and which nobody attested is `OUTSTANDING`,
  never assumed expired — time passing is not evidence
- `obligations_for` sorts the ones somebody else has to perform last, because mixing
  them into the internal work list is how they get lost
- `coverage` reports how much of the estate was declared at all, so a low number reads
  as "we have not mapped this" rather than "we are clean"

Nothing here deletes anything. `erase` still refuses to act outside what it can
verify; this widens what the *report* is honest about.

---

## Not planned

**P125 — two teams compute the same metric differently.** This needs a semantic layer,
which is a product rather than a module. `fathom` models where data *came from*, not
what it *means*; a metric registry that does not own query generation is a second place
for definitions to disagree, which is the problem restated rather than solved. Recorded
here so the decision is visible rather than looking like an oversight.

## Still open

**P124 — restating a metric requires knowing every published number derived from it.**
`descendants` gives the datasets. Nothing models published artefacts — dashboards,
reports, filings — as sinks, so the last hop out of the warehouse is still invisible.
