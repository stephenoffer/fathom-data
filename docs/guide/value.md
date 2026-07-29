# usage, value, and impact — is this table worth keeping

Three questions a platform owner faces quarterly that a lineage graph alone cannot
answer, because all three are about what happens *outside* the warehouse.

## Who actually reads this

The graph answers what *could* consume a dataset. It cannot answer what does, and the
difference is where a large part of a warehouse bill goes: tables built nightly for a
dashboard decommissioned in 2023, kept because the structural question — does anything
depend on this? — keeps answering yes about a consumer that is itself dead.

```bash
fathom usage --days 90
fathom usage --days 90 --retire
```

Feed it reads from your query log:

```python
store.record_reads(
    usage.events_from(rows_from_query_history, kind="query")
)
```

`--retire` lists datasets nothing read whose descendants nothing read either. A
dataset one hop from something used is not unused.

**Nothing here returns "unused".** Everything returns "no reads observed", carries the
window it observed over, and repeats that window in its own output. Query logs have
retention limits, some consumers do not log, and a table read once a quarter looks
identical to a dead one over thirty days. Deleting a table read annually for a
regulatory filing is the one mistake in this area that a rebuild cannot undo, so the
output is a review list and is named like one.

## Is it worth what it costs

```bash
fathom value --threshold 500 --price-per-partition 0.02 --days 90
```

```
3 dataset(s) unread and above the threshold:
    duckdb/gold.legacy_attribution: 1,840.00 spent and unread — no reads observed in
    90 day(s), which is not the same as no reads
    ...
    2 dataset(s) have no cost history and were not judged; unmeasured is not free
    cost is measured, usage is observed — a table read once a year for a filing looks
    identical here to a dead one
```

Cost comes from runs that actually happened, recorded by your orchestrator:

```python
store.record_run(RunRecord(dataset, at=finished, partitions=42, bytes_scanned=n))
```

Never modelled from a per-run figure times an assumed age — that would be a guess
wearing a number's clothing, and it would always flatter recently-created tables. A
dataset with no recorded runs is **unmeasured**, not free, and is reported separately
rather than sorted to the cheap end of the list.

`--threshold` has no default. The right number is a fraction of a budget this tool
cannot see, and a made-up default is one people leave alone.

## What have we already told people

Lineage conventionally stops at the edge of the warehouse. That is the wrong boundary
for the most expensive question anybody asks of it:

> We restated this metric. What have we already told people?

A dashboard, a board pack, a filing, a customer export, and a served endpoint are all
downstream, and none of them is a table. Declare them:

```yaml
publications:
  - name: revenue/exec
    kind: dashboard
    instance: looker
    inputs: [gold.monthly]

  - name: 10-K/2026
    kind: filing
    instance: sec
    inputs: [gold.monthly]
```

```bash
fathom impact --dataset silver.revenue --reason "fx rates were wrong since March"
```

```
Restatement notice (draft) — duckdb/silver.revenue
Reason: fx rates were wrong since March

Affected published artefacts:
  - dashboard revenue/exec on looker
    via duckdb/silver.revenue -> duckdb/gold.monthly -> dashboard://looker/revenue/exec
  - filing 10-K/2026 on sec
    via duckdb/silver.revenue -> duckdb/gold.monthly -> filing://sec/10-K/2026

Of these, the following are filings or signed reports and may require a formal
amendment rather than a refresh:
  - filing 10-K/2026 on sec
```

Exits non-zero when a filing or signed report is affected. A wrong dashboard is
embarrassing and a wrong filing is a legal event; one count would hide the other, so
they are separate lines and `has_regulatory_exposure` is a separate question.

Publications are **declared, not discovered**. No BI tool exposes its queries
uniformly, and a guessed dependency is worse than an absent one — a restatement notice
built on a guess names the wrong people.

The notice is a draft and says so. The graph knows what is downstream, which is not the
same as what was material, and that judgement is not the graph's to make.

## Related

- [configuration](configuration.md) — the `publications` block
- [plan](plan.md) — what to rebuild once you know the scope
