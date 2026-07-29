# What fathom is

Start here if you have not used a lineage or data-observability tool before, or if
the README read like it assumed you had.

## The situation it is built for

You have tables built from other tables. Raw data lands, something cleans it,
something aggregates that, and a dashboard reads the end of the chain. A scheduler
rebuilds the whole thing on a timer.

Then a vendor redelivers one day of March.

Which tables are now wrong? Not "which tables depend on that source" — every one of
the four hundred does, transitively. Which *rows*. The monthly rollup is wrong for
March and correct for the other 35 months. The seven-day rolling metric is wrong for
one week. The EU partition is affected and the US partition is not.

Almost nobody can answer that, so teams pick between two bad options: rebuild
everything on every run, or rebuild what somebody remembers and find the stale
numbers later, usually because a customer found them first.

fathom answers it.

```bash
fathom plan --dirty 'raw.events@dt=2026-03-14,region=eu'
```

```
duckdb/raw.events       dt=2026-03-14T00:00:00/region=eu
duckdb/silver.events    dt=2026-03-14T00:00:00/region=eu
duckdb/gold.monthly     dt=2026-03-01T00:00:00/region=eu
```

One day at the source, one day downstream, one *month* in the rollup, and the whole
answer stays inside `region=eu`. Everything else is provably untouched and does not
need to be rebuilt.

## The one idea

A normal lineage tool records that `gold.monthly` is built from `silver.events`.
That is true, and it is only enough to tell you to rebuild `gold.monthly` entirely.

fathom records the same edge plus one extra fact: **how the partitions line up**.
`silver.events` is sliced by day, `gold.monthly` by month, and the edge says that a
dirty day makes exactly the month containing it dirty. Another edge might say a
dirty day taints the six days after it, because the table computes a 7-day trailing
average. Another says a value passes straight through, so `region=eu` in means
`region=eu` out.

Those little rules are called **partition mappings**, and they are the whole trick.
They chain together along a path through the graph — day → day → month — so the
planner can start from what actually changed and arrive at a list of partitions
rather than a list of tables.

When the SQL is too complicated to prove anything about, fathom says so and widens
the answer to "the whole dataset" instead of guessing. That direction is deliberate:
rebuilding too much wastes money, and rebuilding too little serves wrong numbers
quietly for weeks.

## Four questions, one graph

Once that graph exists, four different jobs turn out to be the same lookup.

| | Question | Command |
|---|---|---|
| **plan** | Something changed at the source. What has to be rebuilt? | [`fathom plan`](plan.md) |
| **check** | A number moved. Did it drift, and what upstream caused it? | [`fathom check`](check.md) |
| **label** | What is in this column, and who is allowed to use it? | [`fathom label`](label.md) |
| **erase** | A person asked to be deleted. Where did their data end up? | [`fathom erase`](erase.md) |

Those are usually four products from four vendors. They are one tool here because
each is much cheaper — and in two cases only possible — once the graph exists:

- Profiling every table nightly costs real warehouse credits. Profiling only the
  partitions the graph says changed is what makes it affordable at all, and `check`
  needs those profiles.
- "Revenue moved 8%" is an alert somebody has to investigate. "Revenue moved because
  `fx_rates` changed three hops upstream" is the answer, and it comes from the graph.
- Nobody hand-labels 40,000 columns, so `label` guesses from the data and propagates
  the guess along edges — a column derived from an email column is still an email
  column.
- Deleting one person from a lakehouse means rewriting every file that might hold
  their rows. Knowing which partitions actually hold them is the difference between
  an afternoon and a quarter.

## Vocabulary

Six words carry most of the documentation.

| Term | What it means here |
|---|---|
| **dataset** | Anything with an address that gets built: a table, a folder of Parquet, a dbt model. Also a trained model, a vector index, or a prompt — see [AI assets](ai.md). |
| **partition** | A slice of a dataset that can be rebuilt on its own, written `dt=2026-03-14/region=eu`. If your table has no such slices, it has exactly one partition: all of it. |
| **grain** | How big a time slice is — hour, day, month, year. A daily source feeding a monthly rollup is a grain change, and the mapping has to do the arithmetic. |
| **edge** | "This dataset is built from that one." Learned from SQL, dbt, warehouse query history, or OpenLineage events, not written by hand. |
| **partition mapping** | The rule attached to an edge: if this input partition is dirty, which output partitions are dirty? |
| **dirty** | Known to have changed and not yet rebuilt. A plan is the set of dirty partitions after propagation. |

Two more worth knowing because the docs lean on them:

**Over-invalidate** means planning a rebuild of something that did not actually need
it. It costs compute. **Under-invalidate** means missing something that did need it.
It serves wrong data. fathom is built to do the first and never the second, and the
same asymmetry runs backwards through `erase`, which would rather refuse to delete
than delete something it should not have.

## Is this for you yet?

It pays off when your tables are partitioned or incremental, when the rebuild bill
is a number somebody notices, or when you get asked questions like *what was this
model trained on* and *where did this user's data go* and answering takes a week.

It pays off less when everything rebuilds in a minute, or when nothing is
partitioned at all. Without partitions, fathom degrades to ordinary
dataset-level lineage: still useful for impact analysis and erasure, but the
rebuild savings mostly go away, because the smallest thing it can name is a whole
table.

Two things to be clear about before you invest an afternoon:

- **It does not run your rebuilds.** `plan` prints a list. Executing that list stays
  with your scheduler, dbt, or whatever you use today. `erase` dry-runs unless you
  tell it not to.
- **It does not ask you to trust it.** [Shadow mode](shadow.md) runs the planner
  beside your existing full rebuild and grades it: how many partitions it skipped,
  and how many it wrongly called clean. The second number must be zero, and the
  tooling fails loudly when it is not. Nothing changes in your pipeline while you
  collect that evidence.

## Next

[Getting started](getting-started.md) builds a working three-table project from
scratch. If you would rather read code, [`examples/01_local_lakehouse.py`](../../examples/01_local_lakehouse.py)
runs the whole loop on local Parquet and ends by asserting that the incremental
rebuild is byte-identical to a full one.

For the mechanics under all this — identity, the partition lattice, and the
soundness invariant written as an equation — read [Concepts](concepts.md).
