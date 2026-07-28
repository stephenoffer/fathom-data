# check — what drifted, and what caused it

Profiling plus drift detection plus upstream attribution. The third part is what
makes this different from an alert.

## Profiling is nearly free

Parquet and ORC footers carry per-row-group min/max, null counts, distinct
estimates, and sizes — enough for most of a profile, readable at metadata cost
without touching a single data page.

```bash
fathom profile
```

```
file:///lake/events
  8,412,336 rows across 214 file(s)
  amount                   double       nulls=0.2%     0.01..99420.5
  dt                       date         nulls=0.0%     2026-01-01..2026-03-14
  region                   string       nulls=0.0%     apac..us
  user_id                  string       nulls=0.0%     u0001..u9999
```

Row-group statistics also give granularity *below* the directory partition, which
no catalog can offer.

Because it costs metadata rather than scans, profiling every partition the planner
reports dirty is affordable on a schedule. Whole-table nightly profiling is how
these tools get uninstalled.

## Detecting drift

```bash
fathom check
```

The first run on a dataset records a baseline. Subsequent runs compare:

```
file:///lake/events:
  [error] amount: float64 -> string
  [warn] row count moved -34.2% (8412336 -> 5534090)
  [warn] region: null rate 0.0% -> 12.4%
```

| Finding | Severity | Meaning |
|---|---|---|
| `column_removed` | error | A column disappeared |
| `type_change` | error | A column's type changed |
| `row_count_shift` | warn | Row count moved beyond tolerance |
| `null_rate_shift` | warn | Null rate moved beyond tolerance |
| `min_raised` / `max_lowered` | info | The value range shrank — usually an upstream filter change |
| `column_added` | info | A new column appeared |

`check` exits non-zero on any error-severity finding, so it drops into CI directly.

## Attribution

An alert says a number moved. A diagnosis says why:

```
file:///lake/gold:
  [warn] revenue: null rate 0.0% -> 8.1%
    revenue derives from: file:///lake/gold#revenue <- file:///lake/silver#amount
```

That trail comes from the graph, walking column edges upstream. It is only as good
as the column lineage on those edges — an edge with no column detail cannot attribute
anything, which is why `fathom doctor` flags them.

## Why small partitions do not page anyone

A threshold without a sample-size guard is a noise generator. A partition of 40 rows
crosses any null-rate tolerance by chance alone.

Below `min_rows` (1000 by default) findings downgrade to `info` rather than
disappearing, so a genuinely tiny partition stays visible without waking anyone:

```python
from fathom import drift

drift(before, after, min_rows=1000, null_rate_tolerance=0.05, row_count_tolerance=0.25)
```

## Statistics that are absent, not zero

Parquet statistics are optional. A column whose writer omitted them yields `None`,
never a fabricated value:

```python
column.null_count is None    # unknown — the writer did not record it
column.null_count == 0       # known: there are no nulls
```

Conflating the two would report "no nulls" for every column written by a tool that
skips statistics, which is worse than reporting nothing.

## Scoping to one partition

```python
from fathom.project import Project

with Project.load() as project:
    got = project.profile(dataset, partition=KeyPredicate.of(dt=datetime(2026, 3, 14)))
```

This is what makes continuous profiling affordable: profile only what the planner
says changed.

## Reconciling two locations

```python
from fathom import profile_parquet, drift

warehouse_copy = profile_parquet(paths_a, dataset=ds)
lake_copy      = profile_parquet(paths_b, dataset=ds)

for finding in drift(warehouse_copy, lake_copy, min_rows=1):
    print(finding)
```

Comparing a source of truth against its analytical copy is the one job every data
team writes by hand.
