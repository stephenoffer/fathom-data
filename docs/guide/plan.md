# plan — what must be rebuilt

Given what changed at the source, compute exactly which partitions downstream are
now wrong.

## Basic use

```bash
fathom plan --dirty 'raw.events@dt=2026-03-14,region=eu'
```

```
duckdb/raw.events
    dt=2026-03-14T00:00:00/region=eu
duckdb/silver.events
    dt=2026-03-14T00:00:00/region=eu
duckdb/gold.monthly
    dt=2026-03-01T00:00:00/region=eu
```

Datasets are listed in rebuild order — sources before anything derived from them.

## Discovering the seeds

Rather than naming what changed, let the source adapters find out:

```bash
fathom detect        # scan sources, report changes, advance tokens
fathom plan --detect # scan and plan in one step
```

`detect` remembers a resume token per source, so the second run reports only what
arrived since the first.

## From Python

```python
from datetime import datetime
from fathom import KeyPredicate
from fathom.project import Project

with Project.load() as project:
    plan = project.plan({
        project.config.resolve("raw.events"): [
            KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
        ]
    })

    for dataset in plan.order:
        for partition in plan.partitions(dataset):
            print(dataset, partition)
```

`plan.order` is topological. `plan.widened` names datasets that lost precision, and
`plan.reasons[dataset]` says why.

## Applying a plan

There is no `fathom apply`. Executing requires an engine binding, which belongs in
a pipeline rather than a shell command where a typo costs you a table.

```python
from fathom.adapters import DuckDBEngine

engine = DuckDBEngine(database="warehouse.duckdb")
engine.register_model(gold, "SELECT ... FROM silver.events GROUP BY 1, 2", month_spec)

for dataset in plan.order:
    if dataset in engine.models:
        engine.apply(dataset, plan.partitions(dataset))
```

`apply` runs a delete-then-insert inside a transaction, filtered on the target's
partition columns:

```sql
DELETE FROM gold.monthly WHERE "dt" >= TIMESTAMP '2026-03-01 00:00:00'
                           AND "dt" <  TIMESTAMP '2026-04-01 00:00:00'
                           AND "region" = 'eu';
INSERT INTO gold.monthly SELECT * FROM (<model>) AS _fathom_rebuild WHERE <same>;
```

Filtering the model's *output* rather than its inputs is correct regardless of what
the model does internally. It does rely on the engine pushing the predicate down
into the sources; DuckDB, Snowflake, and BigQuery all do for simple comparisons, but
a model wrapped around an opaque UDF will scan more than it strictly needs. Correct
and occasionally slow beats fast and occasionally wrong.

**Run [shadow mode](shadow.md) before you enable this.**

## Reading a plan that is wider than expected

The planner never guesses. When it says "whole dataset", something specific was
unprovable, and `--dirty` output tells you which:

```
widened to whole dataset (no provable partition bound):
  duckdb/mart.summary: via duckdb/raw.events {dt: *} [sql:summary.sql]
```

`{dt: *}` means the mapping for `dt` is `UNBOUNDED`. Common causes, in rough order
of frequency:

| Cause | Fix |
|---|---|
| Target has no partition spec | Declare one in `fathom.yml` |
| `MERGE` statement | Nothing to fix; row-level effects are genuinely unbounded |
| A UDF or expression the parser cannot follow | Declare the mapping, or accept it |
| Source and target disagree about grain | Fix whichever spec is wrong |
| Column lineage missing on that edge | Add a spec so `rollup` can derive a mapping |

`fathom doctor` lists all of them at once, without needing a plan.

## Cycles

A model that reads its own history is a cycle whose window grows on every pass. The
planner detects this and widens rather than looping:

```
cycles detected in: duckdb/mart.rolling
```

That is correct and coarse. If a self-referencing model matters to you, splitting
it into a windowed read of a separate table gives the planner something bounded to
work with.

## Options

| Flag | Meaning |
|---|---|
| `--dirty TABLE@FIELD=VALUE[,FIELD=VALUE]` | Seed a changed partition. Repeatable. |
| `--detect` | Discover seeds by scanning configured sources. |

Time values are ISO 8601 and truncate to the field's declared grain, so
`dt=2026-03-14T15:00:00` on a day-grained field means `2026-03-14`.
