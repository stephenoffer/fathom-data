# fathom documentation

## Start here

- **[What fathom is](guide/what-is-fathom.md)** — the problem it solves, in plain
  terms, plus the six words the rest of these docs assume. Read this first if you
  have not used a lineage tool before.
- **[Getting started](guide/getting-started.md)** — a working project in ten minutes
- **[Concepts](guide/concepts.md)** — partition mappings, the soundness invariant, and why they matter
- **[Configuration](guide/configuration.md)** — the complete `fathom.yml` reference

## The four verbs

| Guide | Answers |
|---|---|
| [plan](guide/plan.md) | Given what changed at the source, what must be rebuilt? |
| [check](guide/check.md) | What drifted, and what upstream caused it? |
| [label](guide/label.md) | What does this column mean, and what policy applies? |
| [erase](guide/erase.md) | Where is this subject's data, and can it be destroyed? |

Plus **[shadow mode](guide/shadow.md)**, which is how you decide whether to trust
`plan` at all. Read it before wiring anything into a pipeline that writes.

## The same four verbs, over AI assets

**[AI assets](guide/ai.md)** — a model, feature view, vector index, prompt, and eval
set are datasets, so the graph, planner, profiler, policy engine, and eraser already
work on them. Retraining becomes an invalidation question, re-embedding costs what
changed, a leaked eval set is a reachability check, and an erasure request that
reaches a model says so instead of reporting `complete: true`.

## Beyond the four

| Guide | Answers |
|---|---|
| [completeness](guide/completeness.md) | Which partitions should exist and do not? |
| [usage, value, impact](guide/value.md) | Who reads this, what has it cost, and what have we already published from it? |
| [contracts, risk](guide/contracts.md) | What did we promise whom, and what do the columns jointly reveal? |

`check` and `plan` both read data that arrived. `completeness` is the only thing that
can see the partition that never did — it has no profile to drift and no rows to fail
an expectation. The `value` guide covers the three questions that live outside the
warehouse: who reads a table, what it has cost, and which dashboards and filings a
restatement would touch. The `contracts` guide covers the two checks that read the
same profiles as everything else and answer what per-column analysis structurally
cannot: who a breach is owed to, and which columns identify nobody alone.

## Platforms

| Platform | Lineage | Change detection |
|---|---|---|
| [Snowflake](integrations/snowflake.md) | ACCESS_HISTORY, column-level | declared watermark |
| [Databricks](integrations/databricks.md) | Unity Catalog system tables | Delta log |
| [BigQuery](integrations/bigquery.md) | job SQL, parsed | per-partition mtime |
| [dbt](integrations/dbt.md) | manifest + compiled SQL | via the underlying platform |
| [OpenLineage](integrations/openlineage.md) | events from Spark, Flink, Trino, Airflow, Dagster | n/a |
| [S3, GCS, Azure](integrations/cloud-storage.md) | declared | Delta/Iceberg snapshots, or LIST |

## Reference

- **[Adapters](guide/adapters.md)** — the capability matrix, and how to write a new one
- **[Troubleshooting](guide/troubleshooting.md)** — what the warnings mean and how to fix them
- **[Architecture decisions](adr/)** — why things are the way they are

## Examples

Runnable projects live in [`examples/`](../examples/). Each is self-contained and
tested in CI, so they cannot rot:

- [`01_local_lakehouse.py`](../examples/01_local_lakehouse.py) — the whole loop on local Parquet
- [`02_shadow_mode.py`](../examples/02_shadow_mode.py) — proving the planner before trusting it
- [`03_dbt_project.py`](../examples/03_dbt_project.py) — building a graph from a dbt manifest
- [`04_erasure.py`](../examples/04_erasure.py) — locating and destroying a subject's data
- [`05_cloud_storage.py`](../examples/05_cloud_storage.py) — Delta on object storage
- [`06_worth_keeping.py`](../examples/06_worth_keeping.py) — what never arrived, who reads it, what it cost
