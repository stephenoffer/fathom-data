# fathom documentation

## Start here

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

- `01-local-lakehouse` — the whole loop on local Parquet
- `02-shadow-mode` — proving the planner before trusting it
- `03-dbt-project` — building a graph from a dbt manifest
- `04-erasure` — locating and destroying a subject's data
