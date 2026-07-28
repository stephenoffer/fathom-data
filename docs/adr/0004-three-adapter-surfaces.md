# 4. Three adapter surfaces, and capability degradation

Status: accepted

## Context

An early draft assumed every source has a query log and a catalog. That is true for
Snowflake and BigQuery and false for most of the matrix. Spark, Ray, Dask, and Polars
produce no query log. A raw Parquet prefix in S3 has no catalog and no partition
metadata. Flink jobs never complete, so there is no "query finished" event to read.

Supporting 25 providers only works if the per-provider cost is small.

## Decision

Split adapters by what they can answer, not by vendor:

- **Engine** — execution plans and query logs. Spark, Trino, Flink, ClickHouse,
  DataFusion, DuckDB, Polars.
- **Catalog** — table and partition metadata. Iceberg, Delta, Hudi, Glue, Unity,
  `INFORMATION_SCHEMA`.
- **Storage** — objects, events, inventory manifests. S3, GCS, ADLS, R2, MinIO, local.

"What depends on what" comes from SQL and is ~90% shared, because sqlglot already
normalizes every dialect on the matrix. "What changed" comes from the storage layer
and collapses into a handful of strategies rather than 25 implementations.

Adapters declare capabilities instead of implementing everything. The planner degrades
rather than failing: `LIST_DIFF` with `Pushdown.NONE` is slower and coarser than
`SNAPSHOT_DIFF` with `SKETCHES`, and both work.

Change-detection strategies, ranked by cost per detected change:

1. `SNAPSHOT_DIFF` — Iceberg/Delta/Hudi commits. Exact, cost proportional to commits.
2. `EVENTS` — S3 EventBridge, GCS Pub/Sub, ADLS Event Grid. Needs infra setup.
3. `INVENTORY` — S3 Inventory, Storage Insights. Daily, near-free at petabyte scale.
4. `PARTITION_MTIME` — catalog-reported modification times.
5. `LIST_DIFF` — fine under a few hundred thousand objects, ruinous above.
6. `PROFILE_DELTA` — last resort.

## Consequences

- One SQL extractor serves every engine. A ClickHouse `toStartOfMonth` normalizes to
  the same `TimestampTrunc` node as a Postgres `date_trunc` with no dialect-specific
  code, which is the entire argument for routing through sqlglot.
- Engine listeners need cluster-level installation — a Trino plugin or a Spark JAR is
  a platform-team change, not a `pip install`. The query-log fallback must stay viable
  for every engine so there is always a zero-privilege path.
- The reference adapter (`LocalStorage`) uses the *weakest* strategy on purpose. If
  everything works on `LIST_DIFF` plus mtime, richer adapters are strictly faster
  rather than differently shaped.
- Column-level lineage is achievable for Spark, Trino, ClickHouse, DataFusion, Polars,
  DuckDB, and Flink SQL. Ray, Dask, Beam, and Flink DataStream give dataset level only.
  Document which is which; overpromising granularity loses trust in month two.
