# Getting started

Ten minutes to a working project. We will build a dependency graph, ask what one
day of new data invalidates, and find out why the answer is sometimes "everything".

## Install

```bash
pip install fathom-data            # imports as `fathom`
pip install 'fathom-data[cloud]'   # + S3, GCS, and Azure
pip install 'fathom-data[iceberg]' # + Iceberg (its manifests are Avro)
```

## 1. Create a project

```bash
mkdir analytics && cd analytics
fathom init
```

That writes `fathom.yml`. Open it and describe your datasets:

```yaml
version: 1
system: duckdb          # the identity system for bare table names

datasets:
  - name: raw.events
    partition:
      - {field: dt, grain: day}
      - {field: region}

  - name: silver.events
    partition:
      - {field: dt, grain: day}
      - {field: region}

  - name: gold.monthly
    partition:
      - {field: dt, grain: month}
      - {field: region}

lineage:
  - type: sql
    paths: ["models/*.sql"]
    dialect: duckdb
```

**Why declare partitions by hand?** Because they cannot be reliably inferred.
Snowflake has no partitions to read. Delta records that `dt` is a partition column
but not whether it buckets by day or by month. Guessing wrong does not raise an
error — it silently changes what a rebuild covers. See
[Concepts](concepts.md#why-specs-are-declared).

## 2. Build the graph

With two model files in `models/`:

```sql
-- models/silver.sql
CREATE TABLE silver.events AS
SELECT dt, region, user_id, amount FROM raw.events;
```

```sql
-- models/gold.sql
CREATE TABLE gold.monthly AS
SELECT DATE_TRUNC('month', dt) AS dt, region, SUM(amount) AS revenue
FROM silver.events GROUP BY 1, 2;
```

```bash
fathom ingest
```

```
2 edge(s) from 2 statement(s)
```

Inspect what it found:

```bash
fathom lineage
```

```
duckdb/raw.events -> duckdb/silver.events {dt: dt@day, region: region=} [sql:silver.sql]
    dt -> dt
    region -> region
    user_id -> user_id
    amount -> amount
duckdb/silver.events -> duckdb/gold.monthly {dt: dt@day->month, region: region=} [sql:gold.sql]
    dt -> dt
    region -> region
    amount -> revenue
```

`dt@day->month` is a *partition mapping*: a dirty day in `silver` dirties the month
that contains it. `region=` means the region value passes through unchanged.

## 3. Ask what a change invalidates

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

One day and region in the source becomes one day downstream and one *month* in the
rollup, still scoped to `region=eu`. Everything else is untouched.

## 4. Check for problems

```bash
fathom doctor
```

```
config   /path/to/analytics/fathom.yml
store    /path/to/analytics/.fathom/fathom.db
system   duckdb
datasets 3, edges 2

no problems found
```

`doctor` reports things that make plans *worse* rather than failing them — a
missing partition spec, an edge with no column lineage, a mapping that had to widen.
Every item is a reason the planner would rebuild more than it should. Run it
whenever a plan looks broader than you expected.

## 5. Trust it before you use it

Do not wire `plan` into anything that writes yet. Run [shadow mode](shadow.md)
alongside your existing full rebuild first, for as long as it takes to convince
you. It reports two numbers: how many partitions the plan skipped, and how many it
wrongly called clean. The second must be zero.

```bash
fathom shadow
```

```
runs        14
partitions  38 planned of 420 total
savings     91%
missed      0

no missed partitions across every run recorded here
```

## Where to go next

- Working against a real platform: [Snowflake](../integrations/snowflake.md),
  [Databricks](../integrations/databricks.md), [BigQuery](../integrations/bigquery.md),
  [dbt](../integrations/dbt.md)
- Data in object storage: [S3, GCS, and Azure](../integrations/cloud-storage.md)
- Understanding what the planner is doing: [Concepts](concepts.md)
- A plan that is wider than expected: [Troubleshooting](troubleshooting.md)
