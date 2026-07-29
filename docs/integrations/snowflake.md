# Snowflake

Snowflake is the best case for lineage and the worst case for change detection.

**Lineage is free and exact.** `ACCOUNT_USAGE.ACCESS_HISTORY` records, per query,
which columns of which objects fed which columns of which objects. No SQL parsing,
no dialect edge cases, no ambiguity about unqualified columns.

**Partitions do not exist.** Micro-partitions are an internal detail with no
addressable identity, so "which partition changed" has no native answer.

## Setup

```bash
pip install fathom-data snowflake-connector-python
```

```yaml
# fathom.yml
version: 1
system: snowflake
instance: xy12345

adapters:
  snowflake:
    account: xy12345

datasets:
  - name: PROD.RAW.EVENTS
    adapter: snowflake
    watermark: _loaded_at          # see "change detection" below
    partition:
      - {field: DT, grain: day}
      - {field: REGION}

lineage:
  - type: adapter
    adapter: snowflake
```

```python
import os, snowflake.connector
from fathom.adapters import DBAPIRunner
from fathom.cli.project import Project

connection = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    authenticator="externalbrowser",     # or a key pair; avoid passwords
    warehouse="ANALYTICS_WH",
    role="FATHOM_ROLE",
)

with Project.load() as project:
    project.register_runner("snowflake", DBAPIRunner(connection))
    result = project.ingest()
    print(result.summary())
```

Credentials never go in `fathom.yml`. The file declares *shape*; the connection is
injected.

## Permissions

```sql
CREATE ROLE FATHOM_ROLE;

-- Lineage and query history
GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE TO ROLE FATHOM_ROLE;

-- Reading watermarks for change detection
GRANT USAGE ON DATABASE PROD TO ROLE FATHOM_ROLE;
GRANT USAGE ON ALL SCHEMAS IN DATABASE PROD TO ROLE FATHOM_ROLE;
GRANT SELECT ON ALL TABLES IN DATABASE PROD TO ROLE FATHOM_ROLE;

-- A warehouse to run those queries
GRANT USAGE ON WAREHOUSE ANALYTICS_WH TO ROLE FATHOM_ROLE;
```

`ACCESS_HISTORY` requires **Enterprise Edition or higher**. On Standard, the
adapter falls back to `QUERY_HISTORY` plus SQL parsing, which is coarser but works.

For erasure you additionally need `DELETE` on the target tables — grant that
separately and deliberately, not as part of the read role.

## Lineage

```python
for event in adapter.fetch_lineage(since=token):
    print(event.src, "->", event.dst, event.columns)
```

`objects_modified[].columns[].directSources[]` maps onto a column edge with no
translation. Tables read but not projected — a join key, a filter source — come
from `base_objects_accessed` and produce an edge with no column detail. Dropping
them would leave a real dependency out of the graph.

### The three-hour lag

`ACCOUNT_USAGE` views lag by up to about three hours. That is fine for planning a
nightly backfill and wrong for intraday decisions.

The adapter holds the resume token back by the lag:

```python
token = adapter.lineage_token(events)    # newest event minus 3h
```

Advancing to the newest row seen would permanently skip rows that had not landed in
the view yet. `capabilities.freshness_lag` states this rather than leaving it to be
discovered.

## Change detection

### With a watermark (recommended)

Declare a column that marks new rows:

```yaml
    watermark: _loaded_at
```

One cheap query gives real partition granularity:

```sql
SELECT DISTINCT "DT", "REGION", MAX("_loaded_at") AS _fathom_high_water
FROM PROD.RAW.EVENTS
WHERE "_loaded_at" > TO_TIMESTAMP_LTZ(%(since)s)
GROUP BY "DT", "REGION"
```

The watermark must be **monotonic** — set on insert and update, never backdated. A
non-monotonic watermark is a spec that lies, and it will show up as a shadow-mode
miss.

Index or cluster on it if the table is large.

### Without one

Falls back to `INFORMATION_SCHEMA.TABLES.LAST_ALTERED`, which answers only "did this
table change at all". The result is one unbounded partition: correct, and coarse.

That is usually enough for source tables and never enough for large ones. Adding a
watermark column is the single highest-value change you can make to a Snowflake
project.

## Rebuilds

```python
adapter.register_model(gold, "SELECT ... FROM PROD.SILVER.EVENTS GROUP BY 1, 2", spec)
statements = adapter.render_rebuild(gold, plan.partitions(gold))
```

```sql
DELETE FROM PROD.GOLD.MONTHLY WHERE "DT" >= TIMESTAMP '2026-03-01 00:00:00'
                                AND "DT" <  TIMESTAMP '2026-04-01 00:00:00';
INSERT INTO PROD.GOLD.MONTHLY SELECT * FROM (<model>) AS _fathom_rebuild WHERE <same>;
```

Snowflake pushes those predicates into micro-partition pruning, so the scan is
bounded even though the model is wrapped.

## Identifier case

Snowflake folds unquoted identifiers **up**. These are all the same dataset:

```python
normalize_table("orders", system="snowflake", instance="ac1",
                default_database="db", default_schema="public")
normalize_table("DB.PUBLIC.Orders", system="snowflake", instance="ac1")
# -> snowflake://ac1/DB.PUBLIC.ORDERS
```

Quoted identifiers keep their case: `db.public."MixedCase"` stays `MixedCase`.

Write table names in your config the way Snowflake stores them (uppercase) to avoid
surprises when reading `ACCESS_HISTORY` output.

## Cost

- `ACCESS_HISTORY` and `QUERY_HISTORY` queries run on your warehouse. Keep the time
  window narrow and let the resume token do its job.
- Watermark queries are one indexed scan per source per run.
- `limit` (default 100,000) caps rows per pull. Raise it for a backfill; lower it
  if the ingest warehouse is small.

## Known limitations

- Column lineage covers what `ACCESS_HISTORY` records, which excludes some
  procedural and dynamic SQL.
- Streams and tasks are not modeled; a stream-driven pipeline needs its lineage
  declared or emitted via OpenLineage.
- `capabilities.partition_aware` is `False`, because Snowflake genuinely has no
  partitions. Everything partition-shaped comes from your declarations.
