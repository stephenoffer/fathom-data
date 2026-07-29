# BigQuery

The mirror image of Snowflake.

**Change detection is free and exact.** `INFORMATION_SCHEMA.PARTITIONS` reports a
`last_modified_time` per partition, which is precisely the question, answered
directly.

**Lineage costs a parse.** BigQuery has no native column lineage in
`INFORMATION_SCHEMA` — that lives in Dataplex behind a separate API — so lineage
comes from parsing job SQL, with `referenced_tables` available as a coarser
cross-check.

## Setup

```bash
pip install fathom-data google-cloud-bigquery
```

```yaml
version: 1
system: bigquery

adapters:
  bigquery:
    project: my-project
    region: region-us          # must match where the jobs ran

datasets:
  - name: my-project.raw.events
    adapter: bigquery
    # partition spec is inferred from live partition ids

lineage:
  - type: adapter
    adapter: bigquery
```

```python
from google.cloud import bigquery
from google.cloud.bigquery import dbapi
from fathom.adapters import DBAPIRunner
from fathom.cli.project import Project

connection = dbapi.Connection(bigquery.Client(project="my-project"))

with Project.load() as project:
    project.register_runner("bigquery", DBAPIRunner(connection))
    project.ingest()
```

## Permissions

```
roles/bigquery.metadataViewer      # INFORMATION_SCHEMA
roles/bigquery.jobUser             # to run the queries
roles/bigquery.dataViewer          # on datasets you profile or rebuild
```

Reading `INFORMATION_SCHEMA.JOBS` across a project needs
`roles/bigquery.resourceViewer` as well. Without it you see only your own jobs,
which produces a lineage graph containing your ad-hoc queries and nothing else —
a confusing failure worth checking first.

## Change detection

```python
changes = adapter.changed(dataset, since=token)
```

```sql
SELECT table_name, partition_id, last_modified_time, total_rows
FROM `my-project.raw.INFORMATION_SCHEMA.PARTITIONS`
WHERE table_name = @table
```

One query, exact partition granularity, no scan of the data. This is the best
change detection any adapter here offers.

### Partition ids

Grain is encoded in the id format, and that is the only place BigQuery records it:

| `partition_id` | Grain |
|---|---|
| `2026031409` | hour |
| `20260314` | day |
| `202603` | month |
| `2026` | year |
| `__NULL__` | rows whose partition column is null |
| `__UNPARTITIONED__` | streaming buffer, or an unpartitioned table |

`__NULL__` is a **real partition**, not an error. Treating it as unparseable would
silently drop rows from every rebuild, so it binds to `None` and renders as
`IS NULL` in predicates.

`__UNPARTITIONED__` widens to the whole dataset, which is correct.

Integer-range partitioning yields non-date ids; those bind to `ANY` and widen.
Declare a spec if you need them narrower.

## Lineage

Two routes, and using both is reasonable.

**Parsing job SQL** gives column edges and partition mappings:

```yaml
lineage:
  - type: adapter
    adapter: bigquery
```

**`referenced_tables`** gives dataset-level edges with no parsing at all. Coarser,
but never wrong — BigQuery records exactly which tables a job read, including
through UDFs and scripts the parser cannot follow.

```python
for event in adapter.fetch_lineage(since=token):   # referenced_tables route
    print(event.src, "->", event.dst)
```

### The dialect trap

BigQuery takes its `DATE_TRUNC` arguments in the opposite order from every other
dialect:

```sql
DATE_TRUNC(dt, MONTH)      -- BigQuery
DATE_TRUNC('month', dt)    -- everyone else
```

With the wrong `dialect` set, the second form parses to nonsense in BigQuery mode.
The extractor correctly refuses and widens rather than inventing a grain, but you
lose the mapping. Make sure `dialect: bigquery`.

## Rebuilds

```sql
DELETE FROM `my-project.gold.monthly`
WHERE `dt` >= TIMESTAMP '2026-03-01 00:00:00'
  AND `dt` <  TIMESTAMP '2026-04-01 00:00:00';
INSERT INTO `my-project.gold.monthly`
SELECT * FROM (<model>) AS _fathom_rebuild WHERE <same>;
```

BigQuery prunes on the partition column, so the wrapped model still scans only the
targeted partitions. Check the dry-run byte estimate the first time — if it does not
prune, the predicate is not on the partitioning column.

## Cost

`INFORMATION_SCHEMA` queries are free of slot cost but do count against concurrent
query limits. `INFORMATION_SCHEMA.JOBS` over a wide time window on a busy project
can be slow; keep the resume token doing its job and lower `limit` if needed.

Profiling reads Parquet footers only if you point at exported files. Profiling a
native BigQuery table means querying it, which is billed — the storage adapter path
is the cheap one.

## Case sensitivity

BigQuery is case-sensitive for dataset and table names, so nothing is folded:

```python
normalize_table("Proj.Dataset.Table", system="bigquery")   # Proj.Dataset.Table
```

Write names exactly as they exist.

## Known limitations

- No native column lineage; parsing is the only route. Dataplex lineage is not yet
  supported.
- `region` must match where the jobs ran. Jobs in `region-eu` are invisible to a
  `region-us` query, and the result is an empty graph rather than an error.
- Integer-range and ingestion-time partitioning are partially supported: change
  detection works, grain inference does not.
- The streaming buffer appears as `__UNPARTITIONED__` and widens.
