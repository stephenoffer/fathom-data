# Databricks

Unity Catalog maintains lineage for you, and every table is a Delta table. Both
facts shape the adapter.

**Lineage** comes from `system.access.column_lineage` — column edges with no SQL
parsing, the same deal Snowflake offers from a different table.

**Change detection** delegates to the Delta transaction log at the table's storage
location. `DESCRIBE DETAIL` reports that location, so rather than reimplementing
snapshot diffing over `DESCRIBE HISTORY`, the Delta adapter does the work and gives
exact partition-level changes.

## Setup

```bash
pip install fathom-data databricks-sql-connector 'fathom-data[cloud]'
```

```yaml
version: 1
system: databricks
instance: dbc-abc123

adapters:
  databricks:
    workspace: dbc-abc123

storage_options:
  s3:
    key: "${AWS_ACCESS_KEY_ID}"
    secret: "${AWS_SECRET_ACCESS_KEY}"

datasets:
  - name: main.silver.events
    adapter: databricks
    partition:
      - {field: dt, grain: day}
      - {field: region}

lineage:
  - type: adapter
    adapter: databricks
```

```python
import os
from databricks import sql
from fathom.adapters import DBAPIRunner
from fathom.cli.project import Project

connection = sql.connect(
    server_hostname=os.environ["DATABRICKS_HOST"],
    http_path=os.environ["DATABRICKS_HTTP_PATH"],
    access_token=os.environ["DATABRICKS_TOKEN"],
)

with Project.load() as project:
    project.register_runner("databricks", DBAPIRunner(connection))
    project.ingest()
```

## Permissions

```sql
GRANT SELECT ON TABLE system.access.table_lineage  TO `fathom-service`;
GRANT SELECT ON TABLE system.access.column_lineage TO `fathom-service`;
GRANT SELECT ON TABLE system.query.history         TO `fathom-service`;

GRANT USE CATALOG ON CATALOG main TO `fathom-service`;
GRANT USE SCHEMA  ON SCHEMA main.silver TO `fathom-service`;
GRANT SELECT      ON TABLE  main.silver.events TO `fathom-service`;
```

System tables need `system.access` **enabled on the metastore** — an account admin
does this once, and it is not on by default.

Change detection reads the Delta log directly from cloud storage, so the principal
also needs read access to the table's location. If your workspace uses storage
credentials rather than direct access, pass the corresponding `storage_options`.

## Lineage

```python
events = list(adapter.fetch_lineage(since=token))
```

Two system tables, both needed:

- **`column_lineage`** gives the column edges.
- **`table_lineage`** catches dependencies where a table is read but no column of it
  reaches the output — a join key, a filter. Dropping those would leave a real
  dependency out of the graph.

### The two-hour lag

System table rows can take about two hours to appear. The resume token is held back
by that:

```python
token = adapter.lineage_token(events)    # newest event minus 2h
```

## Change detection

```python
changes = adapter.changed(dataset, since=token)
```

Under the hood:

1. `DESCRIBE DETAIL main.silver.events` → storage location and partition columns
2. `DeltaCatalog` reads `_delta_log` at that location
3. Commits after the token yield exact partition tuples

Cost is proportional to commits since the last run, not to table size. A petabyte
table with three commits costs three small file reads.

Compaction is correctly ignored: `OPTIMIZE` writes `dataChange: false`, and
rebuilding on it would be pure waste.

If `DESCRIBE DETAIL` returns no location — a view, or a table type without one —
the adapter raises rather than silently reporting no changes.

## Partition specs

Inferred from `DESCRIBE DETAIL` plus column types:

| Column type | Result |
|---|---|
| `date` | time field, day grain |
| `timestamp`, `timestamp_ntz` | time field, hour grain |
| anything else | value field |

A `string` column holding dates stays a value field. Guessing grain from content is
exactly the inference that silently changes what a rebuild covers.

Override in `fathom.yml` when the inference is wrong or too coarse:

```yaml
    partition: [{field: dt, grain: month}]
```

## Rebuilds

Databricks quotes identifiers with backticks, and the adapter renders accordingly:

```sql
DELETE FROM main.gold.monthly WHERE `dt` >= TIMESTAMP '2026-03-01 00:00:00'
                                AND `dt` <  TIMESTAMP '2026-04-01 00:00:00'
                                AND `region` = 'eu';
```

## Reading Delta directly

If you have storage access but not a SQL warehouse, skip the Databricks adapter and
point at the table location:

```yaml
datasets:
  - name: s3://lake/silver/events
    adapter: delta
    partition: [{field: dt, grain: day}]
```

That gives change detection and erasure scoping with no cluster running. You lose
Unity Catalog lineage, so pair it with a dbt manifest or OpenLineage events.

## Spark jobs

The Databricks adapter sees what Unity Catalog records, which covers SQL and the
DataFrame API but not arbitrary RDD work. For full coverage from Spark, enable the
OpenLineage Spark listener and ingest its events — see
[OpenLineage](openlineage.md).

## Known limitations

- Lineage covers table and column dependencies, never partition mappings. Those come
  from declared specs or from parsing model SQL.
- Views appear as datasets but have no storage location, so they cannot be
  change-detected.
- Streaming tables and materialized views are modeled as their underlying Delta
  tables; their refresh semantics are not.
