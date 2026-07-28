# dbt

dbt already knows the dependency graph, the warehouse each model lands in, and
often the partitioning. All of it is in `target/manifest.json` after any
`dbt compile`. Reading it is far more reliable than parsing a directory of `.sql`
files full of unresolved `ref()` calls.

## Setup

```yaml
version: 1
system: snowflake          # ignored: the manifest's adapter_type wins
instance: xy12345

lineage:
  - type: dbt
    manifest: target/manifest.json
```

```bash
dbt compile
fathom ingest
```

```
14 edge(s) from 9 statement(s), 2 note(s)
  ! 3 incremental model(s) have no partition spec, so every change to them widens
    to a full rebuild; declare one under config.meta.fathom.partition
```

## What dbt gives, and what it does not

| | |
|---|---|
| **Dataset edges** | From `depends_on`, exactly and authoritatively |
| **Relation names** | Resolved through the target's database and schema |
| **Partition specs** | From `config.partition_by` on BigQuery and Spark |
| **Column lineage** | Not recorded — recovered by parsing `compiled_code` |
| **Partition mappings** | Not recorded — same |

dbt supplies the skeleton; parsing fills in the detail. Disable parsing for a much
faster first look at a very large project:

```python
ingest_dbt("target/manifest.json", parse_sql=False)
```

## Partition specs per adapter

**BigQuery** — read directly:

```yaml
{{ config(partition_by={'field': 'dt', 'data_type': 'date', 'granularity': 'month'}) }}
```

**Spark and Databricks** — column names plus types:

```yaml
{{ config(partition_by=['dt', 'region']) }}
```

Grain comes from the column's `data_type` in the manifest, so document your columns
in `schema.yml` or the grain is unknown and the field becomes a value field.

**Snowflake** — nothing to read. dbt has no way to express it, which is what the
escape hatch is for.

## The escape hatch

```yaml
{{ config(
    materialized='incremental',
    meta={'fathom': {'partition': [
      {'field': 'dt', 'grain': 'day'},
      {'field': 'region'}
    ]}}
) }}
```

A `meta.fathom.partition` declaration beats anything dbt records. Putting it next to
the model rather than in a separate config file is the point — the two cannot drift.

For sources, the same key works in `sources.yml`:

```yaml
sources:
  - name: raw
    tables:
      - name: events
        meta:
          fathom:
            partition:
              - {field: dt, grain: day}
```

## Incremental models

The warning above is the one to act on. An incremental model with no partition spec
widens every change to a full rebuild, which defeats the purpose of it being
incremental.

If your model already has an `is_incremental()` block filtering on a date, that date
column is almost always the partition field:

```sql
{{ config(materialized='incremental', meta={'fathom': {'partition': [
    {'field': 'event_date', 'grain': 'day'}]}}) }}

select ...
{% if is_incremental() %}
  where event_date >= (select max(event_date) from {{ this }})
{% endif %}
```

Note that the `is_incremental` filter itself does not appear in the manifest's
`compiled_code` for a full refresh, so the declaration is doing the work.

## What is skipped

Tests, analyses, and exposures are not part of the data dependency graph a rebuild
plan cares about. Models, sources, snapshots, and seeds are kept.

## Combining with the platform adapter

The most useful configuration uses both. dbt gives the graph; the platform gives
change detection:

```yaml
lineage:
  - type: dbt
    manifest: target/manifest.json

datasets:
  - name: PROD.RAW.EVENTS
    adapter: snowflake
    watermark: _loaded_at
    partition: [{field: DT, grain: day}]
```

```bash
dbt compile && fathom ingest
fathom plan --detect
```

Sources accumulate into one graph rather than competing, so adding OpenLineage
events on top is also fine — edges are keyed by evidence.

## Identifier case

The manifest's `adapter_type` selects the identity system, so Snowflake relations
fold up and Databricks relations fold down automatically. You do not need to match
dbt's casing in your `fathom.yml`; `system:` in the config is overridden by the
manifest.

## In CI

```yaml
- run: dbt compile
- run: fathom ingest
- run: fathom doctor
- run: fathom label            # fails on a policy violation
```

`fathom doctor` after `dbt compile` catches a model that lost its partition config
in review, before it silently starts rebuilding everything.

## Known limitations

- Ephemeral models are inlined by dbt, so they do not appear as datasets. Their
  lineage is preserved through the models that use them.
- Python models have no `compiled_code` SQL, so they yield dataset-level edges only.
- A manifest from `dbt parse` rather than `dbt compile` has no compiled SQL, and the
  graph will have no column detail. The note in the ingest output says so.
