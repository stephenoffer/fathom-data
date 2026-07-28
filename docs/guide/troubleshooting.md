# Troubleshooting

Start with `fathom doctor`. It reports everything that makes plans *worse* rather
than failing them, which is the category of problem that otherwise gets discovered
as "why is it rebuilding everything".

## "It rebuilds everything"

This is the most common report, and it is almost always one of five things.

### No partition spec on the target

```
duckdb/gold.monthly: no partition spec, so any change rebuilds the whole dataset
```

An unpartitioned dataset has exactly one partition. Declare a spec:

```yaml
  - name: gold.monthly
    partition: [{field: dt, grain: month}]
```

### The mapping is unbounded

```
duckdb/raw.events -> duckdb/mart.summary: unbounded mapping (sql:summary.sql);
every source change rebuilds the whole target
```

Look at `{dt: *}` in `fathom lineage`. Causes:

| Cause | What to do |
|---|---|
| `MERGE` statement | Nothing. Row-level effects are genuinely unbounded. |
| A UDF or expression the parser cannot follow | Rewrite the partition expression, or accept it |
| Source and target specs disagree about grain | Fix whichever is wrong |
| The source has no spec | Declare one; `rollup` needs both sides |
| Transformed value field (`UPPER(region)`) | Not a passthrough, so not provable |

### The parser could not read the SQL

```
  ! unparseable (ParseError): ...
```

The dialect is probably wrong. `lineage.dialect` must match the engine that ran the
statement, not the one you are querying from. BigQuery in particular takes its
`DATE_TRUNC` arguments in the opposite order from everyone else — with the wrong
dialect it parses to nonsense and correctly widens.

### Timezone mismatch (fixed, but worth knowing)

If a plan seeded from a warehouse never matches partitions profiled from files,
this used to be why: aware and naive datetimes compare unequal while printing
identically. Partition keys are now normalized to naive UTC at a single point. If
you see this symptom, you are on an old build.

### Non-Hive paths without a template

```python
key_from_path("events/2026/01/15/part-0.parquet", spec)   # dt=ANY
```

Guessing which path segment is a month is exactly the kind of inference that
silently corrupts a plan. Declare it:

```yaml
    template: "events/{yyyy}/{MM}/{dd}"
```

## "It found no lineage"

```
Error: no lineage extracted; check the `lineage` block in your config
```

- **`type: sql`** — do the globs match? Paths are relative to the config file, not
  the working directory. A `SELECT` with no target is correctly ignored.
- **`type: dbt`** — has `dbt compile` run? A manifest without `compiled_code` still
  yields dataset edges but no column detail.
- **`type: openlineage`** — do the events have both `inputs` and `outputs`? Events
  with neither carry no lineage and are dropped. Failed runs are skipped by default.
- **`type: adapter`** — on Snowflake, `ACCESS_HISTORY` is Enterprise Edition and up.
  On Standard, the adapter falls back to `QUERY_HISTORY` and SQL parsing.

## "Detect reports the same partitions every run"

The resume token is not advancing. Check that the store is writable and that you are
not passing `--store` inconsistently between runs.

For object storage specifically, this used to happen when several objects shared the
newest timestamp. The token now carries the boundary etags, so a quiet dataset
converges to reporting nothing. If it does not, the objects probably have no
modification time at all, in which case they are reported every run by design —
skipping them silently would be worse.

## "Cannot read s3://…"

```
cannot read s3://lake/events: NoCredentialsError: Unable to locate credentials
  no credentials found. Set AWS_PROFILE / AWS_ACCESS_KEY_ID, or pass
  storage_options={'key': ..., 'secret': ...}
```

Access failures are never reported as "nothing found", because a credential problem
diagnosed as an empty dataset sends people to debug their SQL. Options go in the
config:

```yaml
storage_options:
  s3:
    key: "${AWS_ACCESS_KEY_ID}"
    secret: "${AWS_SECRET_ACCESS_KEY}"
    endpoint_url: "${S3_ENDPOINT:-}"    # MinIO and other S3-compatible stores
```

Common causes beyond credentials: wrong region, a bucket policy denying
`ListBucket`, or requester-pays not enabled.

## "Query failed"

```
query failed: ProgrammingError: Object 'SNOWFLAKE.ACCOUNT_USAGE.ACCESS_HISTORY' does not exist
  SELECT query_id, query_start_time, objects_modified ...
  check the object name and that the role can see it
```

The role needs `IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE`. See the
[Snowflake guide](../integrations/snowflake.md#permissions) for the exact grants,
and the equivalents for [Databricks](../integrations/databricks.md#permissions) and
[BigQuery](../integrations/bigquery.md#permissions).

## "Config error: unknown key"

```
unknown key(s) in datasets[1]: partiton. Valid keys: adapter, location, model,
name, partition, role, sql, template, watermark
```

Deliberate. A silently ignored typo produces a config that looks right and plans
wrong.

## "Erasure plan is INCOMPLETE"

Expected whenever any target cannot be provably erased. Two common reasons:

- **No adapter configured.** Erasure only acts on datasets with an explicit
  `adapter:`, or path datasets whose format was sniffed. Falling back to the project
  default is a guess, and erasure does not act on guesses.
- **WORM or Object Lock.** Deletion is impossible by design; crypto-shredding is the
  remaining option.

Do not report a request fulfilled while the plan says incomplete. That is what the
flag is for.

## "Shadow mode reports missed partitions"

Stop. This is a soundness failure, and it means the planner would have served stale
data. Do not enable apply mode.

Capture the plan, the seeds, and the dataset, and open an issue. Every widening rule
exists to make this impossible, so a real miss is a bug in the lattice, in an
adapter's change detection, or in a partition spec that does not match reality.

The most likely non-bug cause is a **spec that lies**: a table declared as
day-partitioned whose `dt` column actually holds month starts, or a watermark column
that is not monotonic.

## Getting more detail

```bash
fathom doctor              # everything degrading plans, at once
fathom lineage             # edges, mappings, and column detail
fathom detect              # what each source reports, and its token
```

```python
plan.reasons[dataset]      # why this dataset is dirty, per contributing edge
plan.widened               # which datasets lost precision
plan.cyclic                # which are in cycles
result.notes               # everything ingest could not fully resolve
```
