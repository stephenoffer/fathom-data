# Configuration reference

`fathom.yml` in your project root, or any parent directory. Override with
`--config` or `FATHOM_CONFIG`.

Unknown keys are **errors**, not warnings. A silently ignored typo produces a
project that looks configured and plans wrong.

## Complete example

```yaml
version: 1
store: .fathom/fathom.db

system: snowflake            # identity system for bare table names
instance: xy12345            # account locator, workspace id, or project

storage_options:
  s3:
    key: "${AWS_ACCESS_KEY_ID}"
    secret: "${AWS_SECRET_ACCESS_KEY}"
  gs:
    token: "${GOOGLE_APPLICATION_CREDENTIALS}"

adapters:
  snowflake:
    account: xy12345
  bigquery:
    project: my-project
    region: region-us

datasets:
  - name: PROD.RAW.EVENTS
    adapter: snowflake
    watermark: _loaded_at
    partition:
      - {field: dt, grain: day}
      - {field: region}

  - name: s3://lake/raw/clicks
    adapter: storage
    template: "clicks/{yyyy}/{MM}/{dd}"
    partition:
      - {field: dt, grain: day}

  - name: PROD.GOLD.MONTHLY
    model: models/gold_monthly.sql
    partition:
      - {field: dt, grain: month}
      - {field: region}

lineage:
  - type: dbt
    manifest: target/manifest.json
  - type: openlineage
    events: s3://lineage/events/
  - type: sql
    paths: ["models/*.sql"]
    dialect: snowflake
  - type: adapter
    adapter: snowflake

policies:
  - dataset: PROD.ML.TRAINING_SET
    forbid: [pii]
    reason: not cleared for personal data
```

## Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `version` | int | `1` | Config schema version. Only `1` exists. |
| `store` | path | `.fathom/fathom.db` | Where the graph and profile history live. Relative to the config file. |
| `system` | string | `duckdb` | Identity system for bare table names. Decides identifier case folding. |
| `instance` | string | — | Account locator, workspace id, or project. Distinguishes two warehouses with the same table names. |
| `storage_options` | map | `{}` | Per-protocol options passed to fsspec. |
| `adapters` | map | `{}` | Per-adapter construction options. |
| `datasets` | list | `[]` | Declared datasets. |
| `lineage` | list | `[]` | Where lineage comes from. |
| `policies` | list | `[]` | Sink policies checked by `fathom label`. |

`system` values: `snowflake`, `databricks`, `bigquery`, `redshift`, `postgres`,
`duckdb`, `trino`, `hive`, `clickhouse`.

## Environment references

Any string value may contain `${VAR}` or `${VAR:-default}`, resolved at load.

```yaml
storage_options:
  s3:
    key: "${AWS_ACCESS_KEY_ID}"
    endpoint_url: "${S3_ENDPOINT:-https://s3.amazonaws.com}"
```

A missing variable with no default is an error naming the variable, not a silent
empty string. **Never put a credential literal in this file** — it exists to be
committed.

Live warehouse connections are not configured here at all. They are injected:

```python
project.register_runner("snowflake", DBAPIRunner(snowflake.connector.connect(...)))
```

## `datasets`

| Key | Meaning |
|---|---|
| `name` | Table name or URI. Resolved through `system` and `instance` when bare. |
| `adapter` | Which adapter owns it. Sniffed for paths; required for warehouses. |
| `partition` | The partition spec. See below. |
| `template` | Path template for non-Hive layouts. Storage adapter only. |
| `watermark` | Column marking new rows. Snowflake change detection only. |
| `location` | Storage location override. Databricks only. |
| `model` | Path to the SQL defining this dataset. |
| `sql` | Inline SQL, as an alternative to `model`. |

### `partition`

Three accepted forms:

```yaml
partition: [region]                             # one value field
partition: [{field: dt, grain: day}]            # one time field
partition:                                      # several, order matters
  - {field: dt, grain: day}
  - {field: region}
```

Grains: `hour`, `day`, `month`, `year`. A field without a grain is a value field —
it passes through unchanged and never participates in time windowing.

Omitting `partition` means the dataset is one unit, and any change to it rebuilds
all of it. `fathom doctor` flags this.

### `template`

For object storage layouts that are not self-describing:

```yaml
  - name: s3://lake/events
    adapter: storage
    template: "events/{yyyy}/{MM}/{dd}"
    partition: [{field: dt, grain: day}]
```

Placeholders: `{yyyy}`, `{MM}`, `{dd}`, `{HH}` build a timestamp; `{anything_else}`
captures a named value field. Hive layouts (`dt=2026-03-14/`) need no template.

Without a template, a non-Hive path binds nothing and the dataset widens to whole.
Guessing which path segment is a month is exactly the kind of inference that
silently corrupts a rebuild plan.

## `lineage`

Sources accumulate into one graph rather than competing. A dbt manifest and an
OpenLineage stream describing the same pipeline reinforce each other, because edges
are keyed by evidence.

| `type` | Required keys | Notes |
|---|---|---|
| `sql` | `paths`, optionally `dialect` | Globs are supported. |
| `dbt` | `manifest` | Path to `target/manifest.json`. |
| `openlineage` | `events` | File, directory, or object-storage prefix. |
| `adapter` | `adapter` | Pulls native lineage, falling back to query history. |

## `policies`

```yaml
policies:
  - dataset: PROD.ML.TRAINING_SET
    forbid: [pii, national_id]
    require: ["consent:training"]
    reason: not cleared for personal data
```

`forbid` fails when a label reaches this dataset. `require` fails when one is
absent. `reason` appears in the violation message, so write something a person
reading a failed CI job can act on.

Checked by `fathom label`, which exits non-zero on any violation.

## `publications`

Where a number stops being data and becomes a published claim.

```yaml
publications:
  - name: revenue/exec
    kind: dashboard
    instance: looker
    inputs: [gold.monthly]

  - name: 10-K/2026
    kind: filing
    instance: sec
    inputs: [gold.monthly, silver.revenue]
```

`kind` is one of `dashboard`, `report`, `filing`, `export`, `endpoint`, or
`notebook`. `instance` names the system it lives in — the BI tool, the regulator,
the recipient. `inputs` are the datasets it draws on, and at least one is required:
a publication with no inputs records nothing.

These are **declared, not discovered**. No BI tool exposes its queries uniformly, and
a guessed dashboard dependency is worse than an absent one — a restatement notice
built on a guess names the wrong people.

Read by `fathom impact`, which lists every published artefact downstream of a dataset
and drafts the notice. It exits non-zero when a filing or signed report is among them,
because the remedy there is an amendment rather than a refresh.

Publications are re-applied from config on every load rather than persisted as
ordinary edges, so deleting one here removes it from the graph. A declared artefact
that outlived its declaration would keep appearing in notices, and the value of a
notice is that it is current.

## `contracts`

What one team promises another about a dataset.

```yaml
contracts:
  - dataset: gold.orders
    producer: platform
    consumers: [finance, ml]
    columns: [order_id, amount, currency]
    max_staleness: 6h
    note: the close depends on this landing before 09:00
```

`producer` is required — a contract with no owner names nobody when it is breached,
which is the one thing a contract adds over the checks that already existed.

`columns` are promised to exist; extra columns are permitted. `max_staleness` takes a
number and a unit (`30m`, `6h`, `2d`, `1w`). A bare number is rejected, because every
config format that accepts one ends up with two readers silently disagreeing about
whether it meant seconds or hours.

Checked by `fathom contracts`, which exits non-zero on any error. **Severity follows
the blast radius**: the same missing column is a warning against a dataset nobody
consumes and an error against one three teams read. A promise with no evidence
supplied is reported as unchecked rather than passing silently.

## Adapter options

Passed straight to the adapter constructor:

```yaml
adapters:
  snowflake: {account: xy12345, limit: 50000}
  databricks: {workspace: dbc-123, limit: 50000}
  bigquery: {project: my-project, region: region-eu}
  storage: {suffixes: [".parquet", ".orc"]}
```

See [Adapters](adapters.md) for what each accepts.

## Precedence

For partition specs, most specific wins:

1. `partition:` in `fathom.yml`
2. `config.meta.fathom.partition` in a dbt model
3. What the adapter can infer (Iceberg transforms, BigQuery partition ids, Delta
   column names plus types)
4. Unpartitioned

For adapters: an explicit `adapter:` beats sniffing, which beats the project
`system` default. Erasure only acts on datasets in the first two categories — see
[erase](erase.md#why-unconfigured-datasets-block).
