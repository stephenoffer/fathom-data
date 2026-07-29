# Error messages

What this library raises, what each one means, and what to do next. Grouped by the
type, because the type tells you where the problem is.

Every message here names the offending value and a next action. If you hit one that
does not, that is a bug worth reporting — a message you cannot act on has cost you
the same as no message at all.

---

## The exception types

| Type | Means | Where the fix is |
|---|---|---|
| `ConfigError` | `fathom.yml` is malformed or internally inconsistent | your config file |
| `StorageAccessError` | a filesystem or object store could not be read | credentials, network, or permissions |
| `AdapterUnavailable` | an adapter needs an optional dependency | `pip install` |
| `PlanError` | a plan could not be built from the graph and seeds | the graph, or the seeds |
| `SelectorError` | a selector expression could not be resolved | the selector string |
| `QueryError` | a warehouse rejected a query we sent | the connection or the SQL |
| `ValueError` | a value passed to the library is not valid | the call site |
| `FathomError` | the base of all of the above | — |

All of them derive from `FathomError`, so catching that catches everything this
library raises deliberately. It does **not** catch `FileNotFoundError`, which callers
legitimately treat as "absent" — see below.

---

## Storage and credentials

### `cannot read s3://…: no credentials found`

The store could not authenticate. Set `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`, or pass
`storage_options={"key": ..., "secret": ...}`.

Config files can reference the environment rather than holding secrets:

```yaml
storage_options:
  s3: {key: "${AWS_ACCESS_KEY_ID}", secret: "${AWS_SECRET_ACCESS_KEY}"}
```

### `cannot read …: the storage service rejected the request`

Usually one of: the wrong region, a bucket policy denying `ListBucket`, or
requester-pays not enabled on a bucket that needs it.

### Why is this not just "no data found"?

Because a missing prefix and an expired credential look identical from the call site,
and reporting the second as the first turns a five-minute fix into an afternoon. A
`StorageAccessError` means *we could not tell*, which is never safe to read as empty.

`FileNotFoundError` is the separate, honest case: the location was reachable and
nothing was there.

---

## Optional dependencies

### `the X adapter needs Y: pip install 'fathom-data[Z]'`

The message names the extra. The common ones:

```bash
pip install 'fathom-data[iceberg]'   # Iceberg manifests are Avro
pip install 'fathom-data[duckdb]'
pip install 'fathom-data[s3]'        # or [gcs], [azure], or [cloud] for all three
```

### `no adapter named 'x'; registered right now: [...]`

The registry only reflects modules that have been imported, so an adapter behind an
optional extra is absent until its package is installed and its module loads. The
message lists what *is* available and suggests the nearest name.

---

## Configuration

### `unknown key(s) in …`

A typo, or a key from a different version. The message lists the keys that block
accepts. Unknown keys are rejected rather than ignored, because a silently ignored
`partiton:` is a dataset with no partition spec and a plan that quietly rebuilds it
whole.

### `… is not a duration`

Durations take a number and a unit: `30d`, `12h`, `90m`. See
[configuration](configuration.md).

### `time partition field 'dt' requires a grain`

A time field must say how wide one bucket is, because a plan cannot tell a daily
partition from a monthly one without it:

```yaml
partition:
  - {field: dt, grain: day}
```

### `value partition field 'dt' must not carry a grain`

Only time fields have a grain. Either drop it, or declare the field as a time field
if it really does hold a date.

### `unknown grain 'daly'; expected one of: 'day', 'hour', 'month', 'year'`

Grains also accept the adjective (`daily`), the plural (`days`), and the abbreviation
(`d`). The message suggests the nearest canonical name.

### `X is referenced in the config but not set in the environment`

A `${VAR}` interpolation with nothing behind it. Refused rather than substituted
empty, because an empty credential fails much later and much less clearly.

### `the config file must contain a mapping at the top level`

The file parsed as a list or a scalar. Usually a stray leading `-`.

---

## Identity

### `'orders' is not a URI, so it must be a table — but no `system` was given`

`orders` in Snowflake and `orders` in DuckDB are different datasets, so a bare name
needs to say which system it belongs to. Either pass `system=`, or set the top-level
`system:` key in `fathom.yml` so the project supplies it.

### `X is a catalog dataset … not a location, so there are no bytes to open`

Something tried to read files for a warehouse table. Reach it through the engine
adapter for that system — or, if it is an external table over object storage, declare
the alias so both spellings resolve to one dataset.

### `… has no namespace, so there is no way to tell which system it lives in`

`DatasetId.parse` needs `system/name`, e.g. `duckdb/raw.events` or
`s3://lake/raw/events`. To build one from a bare table name, use
`fathom.normalize(name, system=...)`.

### `aliasing X to Y would form a cycle`

Two identities each declared as an alias of the other leaves neither canonical. Pick
one as canonical and alias the other to it, not both ways.

---

## Partitions and mappings

### `conflicting partition specs for X: already registered as …, now given …`

Two sources disagree about how a dataset is partitioned. This is refused rather than
resolved, because every mapping composed across that dataset would inherit the wrong
one. Fix the declaration, or alias the identities if they are in fact different
datasets.

### `output grain X is finer than input grain Y; refinement must be expressed as UNBOUNDED`

A monthly source feeding an hourly table. One dirty month could touch any hour inside
it, and claiming a narrower reach is the one error that serves stale data. Express it
as unbounded.

### `empty window [6, 0]`

Offsets run earliest to latest. A 7-day trailing window is `(0, 6)` — or, better,
`TimeWindow.trailing("dt", 7, "day")`, which cannot be written backwards.

### `a trailing window over 'dt' must cover at least one bucket`

`length` is how many buckets the aggregate spans, so a 7-day rolling metric is
`length=7`. Zero and negative lengths have no meaning.

### `span from … to … exceeds 100,000 buckets`

A range so large that enumerating it individually is never what a plan wants. Narrow
the range, or use a coarser grain.

### `no partition field 'date'; this dataset is partitioned by 'dt', 'region'`

A field name that does not exist in the spec. The message lists the ones that do and
suggests the nearest.

### `this dataset is unpartitioned, so it has no field 'dt'`

The field is not missing — the spec is. Declare one in `fathom.yml` to plan at
partition granularity rather than whole-dataset.

---

## Plans

### `nothing to plan from`

Neither `--dirty` nor `--detect` was given. Seeds are the input, not a filter.

### `X is not in the graph, so nothing will propagate from it`

A warning, not an error: the plan still runs, and contains only what you named. Almost
always a typo — the message suggests the nearest known dataset.

### `no lineage extracted from any configured source`

The `lineage` block found nothing. In order of likelihood: `paths` matches no files;
the `dialect` is one the parser does not know, which reads as zero edges rather than
as an error; or every statement was a form nothing can be proven from.

### `the selector '…' matched no edges`

An empty selector result usually means the name did not resolve rather than that
nothing matched. Run `fathom lineage` unfiltered to see what is there.
`fathom explain selector` covers the syntax.

---

## Profiles and checks

### `nothing could be profiled`

Only path-backed datasets are profiled directly — Parquet, Delta, or Iceberg under a
filesystem or object store. A warehouse table is profiled through its engine adapter,
which needs a connection supplied with `register_runner`.

### `no path-backed datasets to check`

`check` compares against stored profiles, so `fathom profile` has to run first. Note
that a dataset profiled for the first time has nothing to be compared against yet —
that is not a clean result, it is no result.

### `no labels could be inferred`

Inference reads profiles, not schemas. Run `fathom profile` first: a column name alone
is deliberately not enough evidence to label a column.

---

## Erasure

### `writing a proof needs a secret salt`

Identifiers are low-entropy, so an unsalted digest identifies the subject about as
well as the raw value does — and the proof is the artifact handed to people who must
not learn who they were. Set `FATHOM_SALT` to a per-organization secret.

### An erasure that reports itself incomplete

Not a failure of the tool. Storage under WORM or Object Lock cannot destroy anything,
and a model that was trained on the subject still retains them. Both are reported
rather than rounded up to `complete: true`. See [erase](erase.md).

---

## Shadow mode

### `MISSED PARTITIONS ARE A SOUNDNESS FAILURE`

The planner called a partition clean and a full rebuild proved otherwise. This is the
one result that must never be accepted. Do not enable apply mode; open an issue with
the graph and the seeds.

---

## See also

- [Troubleshooting](troubleshooting.md) — warnings that are not errors, and what they cost
- [FAQ](faq.md) — behaviour that surprises people but is working as intended
- `fathom explain <topic>` — the concept behind a term in any of these messages
