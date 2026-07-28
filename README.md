# fathom

**Lineage, partition-scoped invalidation, profiling, and policy propagation for data platforms.**

> Status: beta. All four verbs work end to end against Snowflake, Databricks,
> BigQuery, DuckDB, Delta, Iceberg, and object storage, with ~470 tests. Nothing
> writes to your data by default.

[Documentation](docs/) · [Getting started](docs/guide/getting-started.md) · [Examples](examples/)

---

## What this is

Most data tooling answers one of three questions and stops:

- *What depends on what?* — lineage catalogs
- *Is my data still correct?* — observability and testing tools
- *What is this column, and who may use it?* — governance platforms

Each is expensive to answer alone, and each is much cheaper once you have the other
two. `fathom` computes all three from one metadata plane.

**Two artifacts:**

| | |
|---|---|
| **dependency graph** | column-level edges between datasets, each carrying a *partition mapping* |
| **profile history** | distributions, ranges, and cardinalities, per partition, over time |

**Four verbs over them:**

| | |
|---|---|
| [`plan`](docs/guide/plan.md) | given what changed at the source, rebuild exactly the affected partitions |
| [`check`](docs/guide/check.md) | detect drift, and attribute the cause upstream |
| [`label`](docs/guide/label.md) | infer what a column means, propagate policy labels along graph edges |
| [`erase`](docs/guide/erase.md) | locate a subject's data in every derived table, and destroy it |

## Why they belong together

Built separately, each of these is worse:

- **Profiling is unaffordable without the graph.** Scanning whole tables nightly
  costs real warehouse credits. Profiling only the partitions the graph says changed
  is what makes continuous profiling viable at all.
- **Drift detection is useless without lineage.** "`revenue` moved 8%" is an alert.
  "`revenue` moved because `fx_rates` changed three hops upstream" is a diagnosis.
- **Annotation dies without inference.** Nobody hand-labels 40,000 columns. A profile
  that sees a `latitude` column whose values top out at 4,000 rejects the name-based
  guess before a human ever sees it.
- **Erasure is ruinous without partition scoping.** Deleting one subject from a
  lakehouse is a rewrite-the-world operation until you know which files in which
  derived tables actually hold their rows.

## Quickstart

```bash
pip install fathom-data              # imports as `fathom`
pip install 'fathom-data[cloud]'     # + S3, GCS, Azure
pip install 'fathom-data[iceberg]'   # + Iceberg (its manifests are Avro)
```

```bash
fathom init          # writes a starter fathom.yml
fathom ingest        # build the dependency graph
fathom plan --dirty 'raw.events@dt=2026-03-14,region=eu'
```

```
duckdb/raw.events       dt=2026-03-14T00:00:00/region=eu
duckdb/silver.events    dt=2026-03-14T00:00:00/region=eu
duckdb/gold.monthly     dt=2026-03-01T00:00:00/region=eu
```

One dirty day and region resolves to one day downstream and one *month* in the
rollup, still scoped to `region=eu`. Everything else is untouched.

```bash
fathom doctor        # what would silently make plans worse
fathom detect        # what changed since the last run
fathom profile       # footer-only profiles; reads no data pages
fathom check         # drift, with upstream attribution
fathom label         # inferred labels, propagated, checked against policy
fathom erase --subject u1 --key-column user_id --origin raw.events --proof p.json
fathom shadow        # accumulated savings, and the miss count
```

Everything is configured in [`fathom.yml`](docs/guide/configuration.md), so partition
specs live in one place rather than drifting across invocations.

## The core invariant

The planner may over-invalidate. It must never under-invalidate.

```
apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }
```

Every operation in [`partitions.py`](src/fathom/partitions.py) preserves this.
Anything unprovable — an opaque UDF, a dialect we cannot parse, a `MERGE`, a
partition spec mismatch, a cycle — widens to `UNBOUNDED`, costing compute and never
costing correctness.

Erasure carries the **mirrored** invariant: it may under-delete and refuse, but must
never over-delete. Hence dry-run by default and an explicit refusal on WORM storage.

## Shadow mode

Nobody should trust a new tool to decide what *not* to rebuild on the strength of a
README. Run it alongside your existing full rebuild and grade it:

```python
report = shadow.run(engine, plan, [silver, gold], store=store)
print(report.summary())
```

```
shadow: SOUND across 2 dataset(s), 78% of partitions skipped
```

Two numbers per dataset: **savings** (partitions skipped) and **missed** (partitions
called clean that a full rebuild proved dirty). `missed` must be zero, and `fathom
shadow` exits non-zero the moment it isn't. It runs at zero risk — the full rebuild
happens either way — so accumulate weeks of evidence before anything writes.

[More on shadow mode](docs/guide/shadow.md).

## Partition mappings

Every graph edge carries a mapping answering: *if this input partition is dirty,
which output partitions are dirty?*

```python
from fathom import Grain, TimeWindow, Passthrough

TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)     # identity
TimeWindow("dt", 0, 0, Grain.DAY, Grain.MONTH)   # daily source -> monthly rollup
TimeWindow("dt", 0, 6, Grain.DAY, Grain.DAY)     # 7-day trailing window
Passthrough("region")                            # non-time dimension carried through
```

They **compose** along paths and **join** where paths reconverge, forming a lattice
whose top is "the whole dataset". Grain conversion rounds outward, and windows only
ever coarsen — a coarse source feeding a finer table widens rather than pretending
to a precision we do not have.

## Platforms

| Platform | Lineage | Change detection | Guide |
|---|---|---|---|
| Snowflake | ACCESS_HISTORY, column-level | declared watermark | [guide](docs/integrations/snowflake.md) |
| Databricks | Unity Catalog system tables | Delta log | [guide](docs/integrations/databricks.md) |
| BigQuery | job SQL, parsed | per-partition mtime | [guide](docs/integrations/bigquery.md) |
| dbt | manifest + compiled SQL | via the platform | [guide](docs/integrations/dbt.md) |
| Spark, Flink, Trino, Airflow, Dagster | OpenLineage events | via the platform | [guide](docs/integrations/openlineage.md) |
| S3, GCS, Azure, MinIO, HDFS | declared | Delta/Iceberg snapshots, or LIST | [guide](docs/integrations/cloud-storage.md) |

Adapters declare capabilities rather than implement everything. One reporting
`LIST_DIFF` and `Pushdown.NONE` still works — it is slower and coarser, and the
planner degrades instead of failing. See the
[capability matrix](docs/guide/adapters.md).

Identities follow the OpenLineage dataset naming convention, so `s3a://`, `s3://`,
`/dbfs/mnt/...`, and `catalog.schema.table` collapse to one node, and what we emit
interoperates with that ecosystem.

## Design decisions

Recorded as ADRs in [`docs/adr/`](docs/adr/):

1. [Python first, with a seam for a Rust core](docs/adr/0001-python-first-with-a-rust-seam.md)
2. [Soundness invariants, and which direction each errs](docs/adr/0002-soundness-invariants.md)
3. [Dataset identity follows OpenLineage naming](docs/adr/0003-dataset-identity.md)
4. [Three adapter surfaces, and capability degradation](docs/adr/0004-three-adapter-surfaces.md)
5. [Shadow mode is the adoption mechanism](docs/adr/0005-shadow-mode-before-apply.md)
6. [Erasing a subject is two operations, and order matters](docs/adr/0006-erasure-is-not-deletion.md)

## What this does not do

Stated plainly, because the gaps matter more than the features:

- **It does not execute rebuilds by default.** `plan` prints; `erase` dry-runs.
  Applying needs an engine binding you supply deliberately.
- **It does not give column lineage everywhere.** Snowflake and Databricks give it
  natively; SQL parsing gives it where the SQL is parseable; Ray, Dask, and Beam give
  dataset level only.
- **It does not cover backups, replicas, or snapshots.** An erasure proof covers what
  the configured adapters can see, and says so.
- **It does not verify itself.** That is what shadow mode is for, and why it reports
  its own miss rate.

## Development

```bash
uv venv && uv pip install -e '.[dev]'
pytest -q
ruff check src tests && ruff format --check src tests && mypy
```

The property tests in `tests/test_partitions.py` enforce the soundness invariant
across generated grain and window combinations. They are not optional — they caught
a real composition bug where `hour → day → hour` lost the truncation anchor.

The examples in [`examples/`](examples/) are executed by the test suite, and
[`tests/test_docs.py`](tests/test_docs.py) checks that every documented command,
adapter, and config key actually exists. Documentation that is not executed rots.

## License

Apache-2.0
