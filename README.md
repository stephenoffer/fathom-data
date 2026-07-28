# fathom

**Lineage, partition-scoped invalidation, profiling, and policy propagation for data platforms.**

> Status: alpha. All four verbs work end to end against DuckDB, Delta, Iceberg, and
> local Parquet, with 217 tests covering them. Adapters for the wider engine matrix
> are not written yet — see [Roadmap](#roadmap).

---

## What this is

Most data tooling answers one of three questions and stops:

- *What depends on what?* — lineage catalogs
- *Is my data still correct?* — observability and testing tools
- *What is this column, and who may use it?* — governance platforms

Each is expensive to answer alone, and each is much cheaper once you have the other two.
`fathom` computes all three from one metadata plane.

**Two artifacts:**

| | |
|---|---|
| **dependency graph** | column-level edges between datasets, each carrying a *partition mapping* |
| **profile history** | distributions, vocabularies, and cardinalities, per partition, over time |

**Four verbs over them:**

| | |
|---|---|
| `plan` | given what changed at the source, rebuild exactly the affected partitions |
| `check` | detect drift, reconcile across systems — and attribute the cause upstream |
| `label` | infer what a column means, propagate policy labels along graph edges |
| `erase` | locate a subject's data in every derived table, and destroy it |

## Why they belong together

Built separately, each of these is worse:

- **Profiling is unaffordable without the graph.** Scanning whole tables nightly costs
  real warehouse credits. Profiling only the partitions the graph says changed is what
  makes continuous profiling viable at all.
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
pip install fathom-data          # imports as `fathom`
pip install 'fathom-data[iceberg]'   # Iceberg manifests are Avro, so this needs a library

# 1. build a dependency graph from your model SQL
fathom ingest models/*.sql \
  --spec raw.events:dt:day     --spec raw.events:region \
  --spec gold.monthly:dt:month --spec gold.monthly:region

# 2. ask what one day of new source data invalidates
fathom plan --dirty 'raw.events@dt=2026-03-14,region=eu'
```

```
duckdb/raw.events       dt=2026-03-14T00:00:00/region=eu
duckdb/silver.events    dt=2026-03-14T00:00:00/region=eu
duckdb/gold.monthly     dt=2026-03-01T00:00:00/region=eu
```

One dirty day and region resolves to one day downstream and one *month* in the
rollup, still scoped to `region=eu`. Everything else stays untouched.

```bash
fathom changed s3-mirror/events    # what moved since last run (auto-detects Delta/Iceberg)
fathom profile lake/events --save  # footer-only profile; reads no data pages
fathom check   lake/events         # drift vs the last profile, with upstream attribution
fathom label   --sink ml/training_set:forbid=pii
fathom erase   --subject u1 --key-column user_id --origin raw.events --proof proof.json
fathom shadow                      # accumulated savings, and the miss count
```

## The core invariant

The planner may over-invalidate. It must never under-invalidate.

```
apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }
```

Every operation in [`partitions.py`](src/fathom/partitions.py) preserves this. Anything
unprovable — an opaque UDF, a dialect we cannot parse, a `MERGE`, a partition spec
mismatch, a cycle — widens to `UNBOUNDED`, costing compute and never costing
correctness.

Erasure carries the **mirrored** invariant: it may under-delete and refuse, but must
never over-delete. Hence dry-run by default and an explicit refusal on WORM storage.

## Shadow mode

Nobody should trust a new tool to decide what *not* to rebuild on the strength of a
README. So run it alongside your existing full rebuild and grade it:

```python
report = shadow.run(engine, plan, [SILVER, GOLD], store=store)
print(report.summary())
```

```
shadow: SOUND across 2 dataset(s), 78% of partitions skipped
```

Two numbers, per dataset: **savings** (partitions skipped) and **missed** (partitions
called clean that a full rebuild proved dirty). `missed` must be zero, and `fathom
shadow` exits non-zero the moment it isn't. Accumulate weeks of evidence before
anything writes.

## Partition mappings

Every graph edge carries a mapping answering: *if this input partition is dirty, which
output partitions are dirty?* Three field-level forms cover real warehouse SQL.

```python
from fathom import Grain, TimeWindow, Passthrough

TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)     # identity
TimeWindow("dt", 0, 0, Grain.DAY, Grain.MONTH)   # daily source -> monthly rollup
TimeWindow("dt", 0, 6, Grain.DAY, Grain.DAY)     # 7-day trailing window
Passthrough("region")                            # non-time dimension carried through
```

They **compose** along paths and **join** where paths reconverge, forming a lattice
whose top is "the whole dataset". Grain conversion always rounds outward, and windows
only ever coarsen — a coarse source feeding a finer table widens rather than
pretending to a precision we don't have.

## Adapters

Three surfaces, because "what depends on what" and "what changed" come from different
places and neither assumes SQL.

| Adapter | Kind | Lineage | Change detection | Notes |
|---|---|---|---|---|
| `delta` | catalog | declared | `SNAPSHOT_DIFF` | Reads `_delta_log` JSON directly — no dependency |
| `iceberg` | catalog | declared | `SNAPSHOT_DIFF` | Needs the `iceberg` extra; opens tables with no catalog service |
| `duckdb` | engine | `QUERY_LOG` | — | Renders and executes partition-scoped rebuilds |
| `local` | storage | declared | `LIST_DIFF` | The reference implementation and conformance target |

Adapters declare capabilities rather than implement everything. One reporting
`LIST_DIFF` and `Pushdown.NONE` still works — it is just slower and coarser. The
`local` adapter deliberately uses the *weakest* strategy so richer adapters are
strictly faster rather than differently shaped.

Identities follow the OpenLineage dataset naming convention, so `s3a://`, `s3://`,
`/dbfs/mnt/...`, and `catalog.schema.table` collapse to one node and what we emit
interoperates with that ecosystem.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **M0** core | IR, partition lattice, identity, graph planner, footer profiler, adapter protocol | done |
| **M1** check | profiling, drift, attribution, Delta + Iceberg change detection, SQLite store | done |
| **M2** plan | invalidation planner, shadow mode, DuckDB rebuild + apply | done |
| **M3** label | inference, propagation, sink enforcement | done |
| **M4** erase | partition-scoped erasure, proof artifacts, WORM refusal | done |
| **M5** breadth | Spark/Trino listeners, S3/GCS/ADLS storage, ClickHouse, dbt package | next |

Nothing writes to your data by default. `plan` prints; `erase` dry-runs. Executing
needs an explicit engine binding, which belongs in a pipeline rather than a shell.

## Design decisions

Recorded as ADRs in [`docs/adr/`](docs/adr/):

1. [Python first, with a seam for a Rust core](docs/adr/0001-python-first-with-a-rust-seam.md)
2. [Soundness invariants, and which direction each errs](docs/adr/0002-soundness-invariants.md)
3. [Dataset identity follows OpenLineage naming](docs/adr/0003-dataset-identity.md)
4. [Three adapter surfaces, and capability degradation](docs/adr/0004-three-adapter-surfaces.md)
5. [Shadow mode is the adoption mechanism](docs/adr/0005-shadow-mode-before-apply.md)
6. [Erasing a subject is two operations, and order matters](docs/adr/0006-erasure-is-not-deletion.md)

## Development

```bash
uv venv && uv pip install -e '.[dev]'
pytest -q          # 217 tests
ruff check src tests && mypy
```

The property tests in `tests/test_partitions.py` enforce the soundness invariant
across generated grain and window combinations. They are not optional — they caught
a real composition bug during development, where `hour → day → hour` lost the
truncation anchor and pointed at the wrong hour.

## License

Apache-2.0
