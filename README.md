# fathom

**Lineage, partition-scoped invalidation, profiling, and policy propagation for data platforms.**

> Status: beta. All four verbs work end to end against Snowflake, Databricks,
> BigQuery, DuckDB, Delta, Iceberg, and object storage, with 850+ tests. Nothing
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

## Models are datasets

A model is produced from named inputs, in slices that can be rebuilt independently,
with a version history somebody will need explained, and it can contain a person's
data. So is a feature view, a vector index, a prompt, an eval set, and an agent run.
Give them `DatasetId`s and the graph, the planner, the profiler, the policy engine,
and the eraser already work on them. There is no second graph for ML — which matters,
because a second graph is a graph that disagrees with the first one.

The four verbs do not change. What they answer does:

| | table | model, index, prompt |
|---|---|---|
| `plan` | rebuild the affected partitions | retrain the affected models; **re-embed only the chunks that changed** |
| `check` | detect drift, attribute it upstream | embedding drift, training/serving skew, **eval sets that leaked into training** |
| `label` | propagate PII along edges | propagate **licence, consent, and purpose** into training sets and prompts |
| `erase` | delete the subject's rows | name every **model that retains them**, and what would actually discharge it |

Five questions nobody could previously answer without a person and a week:

- **What was this model trained on?** `fathom.ai.training.data_bill_of_materials`
  walks the closure and states its own gaps. `training_data_summary` generates the
  prose an EU AI Act filing asks for, from lineage, so it cannot go stale.
- **Did the eval leak?** Contamination is a reachability property. If the eval set
  and the training data share an ancestor, or one is derived from the other, the
  score measures memorization. `fathom.ai.evals.contamination` checks it in the graph
  and reports `clean`, `suspect`, or `contaminated` — never rounding up.
- **What do I actually have to re-embed?** A corpus reindexed nightly pays the full
  embedding bill whether or not anything changed. `fathom.ai.vectors.reindex_plan`
  compares content digests and prices the difference. An embedding-model version
  change short-circuits to a full reindex, because vectors from two spaces are not
  comparable and a partially reindexed store returns confidently wrong neighbours.
- **What was in the context window?** `fathom.ai.rag.ContextManifest` records it, and
  `enforce_context` checks what reached a third-party endpoint against the same sink
  policy the `label` verb uses. Personal data reaching a model through an interpolated
  prompt variable is still a transfer.
- **Which models still hold this person?** `fathom.ai.unlearning.exposures` names
  them and the route. Deleting rows does not remove a subject from weights, and
  `completeness_statement` says so in the first sentence rather than emitting
  `complete: true` over a model that is still serving traffic.

Nothing here claims to remove a subject from a model. It claims to tell you, exactly,
what you have not yet done. Full walkthrough: [**AI assets**](docs/guide/ai.md).

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
fathom completeness  # partitions that should exist and do not
fathom profile       # footer-only profiles; reads no data pages
fathom check         # drift, with upstream attribution
fathom label         # inferred labels, propagated, checked against policy
fathom erase --subject u1 --key-column user_id --origin raw.events --proof p.json
fathom usage         # who reads each dataset, over a stated window
fathom value         # lifetime cost against observed reads
fathom impact        # what has already been published downstream
fathom shadow        # accumulated savings, and the miss count
```

Everything is configured in [`fathom.yml`](docs/guide/configuration.md), so partition
specs live in one place rather than drifting across invocations.

## The core invariant

The planner may over-invalidate. It must never under-invalidate.

```
apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }
```

Every operation in [`partitions.py`](src/fathom/core/partitions.py) preserves this.
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

## The API

The verbs are the product; the library is what you build on. The tree is deep and
the import surface is flat — packages follow the lifecycle so a new capability has
one obvious home, and the names you actually type are re-exported at the top.

```python
from fathom import query, selectors, diff, metrics, render, emit, cost, ai
```

```
core/       the IR: identity, grains, the partition lattice, codecs, errors
  util/     digests, Markdown, clocks, text measurement — no knowledge of datasets
ingest/     how the graph is learned: SQL, native events, dbt, OpenLineage
graph/      the graph, traversal, selection, diffing, coverage
  plan/     what a plan costs, and how it turns into runnable work
observe/    profiles, expectations, schema diffs, shadow mode, freshness
govern/     labels, erasure, licences, consent, purpose limitation
ai/         models, features, vectors, RAG context, prompts, evals, agents
adapters/   everything that talks to another system, by surface
  engines/  catalogs/  storage/  conformance contracts for each
store/      persistence, behind a protocol so SQLite is a default and not a premise
report/     rendering out: Mermaid, DOT, Markdown, OpenLineage, DataHub, compliance
cli/        the command line, project config, and `fathom.yml`
```

| module | what it answers |
|---|---|
| `query` | ancestors, descendants, paths, cycles, blast radius, column-level walks, subgraphs |
| `selectors` | dbt's selection syntax — `+model+`, `2+model`, `@model`, `tag:pii`, `ns:snowflake` |
| `diff` | what a pull request did to the graph; **narrowing a mapping fails the gate, widening does not** |
| `metrics` | coverage — what fraction of the graph is precise enough to plan on, which predicts the saving |
| `render` | Mermaid, Graphviz, D2, PlantUML, Cytoscape, JSON, Markdown |
| `emit` | OpenLineage, DataHub, Atlas, OpenMetadata payloads — pure functions, no clients |
| `quality` | expectations over profiles, and `learn()` to generate a suite from what was observed |
| `seasonal` | the same, bucketed by a cycle — 1,000 rows is normal on Monday and an anomaly on Saturday |
| `completeness` | the partition that **never arrived**, which has no profile to drift and no rows to fail |
| `history` | who narrowed this edge, and when — the question every incident review asks |
| `sinks` | the last hop: which dashboards, reports, and **filings** a restatement would touch |
| `usage` | who actually reads a dataset, over a window it never lets you forget |
| `cost` | plan cost against full-rebuild cost, per partition, per byte, per token, plus carbon |
| `lifetime` | what a dataset has cost since it existed, set against whether anyone reads it |
| `freshness` | transitive freshness — a table rebuilt five minutes ago is not fresh if its input is four days old |
| `schedule` | plans arranged into waves and contiguous batches, exported as tasks, a DAG, or a shell script |
| `contracts` | what one team promised another, and **who a breach is owed to** |
| `reidentification` | columns that identify nobody alone and everybody together |
| `ai` | assets, training, features, vectors, rag, prompts, evals, agents, attribution, unlearning |
| `govern` | licences, consent and purpose limitation, residency; records in `report.compliance` |

**The layering is a test, not a convention.** `tests/test_layering.py` reads every
import in the package and fails if one points upward — `core` may know nothing,
`graph` may know `core`, `ai` may know all of them. It also fails if any directory
grows past twelve modules or ten subpackages, and if a package `__init__` does not
say what belongs in it. Structure that nothing checks lasts about a month.

Two things worth calling out, because they are the difference between a lineage tool
and an inventory of one:

```python
metrics.coverage(graph).summary()
# coverage: 82% of datasets specced, 91% of edges bounded, 64% column-level,
#           88% of field mappings provable
```

An edge with an `UNBOUNDED` mapping is in the graph and contributes nothing to a
plan. `field_ratio` is very close to the ceiling on how much of a rebuild any plan
can skip, so it is the number to publish and the number to move.

```python
diff.diff_graphs(before, after).is_safe   # False if an edge vanished or narrowed
```

Widening costs compute and is always correct. Narrowing and removal are the two ways
a graph edit serves stale data, and they are the two a merge gate should stop.

## Design decisions

Recorded as ADRs in [`docs/adr/`](docs/adr/):

1. [Python first, with a seam for a Rust core](docs/adr/0001-python-first-with-a-rust-seam.md)
2. [Soundness invariants, and which direction each errs](docs/adr/0002-soundness-invariants.md)
3. [Dataset identity follows OpenLineage naming](docs/adr/0003-dataset-identity.md)
4. [Three adapter surfaces, and capability degradation](docs/adr/0004-three-adapter-surfaces.md)
5. [Shadow mode is the adoption mechanism](docs/adr/0005-shadow-mode-before-apply.md)
6. [Erasing a subject is two operations, and order matters](docs/adr/0006-erasure-is-not-deletion.md)
7. [Packages follow the lifecycle, and the layering is enforced](docs/adr/0007-layered-packages.md)

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
- **It does not remove a subject from a model.** It names every model that retains
  them and what would discharge the obligation — retraining, or crypto-shredding data
  that was encrypted per subject. Approximate unlearning is reported where a team says
  it is available, and never as complete.
- **It does not know which chunk a token came from.** `ContextManifest` records what
  was *retrieved*, not what the model *used*. `unused_context` is therefore a cost
  signal, not an attribution claim.
- **It does not give legal advice.** The compliance artefacts are evidence bundles
  for whoever signs them, generated from lineage and carrying their own gaps.

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
