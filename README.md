# fathom

**Lineage, partition-scoped invalidation, profiling, and policy propagation for data platforms.**

> Status: pre-alpha. The partition algebra and graph planner are implemented and tested.
> Adapters beyond the local/DuckDB reference pair are not written yet. See [Roadmap](#roadmap).

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
  that sees three-character uppercase strings drawn from ISO 4217 proposes
  `currency_code` and a human confirms it.
- **Erasure is ruinous without partition scoping.** Deleting one subject from a lakehouse
  is a rewrite-the-world operation until you know which files in which derived tables
  actually hold their rows.

## The core invariant

The planner may over-invalidate. It must never under-invalidate.

```
apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }
```

Every operation in [`partitions.py`](src/fathom/partitions.py) preserves this. Anything
unprovable — an opaque UDF, a dialect we cannot parse, a partition spec mismatch — widens
to `UNBOUNDED`, costing compute and never costing correctness. Erasure carries the
mirrored invariant: it may under-delete and refuse, but must never over-delete.

## Partition mappings

Every graph edge carries a mapping answering: *if this input partition is dirty, which
output partitions are dirty?* Three field-level forms cover real warehouse SQL.

```python
from fathom import Grain, PartitionMapping, TimeWindow, Passthrough

# identity: daily source -> daily model
TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)

# rollup: daily source -> monthly aggregate
TimeWindow("dt", 0, 0, Grain.DAY, Grain.MONTH)

# trailing 7-day window: one dirty day taints the six that follow
TimeWindow("dt", 0, 6, Grain.DAY, Grain.DAY)

# a non-time dimension carried through unchanged
Passthrough("region")
```

They **compose** along paths and **join** where paths reconverge, forming a lattice
whose top element is "the whole dataset". Grain conversion always rounds outward:
six days becomes two months, not zero, because six days from an arbitrary start can
straddle a month boundary.

## Three adapter surfaces

The reason this spans a warehouse, a lakehouse, and a raw bucket is that "what depends
on what" and "what changed" come from different places, and neither assumes SQL.

```
  ENGINE adapters        CATALOG adapters        STORAGE adapters
  execution plans        table / partition       objects, events,
  query logs             metadata                inventory manifests
       │                      │                        │
       └──────────┬───────────┴────────────────────────┘
                  ▼
        dataset identity normalizer
        (s3a:// ≡ s3:// ≡ /dbfs/mnt ≡ catalog.schema.table)
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
  dependency graph      profile history
```

Adapters declare capabilities rather than implement everything. One reporting
`LIST_DIFF` and `Pushdown.NONE` still works — it is just slower and coarser.

Identities follow the OpenLineage dataset naming convention, so what `fathom` emits
interoperates with that ecosystem instead of inventing a fourth spelling.

## Install

```bash
pip install fathom-data      # imports as `fathom`
```

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **M0** core | IR, partition lattice, identity normalizer, graph planner, footer profiler, adapter protocol | in progress |
| **M1** check | profiling, drift, reconciliation over S3/GCS/ADLS + Iceberg/Delta, Snowflake, Databricks | |
| **M2** graph | Spark and Trino listeners, ClickHouse, DataFusion, SQL lineage, incremental profiling | |
| **M3** plan | invalidation planner in shadow mode, backfill planner, dbt package | |
| **M4** label | type inference, annotation propagation, sink enforcement | |
| **M5** erase | partition-scoped erasure with proof artifacts, contract gate, adapter breadth | |

Shadow mode is the adoption strategy for M3: run alongside a full rebuild and report
two numbers — partitions skipped, and partitions wrongly called clean. A tool that
publishes its own miss rate is one people will eventually trust to apply.

## Design decisions

Recorded as ADRs in [`docs/adr/`](docs/adr/).

## License

Apache-2.0
