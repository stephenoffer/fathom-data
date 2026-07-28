# Concepts

Four ideas hold the project up. Everything else is adapters.

## 1. Datasets have one identity

A Spark job writes `s3a://lake/raw/events`. A Trino query reads the same bytes as
`hive.raw.events`. A notebook reads `/dbfs/mnt/lake/raw/events`. A Snowflake
external table calls it `RAW.EVENTS`.

Unless all four collapse to one node, the dependency graph is four disconnected
fragments and everything downstream is worthless — the planner finds no path, drift
attribution finds no upstream, erasure misses derived copies.

Identities follow the [OpenLineage naming convention](../adr/0003-dataset-identity.md):
a namespace locating the system, a name locating the dataset inside it.

```python
from fathom import normalize

normalize("s3a://lake//raw/events/")      # s3://lake/raw/events
normalize("gcs://lake/raw")               # gs://lake/raw
normalize("orders", system="snowflake",
          instance="ac1", default_database="db", default_schema="public")
                                          # snowflake://ac1/DB.PUBLIC.ORDERS
```

Protocol aliases, Azure's two hostnames, duplicate separators, bucket case, and
per-system identifier folding all normalize automatically. Snowflake folds unquoted
identifiers **up**, Databricks folds them **down**, BigQuery is case-sensitive.

What does *not* normalize automatically is anything not discoverable from either
reference — an external Hive table pointing at an S3 prefix, or a DBFS mount. Those
need a declaration, or an OpenLineage producer that emits the `symlinks` facet.

## 2. Every edge carries a partition mapping

A dependency alone tells you `gold` depends on `silver`. That is enough to rebuild
`gold` entirely, which is what most tools do. The interesting question is *which
partitions* of `gold` a given partition of `silver` affects.

Three field-level forms cover real warehouse SQL:

```python
from fathom import Grain, TimeWindow, Passthrough, UNBOUNDED

TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)     # identity
TimeWindow("dt", 0, 0, Grain.DAY, Grain.MONTH)   # daily source -> monthly rollup
TimeWindow("dt", 0, 6, Grain.DAY, Grain.DAY)     # 7-day trailing window
Passthrough("region")                            # value carried through unchanged
UNBOUNDED                                        # we could not prove anything
```

They **compose** along paths and **join** where paths reconverge, forming a lattice
whose top element is "the whole dataset".

```python
from fathom import compose, PartitionMapping

trailing = PartitionMapping.of(dt=TimeWindow("dt", 0, 6, Grain.DAY, Grain.DAY))
rollup   = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.MONTH))
compose(trailing, rollup)      # {dt: dt[+0,+3]@day->month}
```

Grain conversion always rounds **outward**. Seven days from an arbitrary start can
straddle a month boundary, so the window widens rather than risking a miss.

Windows only ever coarsen. A monthly source feeding an hourly table is a
*refinement*, whose honest answer is "some large part of the month", so it widens
to `UNBOUNDED` instead of claiming a precision that does not exist.

## 3. The planner may over-invalidate, never under-invalidate

```
apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }
```

Every operation preserves this. Anything unprovable widens to `UNBOUNDED`:

| Situation | Result |
|---|---|
| Unparseable SQL | whole dataset |
| Opaque UDF, dynamic SQL | whole dataset |
| `MERGE` with a correlated condition | whole dataset |
| Partition spec mismatch between two edges | whole dataset |
| Self-referencing model (a cycle) | whole dataset, after a bounded number of passes |
| Cross product past the enumeration cap | widest dimensions collapse to ANY |

Each of those is a place where a plausible guess was available and is deliberately
refused. Over-invalidating wastes money; under-invalidating serves stale data
silently, for weeks. The two are not symmetric.

Erasure carries the **mirrored** invariant — it may under-delete and refuse, never
over-delete — which is why it dry-runs by default and refuses outright on WORM
storage. See [ADR 2](../adr/0002-soundness-invariants.md).

## 4. Partition keys are predicates, not values

A dirty set is a set of *predicates*, where each field is bound to a concrete value
or to `ANY`:

```python
KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")   # one partition
KeyPredicate.of(dt=datetime(2026, 3, 14), region=ANY)    # every region, one day
KeyPredicate()                                            # the whole dataset
```

This is what lets the planner say "I know the day but not the region" without
enumerating every region, and what makes "rebuild everything" a normal value rather
than a special case. One predicate *subsumes* another when it covers everything the
other does, and that ordering is how the worklist knows it has converged.

Time values are always **naive UTC**. Warehouse drivers return timezone-aware
datetimes and file paths produce naive ones; the two compare unequal while printing
identically, so everything normalizes at a single point.

## Why specs are declared

Partition specs are the one thing you must write down. The reasons differ per
platform and none of them are fixable by trying harder:

- **Snowflake** has no partitions. Micro-partitions have no addressable identity.
- **Delta** records that `dt` is a partition column, not whether it buckets by day
  or month.
- **Iceberg** does record the transform, so its specs *are* inferred.
- **BigQuery** encodes grain in the partition id format, so its specs are inferred.
- **Raw Parquet** in Hive layout gives you the column name and a value to parse; the
  grain is a guess.

Where a spec can be read, it is. Where it cannot, declaring it takes one line and
guessing it wrong is invisible.

## What this does not do

Stated plainly, because the gaps matter more than the features:

- **It does not execute rebuilds by default.** `plan` prints. Applying needs an
  engine binding you supply deliberately.
- **It does not give column lineage everywhere.** Snowflake and Databricks give it
  natively; SQL parsing gives it where the SQL is parseable; Ray, Dask, and Beam
  give dataset level only.
- **It does not model backups or replicas.** An erasure proof covers what the
  configured adapters can see. Snapshots and backups are out of scope and the
  documentation says so rather than implying coverage.
- **It does not verify itself.** That is what [shadow mode](shadow.md) is for, and
  why it reports its own miss rate.
