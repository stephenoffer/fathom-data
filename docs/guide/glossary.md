# Glossary

Every term this library's output uses in a specific sense. The same entries are
available at the terminal, where you will actually want them:

```bash
fathom explain widening
fathom explain            # list every topic
```

---

### adapter

Anything that talks to another system. Three surfaces: **engines** (execution plans
and query logs), **catalogs** (table and partition metadata), **storage** (objects,
prefixes, etags). Adapters declare [capabilities](#capabilities) rather than
implement everything, so one that can do less still works — slower and coarser.

### alias

A declared equivalence between two identities that address the same data. An
external Hive table over an S3 prefix is the common case: nothing in either
reference reveals the connection. Until they are aliased they are two nodes, and a
plan seeded at one reaches nothing that reads the other.

### blast radius

How many datasets a change to one dataset could reach, transitively. The structural
upper bound — the plan gives the partition-level answer, which is usually far smaller.

### capabilities

What a given adapter can actually do: how it learns lineage, how it detects change,
what statistics it can compute for us, and how it can destroy data. Most surprises
about a plan are a declared limit showing through. `fathom adapters --verbose`.

### column lineage

Edges that record which source column feeds which target column, rather than only
that two datasets are related. Without it, drift attributes to a table rather than
to a column. Snowflake and Databricks provide it natively; SQL parsing provides it
where the SQL is parseable.

### completeness

The partitions that should exist and do not. The only check that can see a partition
which never arrived — it has no profile to drift and no rows to fail an expectation,
so nothing that reads data can find it. See [completeness](completeness.md).

### contract

What one team promised another about a dataset: the producer, the consumers, the
columns, and a staleness ceiling. A failing test says a column vanished; a breached
contract says who was promised it. See [contracts](contracts.md).

### coverage

How much of the graph is precise enough to plan on. Four fractions: datasets with a
spec, edges with a bounded mapping, edges with column detail, and provable field
mappings. The last is very close to the ceiling on how much of a rebuild any plan can
skip, so it is the number to move.

### dataset identity

A namespace locating the system and a name locating the dataset inside it, following
the OpenLineage convention — so `s3a://`, `s3://`, `/dbfs/mnt/...`, and
`catalog.schema.table` collapse to one node where they address the same bytes.

### drift

A column's distribution moving between profiles. On its own that is an alert; with
lineage it is a diagnosis, because the graph names what upstream could have moved it.
See [check](check.md).

### erasure

Locating a subject's data in every derived table, and stating what would actually
destroy it. Carries the mirror of the planner's invariant: it may under-delete and
refuse, but must never over-delete. See [erase](erase.md).

### evidence

How an edge was learned — `native`, `listener`, `query_log`, or `declared` — carried
on the edge so a surprising plan can be traced back to whatever claimed the
dependency.

### grain

How wide one time partition bucket is: hour, day, month, or year. A time partition
field must carry one, because a plan cannot tell a daily partition from a monthly one
without it. Conversions only ever go fine to coarse, and always round outward.

### invalidation

Working out which partitions went stale because something upstream changed. What the
`plan` verb does.

### key predicate

A constraint over one dataset's partition space: each field bound to a value or to
`ANY`. A predicate with every field `ANY` denotes the whole dataset, which is how the
planner represents "could not prove anything narrower".

### partition mapping

The rule on a graph edge answering: if this input partition is dirty, which output
partitions are dirty? Three forms — `TimeWindow`, `Passthrough`, `Unbounded`. They
compose along paths and join where paths reconverge. See [concepts](concepts.md).

### partition spec

How a dataset is divided: an ordered list of fields, each either a time field with a
grain or a value field. Declared in `fathom.yml`, because it cannot be reliably
inferred. A dataset with no spec can only be invalidated whole.

### profile

Distributions, ranges, and cardinalities per partition, read from file footers rather
than from the data. Cheap enough to run continuously, which is only true because the
graph says which partitions changed.

### pushdown

Statistics the source can compute for us instead of us reading the data — mergeable
sketches, quantiles, approximate distinct counts. An adapter with none still profiles;
it just costs a read.

### re-identification risk

Columns that identify nobody alone and everybody together — a birth date, a postcode.
Per-column labelling is structurally blind to it, because the property is not a
property of any column. Proves risk; never proves safety.

### seed

What you tell the planner changed at the source. Everything else in a plan is derived
from it. A dataset absent from the seeds is treated as unchanged.

### shadow mode

Running the planner beside your existing full rebuild and grading every decision it
made. Two numbers: savings, and missed. Missed must be zero. See [shadow](shadow.md).

### sink

The last hop out of the warehouse: a dashboard, a report, a regulatory filing.
Terminal nodes, so `fathom impact` can answer what has already been published from a
number that is about to be restated.

### selector

dbt's selection syntax resolved against the graph — `+model`, `model+`, `+model+`,
`2+model`, `@model`, `tag:pii`, `ns:snowflake`.

### soundness

The invariant: the planner may over-invalidate, and must never under-invalidate.
Precision is an optimization; soundness is not. Erasure carries the mirrored rule.
See [ADR 0002](../adr/0002-soundness-invariants.md).

### store

Where the two durable artifacts live — the graph and the profile history. SQLite by
default, behind a protocol so that is a default rather than a premise.

### unbounded

The mapping used wherever a relationship could not be proven. Any change to the input
rebuilds the whole output. The honest answer, not a failure: it costs compute and
never costs correctness.

### widening

Losing partition precision on the way through the graph, so a dataset is rebuilt
whole. Caused by an unbounded mapping, a missing spec, or a cycle — and contagious
downstream.

---

Longer treatments: [concepts](concepts.md) for the partition lattice,
[adapters](adapters.md) for the capability matrix, and the
[ADRs](../adr/) for why each rule errs in the direction it does.
