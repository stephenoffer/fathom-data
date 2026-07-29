# Python API tour

The verbs are the product; the library is what you build on. This is a map of the
package for someone importing it rather than running `fathom`.

If you want recipes instead of a map, go to the [cookbook](cookbook.md).

---

## The shape of it

The tree is deep and the import surface is flat. Packages follow the lifecycle, so a
new capability has one obvious home, and the names you actually type are re-exported
at the top:

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
store/      persistence, behind a protocol so SQLite is a default and not a premise
report/     rendering out: Mermaid, DOT, Markdown, OpenLineage, DataHub, compliance
cli/        the command line, project config, and `fathom.yml`
```

The layering is enforced by a test, not by convention: `core` may know nothing,
`graph` may know `core`, `ai` may know all of them. See
[ADR 0007](../adr/0007-layered-packages.md).

---

## The five types you will actually hold

Everything else is a function over these.

### `DatasetId` — what a dataset is called

```python
from fathom import DatasetId, normalize

DatasetId("duckdb", "raw.events")
DatasetId.parse("s3://lake/raw/events")     # read back what str() printed
normalize("s3a://lake/raw/events")          # resolve a spelling from the wild
normalize("orders", system="snowflake")     # a bare table name
```

Use `normalize` on anything a person or another system typed — it folds identifier
case per platform and unifies `s3a://`, DBFS mounts, and the rest. Use `parse` to
read back a string this library printed.

### `PartitionSpec` — how a dataset is divided

```python
from fathom import PartitionSpec, PartitionField, Grain

PartitionSpec.parse("dt:day, region")       # the compact form
PartitionSpec.of(
    PartitionField.time("dt", Grain.DAY),
    PartitionField.value("region"),
)                                            # the explicit form
```

A dataset with no spec can only be invalidated whole. This is the single most
valuable thing you declare.

### `KeyPredicate` — one partition, or a set of them

```python
from fathom import KeyPredicate, ANY

KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
KeyPredicate.parse("dt=2026-03-14,region=eu", spec)   # the CLI's --dirty syntax
KeyPredicate.of(dt=ANY, region="eu")                  # every day, in the EU
```

Each field is bound to a value or to `ANY`. All-`ANY` denotes the whole dataset,
which is how the planner represents "could not prove anything narrower".

### `PartitionMapping` — what an edge claims

```python
from fathom import TimeWindow, Passthrough, UNBOUNDED
from fathom.core.partitions import PartitionMapping

PartitionMapping.of(
    dt=TimeWindow.rollup("dt", "day", "month"),
    region=Passthrough("region"),
)
```

Three named constructors cover almost every real edge:
`TimeWindow.identity`, `.rollup`, and `.trailing`. Read one back with `.explain()` —
it is the only part of the graph a reviewer cannot check against the SQL.

### `Graph` — the datasets and the edges between them

```python
from fathom import Graph

graph = Graph()
graph.add_dataset("duckdb/raw.events", "dt:day, region")
graph.connect("duckdb/raw.events", "duckdb/gold.monthly",
              evidence="sql", mapping=mapping)
print(graph.describe())
```

Identities and specs may be strings anywhere they are accepted. Traversal does not
live on the class — see `fathom.query`.

---

## The four verbs, in Python

```python
plan  = graph.invalidate(seeds)                  # what must be rebuilt
found = fathom.drift(before, after)              # what moved, and why
labels = fathom.infer(profile); fathom.propagate(graph, labels)
erasure = fathom.plan_erasure(graph, request)    # where a subject lives
```

Each returns an object with `.summary()` for people and structured fields for code.

---

## Reading a plan

```python
plan = graph.invalidate(seeds)

bool(plan)                       # is there work
len(plan)                        # affected datasets
plan.total_partitions            # the size of the rebuild
for dataset in plan: ...         # build order, dependencies first
plan.partitions(dataset)         # the dirty keys for one
plan.why(dataset)                # the reasons it is in the plan
plan.explain(dataset)            # all of the above, as text
plan.widened, plan.cyclic        # where precision was lost
```

`widened` and `cyclic` are the first two things to look at when a plan is bigger
than expected.

---

## Where each question lives

| Question | Module |
|---|---|
| what depends on what, how far, which paths | `fathom.query` |
| select a subgraph the way dbt does | `fathom.selectors` |
| what did this pull request do to the graph | `fathom.diff` |
| how much of the graph is precise enough to plan on | `fathom.metrics` |
| who narrowed this edge, and when | `fathom.history` |
| what does this plan cost | `fathom.cost` |
| how does it turn into runnable work | `fathom.schedule` |
| what does the data look like | `fathom.profile`, `fathom.quality` |
| the same, bucketed by a cycle | `fathom.seasonal` |
| which partitions never arrived | `fathom.completeness` |
| is a table fresh, transitively | `fathom.freshness` |
| who actually reads it | `fathom.usage` |
| what has it cost since it existed | `fathom.lifetime` |
| which dashboards and filings would a restatement touch | `fathom.sinks` |
| what does this column mean, and what policy applies | `fathom.policy` |
| where does a subject's data live | `fathom.erasure` |
| what did we promise whom | `fathom.contracts` |
| what do these columns jointly reveal | `fathom.reidentification` |
| models, training sets, vectors, prompts, evals | `fathom.ai` |
| render it out to people or other tools | `fathom.render`, `fathom.emit` |
| prove the planner before trusting it | `fathom.shadow` |

---

## Persistence

```python
from fathom import Store

store = Store(".fathom/fathom.db")
store.save_graph(graph)
graph = store.load_graph()
store.save_profile(profile)
```

SQLite is a default rather than a premise — persistence sits behind a protocol, so a
team needing the artifacts shared across CI runners can back it with something else
without touching anything above it.

---

## The `Project` facade

Everything the CLI does goes through one object, so the CLI and the Python API cannot
drift apart:

```python
from fathom.cli.project import Project
from fathom.cli.config import load_config

project = Project(config=load_config(), store=Store(".fathom/fathom.db"))
project.ingest()
plan = project.plan(detect=True)
```

Live connections are injected, never configured: `fathom.yml` declares *that* a
dataset lives in Snowflake and how it is partitioned, and you supply the connection
with `register_runner`. That keeps credentials out of a file people want to commit.

---

## Errors

Everything raised deliberately derives from `FathomError`:

```python
from fathom import FathomError, ConfigError, StorageAccessError, AdapterUnavailable
```

`StorageAccessError` deliberately does not collapse into `FileNotFoundError` —
"could not tell" and "not there" are different facts, and reading the first as the
second is how a missing credential becomes a silently empty graph. See
[error messages](errors.md).

---

## See also

- [Cookbook](cookbook.md) — task-first recipes
- [Concepts](concepts.md) — the partition lattice and the soundness invariant
- [`examples/`](../../examples/) — runnable, and executed by the test suite
