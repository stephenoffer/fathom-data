# Cookbook

Task first, then the shortest thing that does it. If you know what you want but not
what it is called here, this is the page to scan.

Every snippet is self-contained apart from the setup block, which the Python recipes
assume:

```python
import fathom
from fathom import Graph, DatasetId, KeyPredicate, PartitionSpec, Store

store = Store(".fathom/fathom.db")
```

---

## Getting oriented

### See what a graph contains without reading it edge by edge

```python
graph = store.load_graph()
print(graph.describe())
```

```
412 datasets, 900 edges
380/412 datasets have a partition spec (the rest can only be rebuilt whole)
611/900 edges carry a provable mapping (289 unbounded, contributing nothing to a plan)
14 source dataset(s) nothing else feeds: duckdb/raw.events, …
```

### Find the number that predicts your savings

```python
print(fathom.metrics.coverage(graph).summary())
```

`field_ratio` is very close to the ceiling on how much of a rebuild any plan can
skip. Publish that one, and move it.

### Understand what one edge actually claims

```python
for edge in graph.in_edges(DatasetId("duckdb", "gold.monthly")):
    print(edge.explain())
```

```
duckdb/silver.events feeds duckdb/gold.monthly, learned from sql.
Columns: amount -> revenue
Partitions:
  dt: a dirty dt day taints the month containing it
  region: a dirty region taints the output rows with that same region
```

Or from the shell: `fathom lineage --explain`.

---

## Planning rebuilds

### Plan from a known change

```python
plan = graph.invalidate({
    DatasetId("duckdb", "raw.events"): [
        KeyPredicate.parse("dt=2026-03-14,region=eu", graph.spec(raw))
    ]
})
print(plan.summary())
```

### Iterate a plan in build order

```python
for dataset in plan:                       # dependencies before dependents
    for key in sorted(plan.partitions(dataset), key=str):
        rebuild(dataset, key)
```

### Find out why a plan is larger than expected

```python
print(plan.explain(DatasetId("duckdb", "gold.monthly")))
```

Or `fathom plan --detect --explain gold.monthly`. Then `fathom explain widening`.

### Cost a plan before running it

```python
from fathom.cost import CostModel, estimate_plan, savings

model = CostModel(price_per_partition=0.02, price_per_tb_scanned=5.0)
print(estimate_plan(plan, model).summary())
print(savings(graph, plan, model).summary())   # against rebuilding everything
```

### Turn a plan into something your orchestrator runs

```bash
fathom dag --flavor airflow --dirty 'raw.events@dt=2026-03-14' --out dags/rebuild.py
fathom dag --flavor shell   --dirty 'raw.events@dt=2026-03-14'
```

Nothing imports Airflow. The output is a file you commit and read.

### Batch a plan into waves

```python
waves = fathom.schedule.schedule(graph, plan)
```

Contiguous partitions are collapsed into ranges, so a 30-day backfill is one task
per dataset rather than thirty.

---

## Describing your data

### Declare a partition spec in one line

```python
spec = PartitionSpec.parse("dt:day, region")
```

The same thing `fathom.yml` spells out as a list of mappings. For tests and
notebooks, where the ceremony outweighs the declaration.

### Say that a daily table rolls up into a monthly one

```python
from fathom.core.partitions import PartitionMapping

daily, monthly = PartitionSpec.parse("dt:day"), PartitionSpec.parse("dt:month")
mapping = PartitionMapping.rollup(daily, monthly)
```

### Say that a table is a 7-day trailing aggregate

```python
from fathom.core.partitions import PartitionMapping, TimeWindow

mapping = PartitionMapping.of(dt=TimeWindow.trailing("dt", 7, "day"))
```

State the length, not the offsets. `(0, 6)` written by hand is where the off-by-one
lives, and an off-by-one here under-invalidates silently.

### Check that a mapping says what you meant

```python
print(mapping.explain())
```

```
dt: a dirty dt taints that day and the 6 days after it
```

This is the one thing in the graph nobody can verify by reading the SQL that
produced it, so read it out loud in review.

### Build a small graph by hand, for a test

```python
graph = Graph()
graph.add_dataset("duckdb/raw.events", "dt:day, region")
graph.add_dataset("duckdb/gold.monthly", "dt:month, region")
graph.connect(
    "duckdb/raw.events", "duckdb/gold.monthly",
    evidence="sql",
    mapping=PartitionMapping.rollup(
        PartitionSpec.parse("dt:day, region"),
        PartitionSpec.parse("dt:month, region"),
    ),
)
```

### Unify two names for the same data

```python
registry = fathom.AliasRegistry()
registry.alias(DatasetId("hive", "raw.events"), DatasetId("s3://lake", "raw/events"))
```

Until this exists they are two nodes, and a plan seeded at one reaches nothing that
reads the other.

---

## Watching the data

### Profile without scanning

```python
profile = fathom.profile_parquet(
    ["s3://lake/raw/events/dt=2026-03-14/part-0.parquet"],
    dataset=DatasetId("s3://lake", "raw/events"),
    partition=KeyPredicate.parse("dt=2026-03-14"),
)
store.save_profile(profile)
```

Footers only. No data pages are read.

### Detect drift and attribute it upstream

```python
findings = fathom.drift(previous, current)   # both are Profiles
for finding in findings:
    paths = graph.upstream_columns(fathom.ColumnRef(dataset, finding.column))
    print(finding.detail, "<-", " <- ".join(str(step) for step in paths[0]))
```

### Generate an expectation suite from what you have observed

```python
suite = fathom.quality.learn(profile)          # bounds from what was observed
findings = fathom.quality.check(suite, later)  # against a later profile
```

Rather than writing bounds by hand and discovering they were wrong during an
incident.

### Bucket a baseline by day of week

```python
baseline = fathom.seasonal.learn_seasonal(observations, cycle=fathom.seasonal.Cycle.DAY_OF_WEEK)
print(fathom.seasonal.strength(observations, cycle=fathom.seasonal.Cycle.DAY_OF_WEEK))
```

Check the strength first. Below about 20% the flat bound is the better tool.

### Find partitions that never arrived

```python
report = fathom.completeness.report(
    dataset, spec, present,
    start=datetime(2026, 3, 1), end=datetime(2026, 3, 31),
)
print(report.summary())
```

---

## Governance

### Find out which models still hold a subject

```python
for exposure in fathom.ai.unlearning.exposures(graph, DatasetId("duckdb", "raw.events")):
    print(exposure)
```

Deleting rows does not remove a subject from weights, and this says so rather than
reporting completion.

### Check whether an eval set leaked into training

```python
report = fathom.ai.evals.contamination(graph, model=model_id, eval_set=eval_id)
print(report.summary())    # clean, suspect, or contaminated — never rounded up
```

Contamination is a reachability property: a shared ancestor is enough.

### Stop a pull request that narrows a mapping

```python
diff = fathom.diff.diff_graphs(before, after)
if not diff.is_safe:
    raise SystemExit(diff.summary())
```

Widening costs compute and is always correct. Narrowing and removal are the two ways
a graph edit serves stale data, and the two a merge gate should stop.

### Name everything already published from a dataset

```bash
fathom impact --dataset gold.monthly --reason "restated fx rates"
```

Exits non-zero when a regulatory filing is downstream, because that is a different
conversation from a stale dashboard.

---

## Adoption

### Prove the planner before trusting it

```python
report = fathom.shadow.run(engine, plan, [silver, gold], store=store)
print(report.summary())
```

```
shadow: SOUND across 2 dataset(s), 78% of partitions skipped
```

Runs beside your existing full rebuild, so there is no risk. `missed` must be zero,
and `fathom shadow` exits non-zero the moment it is not. Accumulate weeks.

### Gate CI on the checks that can fail

```bash
fathom check --json      # drift
fathom label             # policy violations
fathom contracts         # broken promises
fathom shadow            # a missed partition is a soundness failure
```

All exit non-zero on failure. Use `--json` where it exists rather than parsing a
summary written for people.

---

## See also

- [Getting started](getting-started.md) — the same ground, in order, once
- [Python API tour](python-api.md) — how the package is laid out
- [FAQ](faq.md) — why something behaved the way it did
- [`examples/`](../../examples/) — runnable versions of most of the above
