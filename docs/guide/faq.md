# FAQ

The questions people actually ask in the first week, with short answers. Each links
to the longer version.

---

## Getting started

### Will this touch my data?

No. `plan` prints what it would rebuild, `erase` prints what it would destroy, and
neither has an `--execute` flag. Applying either needs an engine binding you supply
deliberately from a pipeline. Profiling reads Parquet footers, not data pages.

The one thing that writes is the store — a SQLite file holding the graph, the
profiles, and the history. It goes in `.fathom/` and should not be committed.

### What is the smallest useful setup?

A `fathom.yml` with two datasets and a partition spec on each, and one lineage
source. That is enough for `fathom plan` to say something true.

```bash
fathom init      # writes a starter fathom.yml
fathom ingest    # build the graph from the lineage block
fathom plan --dirty 'raw.events@dt=2026-03-14'
```

See [getting started](getting-started.md).

### Do I have to use the CLI?

No. Everything the CLI does goes through the `Project` facade, and the library
underneath is the product for anyone embedding this. See the
[Python API tour](python-api.md).

### Do I need a warehouse to try it?

No. The [`01_local_lakehouse.py`](../../examples/01_local_lakehouse.py) example runs
the whole loop on local Parquet files with no external system at all.

---

## Plans

### Why is my plan bigger than I expected?

Almost always one of three things, in order of likelihood:

1. **An unbounded mapping** on the path — the edge exists but nothing proved which
   partitions it maps to. `fathom doctor` names them.
2. **A missing partition spec** — a dataset with no spec has no partitions to name,
   so it can only be invalidated whole.
3. **A cycle** — a self-referencing incremental model, which the planner widens
   rather than loops on.

`fathom plan ... --explain THE_DATASET` says which. Then `fathom explain widening`.

### Why did one dirty day produce three dirty months?

Grain conversion rounds outward, deliberately. A window measured in days, re-expressed
in months, has to allow for the input day sitting anywhere inside its own month and
for the window straddling a boundary. The result covers every bucket the finer window
could touch.

That is the soundness invariant working, not a defect. `fathom explain grain`.

### Why is my plan empty?

Three possibilities:

- Nothing was seeded. `--dirty` and `--detect` are the input, not a filter; with
  neither, there is nothing to propagate.
- `--detect` on a second run. Detection advances a resume token per source, so the
  second run legitimately reports nothing new.
- The seeded dataset is not in the graph. `fathom plan` warns about this on stderr —
  it is usually a typo in the name.

### Can the planner be wrong?

It can be *imprecise* — it may plan more than strictly necessary, which costs
compute. It must never plan less, which would serve stale data.

Do not take that on trust. [Shadow mode](shadow.md) runs it beside your existing full
rebuild and reports a miss count that must be zero. Accumulate weeks of it before
anything writes.

### Does it execute the rebuild?

No, and that is deliberate. It generates the DAG your orchestrator already reads:

```bash
fathom dag --flavor airflow --dirty 'raw.events@dt=2026-03-14'
```

Nothing imports Airflow, Dagster, or Prefect — the output is a file you commit.

---

## The graph

### Where does the graph come from?

Whatever the `lineage` block in `fathom.yml` points at: parsed SQL, a dbt manifest,
OpenLineage events, a warehouse's own lineage tables, or your own declarations. Each
edge records which of those claimed it, so a surprising plan can be traced back.

### Why is a table missing from the graph?

Either nothing claimed an edge to it, or it is in the graph under a second identity.
The second is common: a Spark job writing `s3a://lake/raw/events` and a Trino query
reading `hive.raw.events` are the same bytes and two names. Declare the alias and
they collapse to one node.

### Why does my SQL produce no edges?

A dialect the parser cannot read produces zero edges rather than an error. Check the
`dialect` key. A `MERGE` statement, a dynamic query, or an opaque UDF also yields
nothing provable — those widen to unbounded rather than guess.

### What does `[declared]` mean at the end of a lineage line?

The evidence: how that edge was learned. `native` is the platform's own lineage
table, `query_log` is parsed SQL, `declared` is you. A declared edge is exactly as
right as what you wrote and does not notice when the pipeline changes underneath it.
`fathom explain evidence`.

---

## Partitions

### Why do I have to declare partition specs by hand?

Because they cannot be reliably inferred. Snowflake has no partitions to read at all.
Delta records the partition column names but not the grain — nothing in the metadata
distinguishes a `dt` holding days from one holding months, and guessing wrong makes
every mapping composed across that dataset wrong.

### What grain names are accepted?

`hour`, `day`, `month`, `year` — plus the adjective (`daily`), the plural (`days`),
and the abbreviation (`d`). They all resolve to the same thing.

### My dataset is not partitioned. Is it still useful?

Yes, but it can only ever be invalidated whole, and everything downstream of it
inherits that imprecision. It still contributes structure — blast radius, drift
attribution, policy propagation, and erasure all work at dataset level.

---

## Checks and profiles

### `fathom check` says nothing. Is my data fine?

Not necessarily. A first check has no prior profile to compare against, so it has no
opinion. Run `fathom profile` to establish a baseline, and check after the next run.

Separately, `check` reads data that arrived. A partition that never arrived has no
profile to drift and no rows to fail an expectation — that is what
[completeness](completeness.md) is for.

### Why does my weekly-seasonal table alert every weekend?

Because one flat band across a weekly cycle is wide enough to admit Tuesday's floor
and Sunday's ceiling. Use `fathom seasonal`, which learns a band per bucket — and
check the reported strength first. Below about 20%, the flat bound is the better tool.

### Does profiling cost warehouse credits?

Reading Parquet footers costs a metadata read, not a scan. Where a warehouse can
compute statistics for us it does that instead — see the
[capability matrix](adapters.md). The reason continuous profiling is affordable at
all is that the graph says which partitions changed.

---

## Governance

### Does `erase` actually delete anything?

Not from the CLI. It locates the subject's data everywhere it flowed and reports what
would discharge the obligation. Erasure may under-delete and refuse; it must never
over-delete, so it dry-runs and refuses outright on WORM storage rather than
reporting a success it cannot deliver.

### Why does writing an erasure proof need a salt?

Identifiers are low-entropy, so an unsalted digest identifies the subject about as
well as the raw value does — and the proof is the artifact handed to people who must
not learn who they were. Set `FATHOM_SALT` to a per-organization secret.

### Can it remove someone from a trained model?

No, and it says so rather than reporting `complete: true`. It names every model that
retains them and what would actually discharge it — retraining, or crypto-shredding
data that was encrypted per subject. See [AI assets](ai.md).

---

## Operations

### Should I commit the store?

No. It is a cache of things `fathom ingest` and `fathom profile` rebuild, and it will
conflict. Commit `fathom.yml`.

### Where does the store live?

`.fathom/fathom.db` by default. Change it with the `store:` key, or override per
invocation with `--store` or `FATHOM_STORE`.

### How do I run this in CI?

The commands that can fail exit non-zero: `check` on an error-severity finding,
`label` on a policy violation, `contracts` on a breach, `shadow` on a missed
partition, `erase` on an incomplete plan, `impact` on regulatory exposure.

Use `--json` where it exists rather than parsing the human summary — the summaries
are written for people and their wording will change.

### It needs a dependency I do not have.

Optional extras cover the systems that need a client library:

```bash
pip install 'fathom-data[iceberg]'   # Iceberg manifests are Avro
pip install 'fathom-data[duckdb]'
pip install 'fathom-data[cloud]'     # S3, GCS, Azure
```

The error naming the missing package also names the extra that provides it.

---

## Still stuck

- `fathom explain` lists every concept the output assumes.
- [Troubleshooting](troubleshooting.md) covers the warnings and what to do about them.
- [Error messages](errors.md) is an index of what this library raises.
- [problems.md](../problems.md) is an honest status per problem, including the gaps.
