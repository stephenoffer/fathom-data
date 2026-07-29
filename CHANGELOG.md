# Changelog

Notable changes, newest first. This project follows [semantic
versioning](https://semver.org/), with one addition that matters more than the
version number:

**Any change to what the planner invalidates is called out explicitly**, with the
direction it errs in. A change that makes plans *wider* costs compute and is safe to
take. A change that makes plans *narrower* can serve stale data if it is wrong, and is
flagged here as such so you can decide whether to re-run [shadow
mode](docs/guide/shadow.md) before adopting it.

## Unreleased

### Added

- `fathom explain <topic>` — every term the output uses in a specific sense, with what
  to do about it, at the terminal rather than in a browser. `fathom explain` with no
  argument lists them.
- `fathom lineage --explain` says what each edge claims in sentences, including the
  partition mapping, which is the one part of the graph a reviewer cannot check
  against the SQL.
- `fathom plan --explain DATASET` answers "why is this in the plan, and why so much of
  it" for one dataset.
- `fathom adapters --verbose` prints each capability with what it means for your
  plans, rather than the bare constant.
- `Graph.connect()` builds an edge and registers both endpoints in one call, and
  `Graph.describe()` orients a reader in an unfamiliar graph.
- String constructors where a type was previously required: `DatasetId.parse`,
  `ColumnRef.parse`, `PartitionField.parse`, `PartitionSpec.parse("dt:day, region")`,
  and `KeyPredicate.parse("dt=2026-03-14,region=eu", spec)`. `Graph.add_dataset` and
  `Graph.connect` accept them directly.
- `TimeWindow.identity`, `.rollup`, and `.trailing` — named constructors for the three
  shapes almost every real edge has. `.trailing` takes a length rather than offsets,
  because an off-by-one written as `(0, 6)` under-invalidates silently.
- `.explain()` on every partition mapping, returning the claim as a sentence.
- `InvalidationPlan` is iterable in build order, sized, and truthy when it has work;
  plus `total_partitions`, `why()`, and `explain()`.
- `Capabilities.summary()` and `.explain()`, and a description on every member of the
  four capability enums.
- New documentation: [cookbook](docs/guide/cookbook.md),
  [Python API tour](docs/guide/python-api.md), [glossary](docs/guide/glossary.md),
  [FAQ](docs/guide/faq.md), and an [index of error messages](docs/guide/errors.md).
- `CONTRIBUTING.md`, and this file.

### Changed

- `fathom --help` lists commands by what they are for rather than alphabetically, and
  states up front that nothing writes to your data.
- A mistyped command or topic suggests the nearest real one.
- `Grain.parse` accepts the adjective, the plural, and the abbreviation (`daily`,
  `days`, `d`) alongside the canonical name, and is idempotent over `Grain`.
- Error messages across `core`, the adapter registry, and the CLI now name the
  offending value and the next action. Where a name was rejected, the message lists
  what is accepted and suggests the closest match.
- `fathom.yml` parsing, `--dirty` parsing, and `KeyPredicate.parse` are now one
  implementation, so the CLI syntax and the Python one cannot drift.

### Fixed

- `KeyPredicate.of` carried `PartitionSpec.of`'s docstring, which described rejecting
  duplicate field names — something it does not do.

### Nothing changed about what gets invalidated

Every change above is to names, messages, and documentation. No mapping composes
differently, no plan is narrower, and no existing call behaves differently. If you
have shadow-mode evidence, it still stands.

## 0.1.0

First beta. All four verbs — `plan`, `check`, `label`, `erase` — working end to end
against Snowflake, Databricks, BigQuery, DuckDB, Delta, Iceberg, and object storage.

- Partition-scoped invalidation over a column-level dependency graph, with the
  soundness invariant property-tested across generated grain and window combinations.
- Lineage from parsed SQL, dbt manifests, OpenLineage events, and native warehouse
  lineage tables.
- Footer-only profiling, drift detection with upstream attribution, learned
  expectation suites, and seasonal baselines.
- Label inference and propagation, sink policy enforcement, licences, consent, and
  purpose limitation.
- Erasure planning with proofs, dry-run by default, refusing on WORM storage.
- The same four verbs over AI assets: models, feature views, vector indexes, prompts,
  eval sets, and agent runs.
- Shadow mode, which is how you decide whether to trust any of it.
