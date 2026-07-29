# Adapters

Three surfaces, because "what depends on what" and "what changed" come from
different places and neither assumes SQL.

- **Engine** — execution plans and query logs
- **Catalog** — table and partition metadata
- **Storage** — objects, events, inventory manifests

Adapters declare capabilities rather than implement everything. One reporting
`LIST_DIFF` and `Pushdown.NONE` still works; it is slower and coarser, and the
planner degrades instead of failing.

## Capability matrix

| Adapter | Kind | Lineage | Column-level | Change detection | Erasure | Lag |
|---|---|---|---|---|---|---|
| `snowflake` | engine | `NATIVE` | yes | `WATERMARK` | rewrite | ~3h |
| `databricks` | engine | `NATIVE` | yes | `SNAPSHOT_DIFF` | delete vector | ~2h |
| `bigquery` | engine | `QUERY_LOG` | via parsing | `PARTITION_MTIME` | rewrite | — |
| `duckdb` | engine | `QUERY_LOG` | via parsing | — | rewrite | — |
| `delta` | catalog | declared | — | `SNAPSHOT_DIFF` | delete vector | — |
| `iceberg` | catalog | declared | — | `SNAPSHOT_DIFF` | delete vector | — |
| `storage` | storage | declared | — | `LIST_DIFF` | rewrite | — |

## Change detection, ranked

Cost per detected change, cheapest first. Prefer the highest strategy available.

| Strategy | Cost | Where |
|---|---|---|
| `SNAPSHOT_DIFF` | proportional to commits | Delta, Iceberg, Databricks |
| `EVENTS` | proportional to changes | S3 EventBridge, GCS Pub/Sub, Event Grid |
| `INVENTORY` | one manifest read | S3 Inventory, GCS Storage Insights |
| `PARTITION_MTIME` | one metadata query | BigQuery |
| `WATERMARK` | one indexed query | Snowflake, any SQL source |
| `LIST_DIFF` | proportional to object count | generic object storage |
| `PROFILE_DELTA` | proportional to data | last resort |

`LIST_DIFF` is fine under a few hundred thousand objects and ruinous above. A naive
LIST over a 100M-object bucket costs hours and real money, which is why the local
adapter uses it deliberately: if everything works on the weakest strategy, richer
adapters are strictly faster rather than differently shaped.

## Freshness lag

Some sources report metadata late. Snowflake's `ACCOUNT_USAGE` views lag up to
three hours; Databricks system tables about two.

Adapters state this in `capabilities.freshness_lag`, and resume tokens are held
back by it. Advancing a token to the newest row seen would permanently skip rows
that had not landed in the view when we read it.

```python
adapter.capabilities.freshness_lag    # timedelta(hours=3)
```

## Writing an adapter

Implement the protocol for your surface and register it:

```python
from fathom.adapters.base import ChangeSet, register
from fathom.core.types import Capabilities, ChangeSource, LineageSource

@register("mysystem")
@dataclass
class MySystemCatalog:
    name: str = "mysystem"
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.DECLARED,
        change=ChangeSource.WATERMARK,
        partition_aware=True,
    )

    def describe_partitioning(self, dataset) -> PartitionSpec: ...
    def changed(self, dataset, since) -> ChangeSet: ...
```

### The rule that matters

An adapter may report a partition that did not change. It may **never** omit one
that did.

Two corollaries people get wrong:

- If you cannot enumerate exhaustively — an expired event subscription, a truncated
  history, a token you no longer recognize — return `complete=False` **and** the
  partitions you can see. Returning an empty changeset reads as "nothing changed"
  and is the one unforgivable bug in a change detector.
- If a partition value fails to parse, bind it to `ANY` rather than skipping the
  row.

### Certify it

Subclass the conformance contract:

```python
from suite import StorageAdapterContract

class TestMySystem(StorageAdapterContract):
    def make_adapter(self, root, spec): ...
    def add_partition(self, root, *, dt, region, rows): ...
```

The suite asserts what a planner is entitled to rely on, not how you achieve it.
An adapter using snapshot diffs and one using LIST plus mtime both pass unchanged.

## Adapter options

Passed from `fathom.yml` under `adapters:`.

**`snowflake`** — `account`, `limit`, `runner`
**`databricks`** — `workspace`, `limit`, `storage_options`, `runner`
**`bigquery`** — `project`, `region`, `limit`, `runner`
**`duckdb`** — `database`
**`delta`** / **`iceberg`** — `storage_options`
**`storage`** — `storage_options`, `suffixes`

`runner` is never configured in the file. Inject it:

```python
project.register_runner("snowflake", DBAPIRunner(connection))
```

## Query runners

Warehouse adapters take a `QueryRunner` rather than a driver, for three reasons in
order of importance:

1. **Testability.** An adapter whose only test path is a live account has no tests.
   `RecordedRunner` replays captured rows, so query shaping and result parsing —
   where the bugs live — are covered offline.
2. **No hard dependencies.** Installing this should not drag in three cloud SDKs.
   `DBAPIRunner` wraps anything PEP 249, which all of them are.
3. **One place** to log, time, and cap queries.

```python
from fathom.adapters import DBAPIRunner, RecordedRunner

DBAPIRunner(snowflake.connector.connect(...))       # production
RecordedRunner({"access_history": [...]})            # tests
```
