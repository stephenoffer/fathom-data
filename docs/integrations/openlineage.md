# OpenLineage — Spark, Flink, Trino, Airflow, Dagster

The highest-leverage integration here, because the emitters already exist. Rather
than writing a listener per engine, consume the format they all speak.

It is also why identities follow the
[OpenLineage naming convention](../adr/0003-dataset-identity.md): a `RunEvent` maps
onto the graph with no translation layer.

## Setup

Point at a file, a directory, or an object-storage prefix:

```yaml
lineage:
  - type: openlineage
    events: s3://lineage/events/
```

```bash
fathom ingest
```

Three file formats are accepted, because producers disagree: a JSON array, a single
object, or newline-delimited JSON. A truncated final line — from reading a log
mid-write — is skipped rather than fatal.

## Emitting events

### Spark

```python
spark = (SparkSession.builder
    .config("spark.jars.packages", "io.openlineage:openlineage-spark_2.12:1.+")
    .config("spark.extraListeners", "io.openlineage.spark.agent.OpenLineageSparkListener")
    .config("spark.openlineage.transport.type", "file")
    .config("spark.openlineage.transport.location", "/lineage/events.jsonl")
    .getOrCreate())
```

Spark's integration emits the `columnLineage` facet, so you get column edges. It
covers SQL and the DataFrame API; arbitrary RDD work is opaque.

### Airflow

Airflow 2.7+ ships the provider:

```bash
pip install apache-airflow-providers-openlineage
```

```ini
[openlineage]
transport = {"type": "file", "log_file_path": "/lineage/events.jsonl"}
```

### Flink, Trino, Dagster

Flink and Trino have OpenLineage integrations; Dagster emits events natively. All
three produce dataset-level lineage. Column lineage varies by connector.

## What events give you

| | |
|---|---|
| **Dataset edges** | Always, from `inputs` × `outputs` |
| **Column edges** | From the `columnLineage` facet, where the producer emits it |
| **Aliases** | From the `symlinks` facet |
| **Partition mappings** | Never — OpenLineage has no partition facet |

Specs come from your config:

```yaml
datasets:
  - name: s3://lake/silver/events
    partition: [{field: dt, grain: day}]
  - name: s3://lake/gold/monthly
    partition: [{field: dt, grain: month}]
```

With both declared, an edge between them gets a real `day->month` mapping. Without
them, it stays unbounded and every change rebuilds the whole target.

## Symlinks solve the alias problem

The `symlinks` facet names the same bytes under another identity — exactly the
"external Hive table pointing at an S3 prefix" case that
[ADR 3](../adr/0003-dataset-identity.md) otherwise leaves to manual declaration:

```json
"symlinks": {"identifiers": [{"namespace": "hive://cluster", "name": "gold.monthly"}]}
```

```python
from fathom.core.ids import AliasRegistry

registry = AliasRegistry()
ingest_openlineage(events, aliases=registry)
registry.resolve(DatasetId("hive://cluster", "gold.monthly"))
# -> DatasetId("s3://lake", "gold/monthly")
```

A producer that emits it saves you the declaration.

## One run, several events

Runs emit START, RUNNING, and COMPLETE for the same work. Ingesting all three would
triple the edges and, worse, record lineage for runs that later failed.

Events are collapsed per `runId`, keeping the most authoritative one — COMPLETE
first, then RUNNING, then START. Failed runs (`FAIL`, `ABORT`) are skipped:

```
  ! skipped 3 failed run(s)
```

A job that died halfway wrote something, but not a dependency anyone should plan
against. Override deliberately if you disagree:

```python
ingest_openlineage(events, include_failed=True)
```

## From Python

```python
from fathom.ingest import ingest_openlineage, load_events

events = load_events("s3://lineage/events/")
result = ingest_openlineage(events, specs=specs, aliases=registry)
print(result.summary())
```

## Streaming events in

For a live pipeline, land events in object storage and ingest on a schedule. The
loader reads a whole prefix and dedupes by `runId`, so re-reading overlapping
windows is harmless:

```bash
*/15 * * * *  cd /srv/analytics && fathom ingest
```

There is no HTTP receiver in this project. If you run the OpenLineage proxy or
Marquez, point `events:` at wherever it archives, or export from its API.

## Combining sources

Sources accumulate rather than compete. A dbt manifest and an OpenLineage stream
describing the same pipeline reinforce each other, because edges are keyed by
evidence:

```yaml
lineage:
  - type: dbt
    manifest: target/manifest.json
  - type: openlineage
    events: s3://lineage/events/
```

Use dbt for the modeled warehouse and OpenLineage for the Spark jobs feeding it.

## Known limitations

- No partition facet exists, so mappings always come from your declarations.
- Column lineage depends entirely on the producer. Spark emits it; many connectors
  do not, and the ingest notes say which.
- Namespaces are taken as-is. A producer emitting inconsistent namespaces for the
  same storage fragments the graph, and `fathom lineage` is where you will notice.
