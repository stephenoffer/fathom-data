"""dbt manifest ingest.

dbt already knows the dependency graph, the warehouse each model lands in, and
often the partitioning — all of it sitting in `target/manifest.json` after any
`dbt compile`. Reading it is far more reliable than re-deriving the same facts by
parsing a directory of `.sql` files with unresolved `ref()` calls.

What dbt gives us and what it does not:

- **Dataset edges** — from `depends_on`, exactly and authoritatively.
- **Relation names** — resolved through the target's database and schema, so
  identities match what the warehouse actually holds.
- **Partition specs** — from `config.partition_by` on BigQuery and Spark. Snowflake
  has no partitioning concept, so those models need a declaration.
- **Column lineage and partition mappings** — never directly. But `compiled_code`
  is fully-resolved SQL, so the sqlglot extractor recovers both. dbt supplies the
  skeleton; parsing fills in the detail.

The escape hatch is `config.meta.fathom`, so a partition spec dbt cannot express
lives next to the model rather than in a separate config file that drifts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigError
from ..fs import FileSystem, filesystem_for, join
from ..grains import Grain
from ..graph import Edge, Graph
from ..ids import normalize_table
from ..ingest import IngestResult
from ..lineage import extract
from ..partitions import PartitionMapping
from ..types import UNPARTITIONED, DatasetId, PartitionField, PartitionSpec

__all__ = ["DbtManifest", "ingest_dbt", "load_manifest"]

# dbt adapter names map onto our identity systems one to one.
_ADAPTERS = {
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "databricks": "databricks",
    "spark": "databricks",
    "redshift": "redshift",
    "postgres": "postgres",
    "duckdb": "duckdb",
    "trino": "trino",
    "athena": "trino",
    "clickhouse": "clickhouse",
}

# BigQuery granularity values, and the Spark/dbt column types that imply a grain.
_GRANULARITY = {"hour": Grain.HOUR, "day": Grain.DAY, "month": Grain.MONTH, "year": Grain.YEAR}
_TYPE_GRAIN = {
    "date": Grain.DAY,
    "datetime": Grain.HOUR,
    "timestamp": Grain.HOUR,
    "timestamp_ntz": Grain.HOUR,
}


@dataclass
class DbtManifest:
    """The parts of a dbt manifest this project cares about."""

    system: str = "duckdb"
    project: str = ""
    dbt_version: str = ""
    relations: dict[str, DatasetId] = field(default_factory=dict)
    specs: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    parents: dict[str, list[str]] = field(default_factory=dict)
    compiled: dict[str, str] = field(default_factory=dict)
    materializations: dict[DatasetId, str] = field(default_factory=dict)

    @property
    def datasets(self) -> list[DatasetId]:
        return sorted(set(self.relations.values()), key=str)


def load_manifest(
    uri: str, *, fs: FileSystem | None = None, **storage_options: Any
) -> dict[str, Any]:
    """Read a manifest from a path, a URI, or a dbt project directory."""
    filesystem = fs or filesystem_for(uri, **storage_options)
    target = uri
    if filesystem.is_dir(uri):
        for candidate in (join(uri, "manifest.json"), join(uri, "target", "manifest.json")):
            if filesystem.exists(candidate):
                target = candidate
                break
        else:
            raise ConfigError(
                f"no manifest.json under {uri}; run `dbt compile` first, or point at "
                "the file directly"
            )
    try:
        blob = json.loads(filesystem.read_text(target))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(blob, dict) or "nodes" not in blob:
        raise ConfigError(f"{target} does not look like a dbt manifest (no `nodes` key)")
    return blob


def _relation(node: Mapping[str, Any], system: str) -> DatasetId | None:
    """The warehouse identity a node materializes to."""
    relation_name = node.get("relation_name")
    if relation_name:
        return normalize_table(str(relation_name), system=system)

    parts = [node.get("database"), node.get("schema"), node.get("alias") or node.get("name")]
    present = [str(p) for p in parts if p]
    if not present:
        return None
    return normalize_table(".".join(present), system=system)


def _spec_from_meta(meta: Mapping[str, Any]) -> PartitionSpec | None:
    """Read `config.meta.fathom.partition`, the escape hatch for what dbt cannot say."""
    declared = (meta.get("fathom") or {}).get("partition")
    if not declared:
        return None
    fields = []
    for entry in declared:
        if isinstance(entry, str):
            fields.append(PartitionField.value(entry))
            continue
        name = entry.get("field") or entry.get("name")
        if not name:
            continue
        grain = entry.get("grain") or entry.get("granularity")
        fields.append(
            PartitionField.time(str(name), Grain.parse(str(grain)))
            if grain
            else PartitionField.value(str(name))
        )
    return PartitionSpec.of(*fields) if fields else None


def _spec_from_config(node: Mapping[str, Any]) -> PartitionSpec:
    """Derive a partition spec from dbt config, preferring an explicit declaration."""
    config = node.get("config") or {}

    declared = _spec_from_meta(config.get("meta") or {})
    if declared is not None:
        return declared

    partition_by = config.get("partition_by")
    if not partition_by:
        return UNPARTITIONED

    # BigQuery: a dict with field, data_type, and granularity.
    if isinstance(partition_by, Mapping):
        name = partition_by.get("field")
        if not name:
            return UNPARTITIONED
        granularity = str(partition_by.get("granularity") or "").lower()
        data_type = str(partition_by.get("data_type") or "").lower()
        grain = _GRANULARITY.get(granularity) or _TYPE_GRAIN.get(data_type)
        return PartitionSpec.of(
            PartitionField.time(str(name), grain) if grain else PartitionField.value(str(name))
        )

    # Spark and Databricks: a list of column names, with types on the columns.
    names = [partition_by] if isinstance(partition_by, str) else list(partition_by)
    columns = node.get("columns") or {}
    fields = []
    for name in names:
        data_type = str((columns.get(name) or {}).get("data_type") or "").lower()
        grain = _TYPE_GRAIN.get(data_type)
        fields.append(
            PartitionField.time(str(name), grain) if grain else PartitionField.value(str(name))
        )
    return PartitionSpec.of(*fields) if fields else UNPARTITIONED


def parse_manifest(blob: Mapping[str, Any]) -> DbtManifest:
    """Extract the graph, relations, and specs from a raw manifest."""
    metadata = blob.get("metadata") or {}
    adapter = str(metadata.get("adapter_type") or "").lower()
    manifest = DbtManifest(
        system=_ADAPTERS.get(adapter, adapter or "duckdb"),
        project=str(metadata.get("project_name") or ""),
        dbt_version=str(metadata.get("dbt_version") or ""),
    )

    nodes: dict[str, Mapping[str, Any]] = {}
    nodes.update(blob.get("sources") or {})
    nodes.update(blob.get("nodes") or {})

    for unique_id, node in nodes.items():
        resource = str(node.get("resource_type") or "")
        # Tests, analyses, and seeds are not part of the data dependency graph a
        # rebuild plan cares about. Snapshots are, since they materialize tables.
        if resource not in {"model", "source", "snapshot", "seed"}:
            continue

        relation = _relation(node, manifest.system)
        if relation is None:
            continue

        manifest.relations[unique_id] = relation
        spec = _spec_from_config(node)
        if spec.fields:
            manifest.specs[relation] = spec
        manifest.materializations[relation] = str(
            (node.get("config") or {}).get("materialized") or resource
        )

        parents = [
            parent
            for parent in ((node.get("depends_on") or {}).get("nodes") or [])
            if isinstance(parent, str)
        ]
        if parents:
            manifest.parents[unique_id] = parents

        compiled = node.get("compiled_code") or node.get("compiled_sql")
        if compiled:
            manifest.compiled[unique_id] = str(compiled)

    return manifest


def ingest_dbt(
    manifest_or_uri: Mapping[str, Any] | str,
    *,
    specs: Mapping[DatasetId, PartitionSpec] | None = None,
    graph: Graph | None = None,
    parse_sql: bool = True,
    fs: FileSystem | None = None,
) -> IngestResult:
    """Build a graph from a dbt manifest.

    `parse_sql` runs the sqlglot extractor over each model's compiled SQL to recover
    column edges and partition mappings dbt does not record. Turning it off gives a
    dataset-level graph much faster, which is the right trade for a first look at a
    very large project.
    """
    blob = (
        load_manifest(manifest_or_uri, fs=fs)
        if isinstance(manifest_or_uri, str)
        else manifest_or_uri
    )
    manifest = parse_manifest(blob)

    result = IngestResult(graph=graph or Graph())
    known: dict[DatasetId, PartitionSpec] = {**manifest.specs, **dict(specs or {})}
    for dataset, spec in known.items():
        result.graph.add_dataset(dataset, spec)
    for dataset in manifest.datasets:
        result.graph.add_dataset(dataset, known.get(dataset, UNPARTITIONED))

    for unique_id, parents in manifest.parents.items():
        target = manifest.relations.get(unique_id)
        if target is None:
            continue
        result.statements += 1
        dst_spec = known.get(target, UNPARTITIONED)

        # Parse the compiled SQL once per model; dbt gives us the edges, sqlglot
        # gives us the column detail and the partition mapping.
        parsed_mappings: dict[DatasetId, PartitionMapping] = {}
        parsed_columns: dict[DatasetId, tuple[tuple[str, str], ...]] = {}
        sql = manifest.compiled.get(unique_id)
        if parse_sql and sql:
            statement = sql if _is_statement(sql) else f"CREATE TABLE {target.name} AS {sql}"
            for extraction in extract(
                statement, dialect=manifest.system, system=manifest.system, specs=known
            ):
                if extraction.target is None:
                    result.unparsed += 1
                    result.notes.extend(f"{target}: {n}" for n in extraction.notes)
                    continue
                parsed_mappings.update(extraction.mappings)
                parsed_columns.update(extraction.column_edges)

        for parent in parents:
            source = manifest.relations.get(parent)
            if source is None or source == target:
                continue
            src_spec = known.get(source, UNPARTITIONED)
            mapping = parsed_mappings.get(source)
            if mapping is None:
                mapping = (
                    PartitionMapping.rollup(src_spec, dst_spec)
                    if src_spec.fields and dst_spec.fields
                    else PartitionMapping.unknown(dst_spec)
                )
            result.graph.add_edge(
                Edge(
                    src=source,
                    dst=target,
                    mapping=mapping,
                    columns=parsed_columns.get(source, ()),
                    evidence=f"dbt:{unique_id}",
                )
            )

    incremental = [
        str(ds)
        for ds, mat in manifest.materializations.items()
        if mat == "incremental" and not known.get(ds, UNPARTITIONED).fields
    ]
    if incremental:
        result.notes.append(
            f"{len(incremental)} incremental model(s) have no partition spec, so every "
            "change to them widens to a full rebuild; declare one under "
            "config.meta.fathom.partition"
        )
    return result


def _is_statement(sql: str) -> bool:
    """True when the SQL already has a target, rather than being a bare SELECT."""
    head = sql.lstrip().lstrip("(").lstrip().upper()
    return head.startswith(("CREATE", "INSERT", "MERGE", "UPDATE", "DELETE"))
