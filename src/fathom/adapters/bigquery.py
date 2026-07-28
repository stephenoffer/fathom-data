"""BigQuery adapter.

The mirror image of Snowflake. BigQuery has no native column lineage in
`INFORMATION_SCHEMA` — that lives in Dataplex, behind a separate API — so lineage
comes from parsing job SQL. But its *partition* metadata is the best of any
warehouse here: `INFORMATION_SCHEMA.PARTITIONS` reports a `last_modified_time` per
partition, which is exactly the question change detection asks, answered directly
and cheaply.

So the trade runs opposite to Snowflake: lineage costs a SQL parse, and change
detection is free and exact.

Partition grain is read off the `partition_id` format, because that is where
BigQuery actually encodes it: `2026031409` is hourly, `20260314` daily, `202603`
monthly, `2026` yearly. `__NULL__` and `__UNPARTITIONED__` are real partitions with
reserved names, not errors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..errors import ConfigError
from ..grains import Grain
from ..ids import normalize_table
from ..types import (
    ANY,
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    DatasetId,
    ErasureMode,
    KeyPredicate,
    LineageSource,
    PartitionField,
    PartitionSpec,
    Pushdown,
)
from .base import ChangeSet, LineageEvent, QueryEvent, Token, register
from .predicates import literal, render_predicate
from .sql_runner import QueryRunner, quote_identifier

__all__ = ["BigQueryAdapter"]

# Reserved partition ids. Both are real partitions, and treating them as parse
# failures would silently drop rows from every rebuild.
NULL_PARTITION = "__NULL__"
UNPARTITIONED_ID = "__UNPARTITIONED__"

_PARTITIONS = """
SELECT table_name, partition_id, last_modified_time, total_rows
FROM `{project}.{dataset}.INFORMATION_SCHEMA.PARTITIONS`
WHERE table_name = @table
ORDER BY last_modified_time
"""

_PARTITION_COLUMN = """
SELECT column_name, data_type
FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = @table AND is_partitioning_column = 'YES'
"""

_JOBS = """
SELECT job_id, query, creation_time, destination_table, referenced_tables
FROM `{region}`.INFORMATION_SCHEMA.JOBS
WHERE creation_time > @since
  AND job_type = 'QUERY' AND state = 'DONE' AND error_result IS NULL
  AND statement_type IN ('CREATE_TABLE_AS_SELECT', 'INSERT', 'MERGE', 'UPDATE', 'DELETE')
ORDER BY creation_time
LIMIT @limit
"""

# partition_id length -> the grain it encodes.
_ID_GRAIN = {4: Grain.YEAR, 6: Grain.MONTH, 8: Grain.DAY, 10: Grain.HOUR}
_ID_FORMAT = {Grain.YEAR: "%Y", Grain.MONTH: "%Y%m", Grain.DAY: "%Y%m%d", Grain.HOUR: "%Y%m%d%H"}


def _row(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        for key in (name, name.lower(), name.upper()):
            if key in row:
                return row[key]
    return None


def grain_of(partition_id: str) -> Grain | None:
    """The grain a partition id encodes, or None for reserved and range ids."""
    if not partition_id or partition_id in {NULL_PARTITION, UNPARTITIONED_ID}:
        return None
    if not partition_id.isdigit():
        return None
    return _ID_GRAIN.get(len(partition_id))


def parse_partition_id(partition_id: str) -> datetime | None:
    grain = grain_of(partition_id)
    if grain is None:
        return None
    try:
        return datetime.strptime(partition_id, _ID_FORMAT[grain])
    except ValueError:
        return None


@register("bigquery")
@dataclass
class BigQueryAdapter:
    """Lineage from job SQL; change detection from per-partition modification times."""

    runner: QueryRunner | None = None
    project: str | None = None
    region: str = "region-us"
    name: str = "bigquery"
    dialect: str = "bigquery"
    specs: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    models: dict[DatasetId, str] = field(default_factory=dict)
    limit: int = 100_000
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.QUERY_LOG,
        change=ChangeSource.PARTITION_MTIME,
        pushdown=Pushdown.APPROX_DISTINCT,
        erasure=ErasureMode.REWRITE,
        column_lineage=True,  # via SQL parsing, not natively
        partition_aware=True,
    )

    # -- configuration ---------------------------------------------------------

    def declare(self, dataset: DatasetId, spec: PartitionSpec) -> None:
        self.specs[dataset] = spec

    def register_model(
        self, dataset: DatasetId, sql: str, spec: PartitionSpec | None = None
    ) -> None:
        self.models[dataset] = sql.strip().rstrip(";")
        if spec is not None:
            self.specs[dataset] = spec

    def _require_runner(self) -> QueryRunner:
        if self.runner is None:
            raise ConfigError(
                "BigQueryAdapter needs a runner: "
                "BigQueryAdapter(runner=DBAPIRunner(bigquery.dbapi.connect(...)))"
            )
        return self.runner

    def _dataset(self, name: str) -> DatasetId:
        return normalize_table(name, system="bigquery", instance=self.project)

    def _split(self, dataset: DatasetId) -> tuple[str, str, str]:
        parts = dataset.name.split(".")
        if len(parts) == 2 and self.project:
            parts = [self.project, *parts]
        if len(parts) != 3:
            raise ConfigError(
                f"{dataset} is not project.dataset.table, which INFORMATION_SCHEMA needs"
            )
        return parts[0], parts[1], parts[2]

    # -- partition spec --------------------------------------------------------

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        """The partitioning column, with its grain read from live partition ids.

        `INFORMATION_SCHEMA.COLUMNS` names the column but not the granularity, and
        the column type does not imply it either — a DATE column can be partitioned
        monthly. The ids are the only place the grain is actually recorded.
        """
        if dataset in self.specs:
            return self.specs[dataset]

        runner = self._require_runner()
        project, bq_dataset, table = self._split(dataset)
        columns = runner.rows(
            _PARTITION_COLUMN.format(project=project, dataset=bq_dataset), {"table": table}
        )
        if not columns:
            self.specs[dataset] = UNPARTITIONED
            return UNPARTITIONED

        column = str(_row(columns[0], "column_name"))
        grain: Grain | None = None
        for row in runner.rows(
            _PARTITIONS.format(project=project, dataset=bq_dataset), {"table": table}
        ):
            grain = grain_of(str(_row(row, "partition_id") or ""))
            if grain is not None:
                break

        spec = PartitionSpec.of(
            PartitionField.time(column, grain) if grain else PartitionField.value(column)
        )
        self.specs[dataset] = spec
        return spec

    # -- change detection ------------------------------------------------------

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Partitions whose `last_modified_time` moved. Exact, and one query."""
        runner = self._require_runner()
        project, bq_dataset, table = self._split(dataset)
        spec = self.describe_partitioning(dataset)
        field_name = spec.fields[0].name if spec.fields else None

        cutoff: datetime | None = None
        if since:
            try:
                cutoff = datetime.fromisoformat(since)
            except ValueError:
                cutoff = None

        partitions: set[KeyPredicate] = set()
        high_water: datetime | None = cutoff

        for row in runner.rows(
            _PARTITIONS.format(project=project, dataset=bq_dataset), {"table": table}
        ):
            modified = _row(row, "last_modified_time")
            if isinstance(modified, str):
                try:
                    modified = datetime.fromisoformat(modified)
                except ValueError:
                    modified = None
            if not isinstance(modified, datetime):
                continue
            if high_water is None or modified > high_water:
                high_water = modified
            if cutoff is not None and modified <= cutoff:
                continue

            partition_id = str(_row(row, "partition_id") or "")
            if field_name is None or partition_id == UNPARTITIONED_ID:
                partitions.add(KeyPredicate.unbounded(spec))
                continue
            if partition_id == NULL_PARTITION:
                # A real partition holding rows whose partition column is null.
                partitions.add(KeyPredicate(bindings=((field_name, None),)))
                continue

            parsed = parse_partition_id(partition_id)
            partitions.add(KeyPredicate(bindings=((field_name, parsed if parsed else ANY),)))

        return ChangeSet(
            partitions=frozenset(partitions),
            token=high_water.isoformat() if high_water else (since or ""),
            complete=True,
        )

    # -- lineage ---------------------------------------------------------------

    def fetch_queries(self, since: Token | None) -> Iterable[QueryEvent]:
        """Job SQL from INFORMATION_SCHEMA.JOBS, for the parser to work on."""
        runner = self._require_runner()
        for row in runner.rows(
            _JOBS.format(region=self.region),
            {"since": since or "1970-01-01T00:00:00", "limit": self.limit},
        ):
            yield QueryEvent(
                sql=str(_row(row, "query") or ""),
                dialect=self.dialect,
                query_id=str(_row(row, "job_id") or ""),
                started=_row(row, "creation_time"),
            )

    def fetch_lineage(self, since: Token | None) -> Iterable[LineageEvent]:
        """Dataset-level edges from `referenced_tables`, without parsing.

        Coarser than the parser but never wrong: BigQuery records exactly which
        tables a job read. Useful as a cross-check, and as the answer when a query
        uses a UDF or a script the parser cannot follow.
        """
        runner = self._require_runner()
        for row in runner.rows(
            _JOBS.format(region=self.region),
            {"since": since or "1970-01-01T00:00:00", "limit": self.limit},
        ):
            destination = _row(row, "destination_table")
            if not destination:
                continue
            target = self._dataset(_qualified(destination))
            for reference in _row(row, "referenced_tables") or []:
                source = self._dataset(_qualified(reference))
                if source != target:
                    yield LineageEvent(
                        src=source,
                        dst=target,
                        evidence=f"bigquery:jobs:{_row(row, 'job_id')}",
                        observed=_row(row, "creation_time"),
                    )

    # -- rebuild ---------------------------------------------------------------

    def render_rebuild(self, dataset: DatasetId, partitions: Iterable[KeyPredicate]) -> list[str]:
        model = self.models.get(dataset)
        if model is None:
            raise KeyError(f"no model registered for {dataset}")
        predicate = render_predicate(self.describe_partitioning(dataset), partitions, quote="`")
        table = f"`{dataset.name}`"
        return [
            f"DELETE FROM {table} WHERE {predicate}",
            f"INSERT INTO {table} SELECT * FROM ({model}) AS _fathom_rebuild WHERE {predicate}",
        ]

    def erase(
        self,
        dataset: DatasetId,
        *,
        key_column: str,
        subject: Any,
        partitions: Sequence[KeyPredicate],
    ) -> int:
        runner = self._require_runner()
        if dataset in self.models:
            for statement in self.render_rebuild(dataset, partitions):
                runner.rows(statement)
            return 0
        predicate = render_predicate(self.describe_partitioning(dataset), partitions, quote="`")
        runner.rows(
            f"DELETE FROM `{dataset.name}` "
            f"WHERE {quote_identifier(key_column, '`')} = {literal(subject)} AND ({predicate})"
        )
        return 0


def _qualified(reference: Any) -> str:
    """BigQuery table references arrive as dicts or as already-joined strings."""
    if isinstance(reference, str):
        return reference
    if isinstance(reference, dict):
        parts = [
            reference.get("project_id") or reference.get("projectId"),
            reference.get("dataset_id") or reference.get("datasetId"),
            reference.get("table_id") or reference.get("tableId"),
        ]
        return ".".join(str(p) for p in parts if p)
    return str(reference)
