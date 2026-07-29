"""Databricks adapter.

Unity Catalog maintains lineage for us, so `system.access.column_lineage` gives
column edges with no SQL parsing — the same deal Snowflake offers, from a different
table.

Change detection takes a different route. Databricks tables are Delta tables, and
`DESCRIBE DETAIL` reports the storage location, so rather than reimplementing
snapshot diffing over `DESCRIBE HISTORY` we hand the location to `DeltaCatalog` and
get exact partition-level changes from the transaction log. Reusing the adapter we
already have beats a second, weaker implementation of the same idea.

Two constraints worth knowing before relying on this:

- System tables need `system.access` enabled on the metastore, and lineage rows can
  take up to about two hours to appear.
- Lineage covers table and column dependencies, not partition mappings. Those still
  come from declared specs or from parsing the model SQL.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ...core.errors import ConfigError
from ...core.grains import Grain
from ...core.ids import normalize, normalize_table
from ...core.types import (
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
from ..base import ChangeSet, LineageEvent, QueryEvent, Token, register
from ..catalogs.delta import DeltaCatalog
from ..predicates import literal, render_predicate
from ..sql_runner import QueryRunner, quote_identifier

__all__ = ["DatabricksAdapter"]

SYSTEM_TABLE_LAG = timedelta(hours=2)

_COLUMN_LINEAGE = """
SELECT source_table_full_name, target_table_full_name,
       source_column_name, target_column_name, event_time
FROM system.access.column_lineage
WHERE event_time > :since
  AND source_table_full_name IS NOT NULL
  AND target_table_full_name IS NOT NULL
ORDER BY event_time
LIMIT :limit
"""

_TABLE_LINEAGE = """
SELECT source_table_full_name, target_table_full_name, event_time
FROM system.access.table_lineage
WHERE event_time > :since
  AND source_table_full_name IS NOT NULL
  AND target_table_full_name IS NOT NULL
ORDER BY event_time
LIMIT :limit
"""

_QUERY_HISTORY = """
SELECT statement_id, statement_text, start_time
FROM system.query.history
WHERE start_time > :since AND execution_status = 'FINISHED'
ORDER BY start_time
LIMIT :limit
"""

# Databricks column types that imply a time grain when used as a partition column.
_TYPE_GRAIN = {"date": Grain.DAY, "timestamp": Grain.HOUR, "timestamp_ntz": Grain.HOUR}


def _row(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        for key in (name, name.lower(), name.upper()):
            if key in row:
                return row[key]
    return None


@register("databricks")
@dataclass
class DatabricksAdapter:
    """Lineage from Unity Catalog; change detection delegated to the Delta log."""

    runner: QueryRunner | None = None
    workspace: str | None = None
    name: str = "databricks"
    dialect: str = "databricks"
    specs: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    locations: dict[DatasetId, str] = field(default_factory=dict)
    models: dict[DatasetId, str] = field(default_factory=dict)
    storage_options: dict[str, Any] = field(default_factory=dict)
    limit: int = 100_000
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.NATIVE,
        change=ChangeSource.SNAPSHOT_DIFF,
        pushdown=Pushdown.APPROX_DISTINCT,
        erasure=ErasureMode.DELETE_VECTOR,
        column_lineage=True,
        freshness_lag=SYSTEM_TABLE_LAG,
        partition_aware=True,
    )

    # -- configuration ---------------------------------------------------------

    def declare(
        self, dataset: DatasetId, spec: PartitionSpec, *, location: str | None = None
    ) -> None:
        """Record a partition spec for a table."""
        self.specs[dataset] = spec
        if location:
            self.locations[dataset] = location

    def register_model(
        self, dataset: DatasetId, sql: str, spec: PartitionSpec | None = None
    ) -> None:
        """Associate a dataset with the SQL that produces it."""
        self.models[dataset] = sql.strip().rstrip(";")
        if spec is not None:
            self.specs[dataset] = spec

    def _require_runner(self) -> QueryRunner:
        if self.runner is None:
            raise ConfigError(
                "DatabricksAdapter needs a runner: "
                "DatabricksAdapter(runner=DBAPIRunner(databricks.sql.connect(...)))"
            )
        return self.runner

    def _dataset(self, name: str) -> DatasetId:
        return normalize_table(name, system="databricks", instance=self.workspace)

    # -- table metadata --------------------------------------------------------

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        """Partition columns from `DESCRIBE DETAIL`, with grains from column types."""
        if dataset in self.specs:
            return self.specs[dataset]

        runner = self._require_runner()
        detail = runner.rows(f"DESCRIBE DETAIL {dataset.name}")
        if not detail:
            return UNPARTITIONED

        columns = _row(detail[0], "partitionColumns") or []
        if isinstance(columns, str):
            columns = [c.strip() for c in columns.strip("[]").split(",") if c.strip()]
        if not columns:
            self.specs[dataset] = UNPARTITIONED
            return UNPARTITIONED

        types: dict[str, str] = {}
        for row in runner.rows(f"DESCRIBE TABLE {dataset.name}"):
            column = str(_row(row, "col_name") or "")
            if column and not column.startswith("#"):
                types[column] = str(_row(row, "data_type") or "").lower()

        fields = []
        for column in columns:
            grain = _TYPE_GRAIN.get(types.get(str(column), ""))
            fields.append(
                PartitionField.time(str(column), grain)
                if grain
                else PartitionField.value(str(column))
            )
        spec = PartitionSpec.of(*fields)
        self.specs[dataset] = spec
        return spec

    def location_of(self, dataset: DatasetId) -> str | None:
        """The table's storage location, which is what makes Delta delegation possible."""
        if dataset in self.locations:
            return self.locations[dataset]
        runner = self._require_runner()
        detail = runner.rows(f"DESCRIBE DETAIL {dataset.name}")
        location = str(_row(detail[0], "location") or "") if detail else ""
        if location:
            self.locations[dataset] = location
        return location or None

    # -- lineage ---------------------------------------------------------------

    def fetch_lineage(self, since: Token | None) -> Iterable[LineageEvent]:
        """Column edges from Unity Catalog, falling back to table-level rows.

        `column_lineage` misses dependencies where a table is read but no column of
        it reaches the output — a join key, a filter. `table_lineage` catches those,
        and dropping them would leave a real dependency out of the graph.
        """
        runner = self._require_runner()
        params = {"since": since or "1970-01-01T00:00:00", "limit": self.limit}

        edges: dict[tuple[DatasetId, DatasetId], list[tuple[str, str]]] = {}
        observed: dict[tuple[DatasetId, DatasetId], datetime] = {}

        for row in runner.rows(_COLUMN_LINEAGE, params):
            src = self._dataset(str(_row(row, "source_table_full_name")))
            dst = self._dataset(str(_row(row, "target_table_full_name")))
            if src == dst:
                continue
            source_column = _row(row, "source_column_name")
            target_column = _row(row, "target_column_name")
            key = (src, dst)
            if source_column and target_column:
                pair = (str(source_column), str(target_column))
                if pair not in edges.setdefault(key, []):
                    edges.setdefault(key, []).append(pair)
            else:
                edges.setdefault(key, [])
            when = _row(row, "event_time")
            if isinstance(when, datetime):
                observed[key] = max(observed.get(key, when), when)

        for row in runner.rows(_TABLE_LINEAGE, params):
            src = self._dataset(str(_row(row, "source_table_full_name")))
            dst = self._dataset(str(_row(row, "target_table_full_name")))
            if src != dst:
                edges.setdefault((src, dst), [])

        for (src, dst), columns in edges.items():
            yield LineageEvent(
                src=src,
                dst=dst,
                columns=tuple(columns),
                evidence="databricks:unity_catalog",
                observed=observed.get((src, dst)),
            )

    def fetch_queries(self, since: Token | None) -> Iterable[QueryEvent]:
        """Statement text from query history, where system lineage is unavailable."""
        runner = self._require_runner()
        for row in runner.rows(
            _QUERY_HISTORY, {"since": since or "1970-01-01T00:00:00", "limit": self.limit}
        ):
            yield QueryEvent(
                sql=str(_row(row, "statement_text") or ""),
                dialect=self.dialect,
                query_id=str(_row(row, "statement_id") or ""),
                started=_row(row, "start_time"),
            )

    def lineage_token(self, events: Iterable[LineageEvent]) -> Token:
        """Resume point, held back by system-table lag so nothing is skipped."""
        seen = [e.observed for e in events if e.observed is not None]
        if not seen:
            return ""
        return (max(seen) - SYSTEM_TABLE_LAG).astimezone(UTC).isoformat()

    # -- change detection ------------------------------------------------------

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Delegate to the Delta transaction log at the table's storage location.

        Exact partition granularity, at the cost of one `DESCRIBE DETAIL` and read
        access to the table's files.
        """
        location = self.location_of(dataset)
        if not location:
            raise ConfigError(
                f"cannot locate storage for {dataset}; DESCRIBE DETAIL returned no "
                "location, so change detection has nothing to read"
            )
        catalog = DeltaCatalog(storage_options=self.storage_options)
        target = normalize(location)
        spec = self.describe_partitioning(dataset)
        if spec.fields:
            catalog.declare(target, spec)
        changes = catalog.changed(target, since)
        return changes

    # -- rebuild ---------------------------------------------------------------

    def render_rebuild(self, dataset: DatasetId, partitions: Iterable[KeyPredicate]) -> list[str]:
        """Statements rebuilding exactly these partitions. Never executed here."""
        model = self.models.get(dataset)
        if model is None:
            raise KeyError(f"no model registered for {dataset}")
        # Databricks quotes identifiers with backticks, not double quotes.
        predicate = render_predicate(self.describe_partitioning(dataset), partitions, quote="`")
        return [
            f"DELETE FROM {dataset.name} WHERE {predicate}",
            f"INSERT INTO {dataset.name} "
            f"SELECT * FROM ({model}) AS _fathom_rebuild WHERE {predicate}",
        ]

    def erase(
        self,
        dataset: DatasetId,
        *,
        key_column: str,
        subject: Any,
        partitions: Sequence[KeyPredicate],
    ) -> int:
        """Delete a subject's rows from the named partitions."""
        runner = self._require_runner()
        if dataset in self.models:
            for statement in self.render_rebuild(dataset, partitions):
                runner.rows(statement)
            return 0
        predicate = render_predicate(self.describe_partitioning(dataset), partitions, quote="`")
        runner.rows(
            f"DELETE FROM {dataset.name} "
            f"WHERE {quote_identifier(key_column, '`')} = {literal(subject)} AND ({predicate})"
        )
        return 0
