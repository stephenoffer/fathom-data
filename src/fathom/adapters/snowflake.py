"""Snowflake adapter.

Snowflake is the best case for lineage and the worst case for change detection, and
the adapter is shaped by both facts.

**Lineage is free and exact.** `ACCOUNT_USAGE.ACCESS_HISTORY` records, per query,
which columns of which objects fed which columns of which objects. No SQL parsing,
no dialect edge cases, no ambiguity about unqualified columns. Nothing else in this
project gets column lineage this cheaply.

**Partitions do not exist.** Micro-partitions are an internal detail with no
addressable identity, so "which partition changed" has no native answer. Two
strategies, in order of preference:

1. A declared **watermark column** (`_loaded_at`, `updated_at`). One cheap
   `SELECT DISTINCT <partition columns> WHERE <watermark> > token` gives real
   partition granularity.
2. `INFORMATION_SCHEMA.TABLES.LAST_ALTERED`, which answers only "did this table
   change at all". The result is a single unbounded partition — correct, and coarse.

The one operational fact that surprises people: `ACCOUNT_USAGE` views lag by up to
about three hours. That is fine for planning a nightly backfill and wrong for
intraday decisions, so `capabilities.freshness_lag` states it rather than leaving it
to be discovered.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ..errors import ConfigError
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
    PartitionSpec,
    Pushdown,
)
from .base import ChangeSet, LineageEvent, QueryEvent, Token, register
from .predicates import render_predicate
from .sql_runner import QueryRunner, quote_identifier

__all__ = ["SnowflakeAdapter"]

ACCESS_HISTORY_LAG = timedelta(hours=3)

_ACCESS_HISTORY = """
SELECT query_id, query_start_time, objects_modified, base_objects_accessed
FROM snowflake.account_usage.access_history
WHERE query_start_time > TO_TIMESTAMP_LTZ(%(since)s)
  AND ARRAY_SIZE(objects_modified) > 0
ORDER BY query_start_time
LIMIT %(limit)s
"""

_QUERY_HISTORY = """
SELECT query_id, query_text, database_name, schema_name, start_time
FROM snowflake.account_usage.query_history
WHERE start_time > TO_TIMESTAMP_LTZ(%(since)s)
  AND execution_status = 'SUCCESS'
  AND query_type IN ('CREATE_TABLE_AS_SELECT', 'INSERT', 'MERGE', 'UPDATE', 'DELETE')
ORDER BY start_time
LIMIT %(limit)s
"""

_LAST_ALTERED = """
SELECT last_altered
FROM {database}.information_schema.tables
WHERE table_schema = %(schema)s AND table_name = %(table)s
"""


def _as_list(value: Any) -> list[dict[str, Any]]:
    """ACCESS_HISTORY arrays arrive as JSON text or as parsed lists, per driver."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def _row(row: dict[str, Any], *names: str) -> Any:
    """Read a column case-insensitively; Snowflake returns names uppercased."""
    for name in names:
        for key in (name, name.upper(), name.lower()):
            if key in row:
                return row[key]
    return None


@register("snowflake")
@dataclass
class SnowflakeAdapter:
    """Lineage from ACCESS_HISTORY; change detection from a declared watermark."""

    runner: QueryRunner | None = None
    account: str | None = None
    name: str = "snowflake"
    dialect: str = "snowflake"
    specs: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    watermarks: dict[DatasetId, str] = field(default_factory=dict)
    models: dict[DatasetId, str] = field(default_factory=dict)
    limit: int = 100_000
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.NATIVE,
        change=ChangeSource.WATERMARK,
        pushdown=Pushdown.APPROX_DISTINCT,
        erasure=ErasureMode.REWRITE,
        column_lineage=True,
        freshness_lag=ACCESS_HISTORY_LAG,
        partition_aware=False,
    )

    # -- configuration ---------------------------------------------------------

    def declare(
        self,
        dataset: DatasetId,
        spec: PartitionSpec,
        *,
        watermark: str | None = None,
    ) -> None:
        """Declare partitioning, and optionally the column that marks new rows."""
        self.specs[dataset] = spec
        if watermark:
            self.watermarks[dataset] = quote_identifier(watermark).strip('"')

    def register_model(
        self, dataset: DatasetId, sql: str, spec: PartitionSpec | None = None
    ) -> None:
        self.models[dataset] = sql.strip().rstrip(";")
        if spec is not None:
            self.specs[dataset] = spec

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        return self.specs.get(dataset, UNPARTITIONED)

    def _require_runner(self) -> QueryRunner:
        if self.runner is None:
            raise ConfigError(
                "SnowflakeAdapter needs a runner: "
                "SnowflakeAdapter(runner=DBAPIRunner(snowflake.connector.connect(...)))"
            )
        return self.runner

    def _dataset(self, name: str) -> DatasetId:
        return normalize_table(name, system="snowflake", instance=self.account)

    # -- lineage ---------------------------------------------------------------

    def fetch_lineage(self, since: Token | None) -> Iterable[LineageEvent]:
        """Column-level lineage straight from ACCESS_HISTORY.

        `objects_modified[].columns[].directSources[]` is exactly a column edge, so
        there is nothing to parse and nothing to guess.
        """
        runner = self._require_runner()
        rows = runner.rows(
            _ACCESS_HISTORY, {"since": since or "1970-01-01 00:00:00", "limit": self.limit}
        )

        for row in rows:
            observed = _row(row, "query_start_time")
            query_id = str(_row(row, "query_id") or "")
            for modified in _as_list(_row(row, "objects_modified")):
                target_name = modified.get("objectName")
                if not target_name:
                    continue
                target = self._dataset(str(target_name))

                by_source: dict[DatasetId, list[tuple[str, str]]] = {}
                for column in modified.get("columns") or []:
                    target_column = column.get("columnName")
                    if not target_column:
                        continue
                    for source in (column.get("directSources") or []) + (
                        column.get("baseSources") or []
                    ):
                        source_name = source.get("objectName")
                        source_column = source.get("columnName")
                        if not source_name or not source_column:
                            continue
                        src = self._dataset(str(source_name))
                        if src == target:
                            continue
                        pair = (str(source_column), str(target_column))
                        if pair not in by_source.setdefault(src, []):
                            by_source[src].append(pair)

                # A statement can read a table without any column feeding the output
                # (a join key, a filter). That is still a dependency.
                for accessed in _as_list(_row(row, "base_objects_accessed")):
                    accessed_name = accessed.get("objectName")
                    if not accessed_name:
                        continue
                    src = self._dataset(str(accessed_name))
                    if src != target:
                        by_source.setdefault(src, [])

                for src, columns in by_source.items():
                    yield LineageEvent(
                        src=src,
                        dst=target,
                        columns=tuple(columns),
                        evidence=f"snowflake:access_history:{query_id}",
                        observed=observed if isinstance(observed, datetime) else None,
                    )

    def fetch_queries(self, since: Token | None) -> Iterable[QueryEvent]:
        """Statement text, as a fallback where ACCESS_HISTORY is not enabled.

        ACCESS_HISTORY is Enterprise Edition and up. On Standard, this is the only
        lineage available, and it costs a SQL parse.
        """
        runner = self._require_runner()
        for row in runner.rows(
            _QUERY_HISTORY, {"since": since or "1970-01-01 00:00:00", "limit": self.limit}
        ):
            yield QueryEvent(
                sql=str(_row(row, "query_text") or ""),
                dialect=self.dialect,
                query_id=str(_row(row, "query_id") or ""),
                started=_row(row, "start_time"),
                default_database=_row(row, "database_name"),
                default_schema=_row(row, "schema_name"),
            )

    def lineage_token(self, events: Iterable[LineageEvent]) -> Token:
        """The resume point, held back by the ACCOUNT_USAGE lag.

        Advancing the token to the newest event we saw would permanently skip rows
        that had not yet landed in the view when we read it.
        """
        seen = [e.observed for e in events if e.observed is not None]
        if not seen:
            return ""
        return (max(seen) - ACCESS_HISTORY_LAG).astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # -- change detection ------------------------------------------------------

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Changed partitions, at whatever granularity this table can support."""
        runner = self._require_runner()
        spec = self.describe_partitioning(dataset)
        watermark = self.watermarks.get(dataset)

        if watermark and spec.fields:
            return self._changed_by_watermark(runner, dataset, spec, watermark, since)
        return self._changed_by_last_altered(runner, dataset, spec, since)

    def _changed_by_watermark(
        self,
        runner: QueryRunner,
        dataset: DatasetId,
        spec: PartitionSpec,
        watermark: str,
        since: Token | None,
    ) -> ChangeSet:
        columns = ", ".join(quote_identifier(f.name) for f in spec.fields)
        sql = (
            f"SELECT DISTINCT {columns}, MAX({quote_identifier(watermark)}) AS _fathom_high_water "
            f"FROM {dataset.name} "
            f"WHERE {quote_identifier(watermark)} > TO_TIMESTAMP_LTZ(%(since)s) "
            f"GROUP BY {columns}"
        )
        rows = runner.rows(sql, {"since": since or "1970-01-01 00:00:00"})

        partitions: set[KeyPredicate] = set()
        high_water: datetime | None = None
        for row in rows:
            bindings: list[tuple[str, object]] = []
            for f in spec.fields:
                value = _row(row, f.name)
                if f.kind == "time" and isinstance(value, datetime):
                    assert f.grain is not None
                    from ..grains import truncate

                    bindings.append((f.name, truncate(value, f.grain)))
                else:
                    bindings.append((f.name, ANY if value is None else value))
            partitions.add(KeyPredicate(bindings=tuple(bindings)))

            mark = _row(row, "_fathom_high_water")
            if isinstance(mark, datetime) and (high_water is None or mark > high_water):
                high_water = mark

        return ChangeSet(
            partitions=frozenset(partitions),
            token=high_water.strftime("%Y-%m-%d %H:%M:%S") if high_water else (since or ""),
            complete=True,
        )

    def _changed_by_last_altered(
        self,
        runner: QueryRunner,
        dataset: DatasetId,
        spec: PartitionSpec,
        since: Token | None,
    ) -> ChangeSet:
        parts = dataset.name.split(".")
        if len(parts) != 3:
            raise ConfigError(
                f"{dataset} is not a fully qualified DATABASE.SCHEMA.TABLE, which "
                "INFORMATION_SCHEMA lookups require"
            )
        database, schema, table = parts
        rows = runner.rows(
            _LAST_ALTERED.format(database=database), {"schema": schema, "table": table}
        )
        altered = _row(rows[0], "last_altered") if rows else None
        if not isinstance(altered, datetime):
            return ChangeSet(token=since or "", complete=True)

        token = altered.strftime("%Y-%m-%d %H:%M:%S")
        if since and token <= since:
            return ChangeSet(token=since, complete=True)

        # No watermark declared, so the honest answer is "somewhere in this table".
        return ChangeSet(
            partitions=frozenset({KeyPredicate.unbounded(spec)}),
            token=token,
            complete=True,
        )

    # -- rebuild ---------------------------------------------------------------

    def render_rebuild(self, dataset: DatasetId, partitions: Iterable[KeyPredicate]) -> list[str]:
        model = self.models.get(dataset)
        if model is None:
            raise KeyError(f"no model registered for {dataset}")
        predicate = render_predicate(self.describe_partitioning(dataset), partitions)
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
        """Delete a subject's rows, or re-derive a modeled table without them."""
        runner = self._require_runner()
        if dataset in self.models:
            for statement in self.render_rebuild(dataset, partitions):
                runner.rows(statement)
            return 0  # Snowflake reports affected rows out of band; treat as unknown.

        from .predicates import literal

        predicate = render_predicate(self.describe_partitioning(dataset), partitions)
        runner.rows(
            f"DELETE FROM {dataset.name} "
            f"WHERE {quote_identifier(key_column)} = {literal(subject)} AND ({predicate})"
        )
        return 0
