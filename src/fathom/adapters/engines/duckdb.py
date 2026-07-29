"""DuckDB engine adapter.

The reference engine, and the one that makes end-to-end verification possible: it
can extract lineage from model SQL, render a partition-scoped rebuild, execute it,
and let a test assert the result matches a full rebuild.

Rebuilds are rendered as delete-then-insert filtered on the *target's* partition
columns:

    DELETE FROM target WHERE <predicate>;
    INSERT INTO target SELECT * FROM (<model sql>) WHERE <predicate>;

Filtering the model's output rather than its inputs is correct regardless of what
the model does internally, which is the property that matters. It does mean we rely
on the engine to push the predicate down into the sources; DuckDB does this for
simple comparisons, but a model wrapped around an opaque UDF will scan more than it
strictly needs. Correct and occasionally slow beats fast and occasionally wrong.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...core.types import (
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
from ..base import ChangeSet, LineageEvent, QueryEvent, Token, register
from ..predicates import identifier as _identifier
from ..predicates import literal as _literal
from ..predicates import render_predicate

__all__ = ["DuckDBEngine"]


@register("duckdb")
@dataclass
class DuckDBEngine:
    """Models are registered as (dataset, SQL) pairs; everything else follows."""

    database: str = ":memory:"
    name: str = "duckdb"
    dialect: str = "duckdb"
    models: dict[DatasetId, str] = field(default_factory=dict)
    specs: dict[DatasetId, PartitionSpec] = field(default_factory=dict)
    capabilities: Capabilities = Capabilities(
        lineage=LineageSource.QUERY_LOG,
        change=ChangeSource.WATERMARK,
        pushdown=Pushdown.APPROX_DISTINCT,
        erasure=ErasureMode.REWRITE,
        column_lineage=True,
        partition_aware=True,
    )
    _conn: Any = field(default=None, repr=False, compare=False)

    # -- connection ------------------------------------------------------------

    def connect(self) -> Any:
        """The live connection, opened on first use."""
        if self._conn is None:
            import duckdb

            self._conn = duckdb.connect(self.database)
        return self._conn

    def close(self) -> None:
        """Close the connection if one was opened."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- model registry --------------------------------------------------------

    def register_model(
        self, dataset: DatasetId, sql: str, spec: PartitionSpec = UNPARTITIONED
    ) -> None:
        """Register the SELECT that defines a dataset. Not a CREATE, just the query."""
        self.models[dataset] = sql.strip().rstrip(";")
        self.specs[dataset] = spec

    def describe_partitioning(self, dataset: DatasetId) -> PartitionSpec:
        """The declared spec for a table, or `UNPARTITIONED`."""
        return self.specs.get(dataset, UNPARTITIONED)

    def table_name(self, dataset: DatasetId) -> str:
        """The engine-native identifier for a dataset."""
        return dataset.name

    # -- lineage ---------------------------------------------------------------

    def fetch_lineage(self, since: Token | None) -> Iterable[LineageEvent]:
        """DuckDB maintains no lineage of its own; the query log is the only source."""
        return ()

    def fetch_queries(self, since: Token | None) -> Iterable[QueryEvent]:
        """Registered models, shaped as the CTAS statements they are equivalent to."""
        for dataset, sql in sorted(self.models.items(), key=lambda kv: str(kv[0])):
            yield QueryEvent(
                sql=f"CREATE TABLE {self.table_name(dataset)} AS {sql}",
                dialect=self.dialect,
                query_id=str(dataset),
            )

    def changed(self, dataset: DatasetId, since: Token | None) -> ChangeSet:
        """Engines do not own change detection here; a catalog or storage adapter does."""
        return ChangeSet(token=since or "", complete=True)

    # -- rebuild ---------------------------------------------------------------

    def render_rebuild(self, dataset: DatasetId, partitions: Iterable[KeyPredicate]) -> list[str]:
        """Statements rebuilding exactly these partitions. Nothing is executed here."""
        model = self.models.get(dataset)
        if model is None:
            raise KeyError(f"no model registered for {dataset}")
        spec = self.describe_partitioning(dataset)
        predicate = render_predicate(spec, partitions)
        table = self.table_name(dataset)
        return [
            f"DELETE FROM {table} WHERE {predicate}",
            f"INSERT INTO {table} SELECT * FROM ({model}) AS _fathom_rebuild WHERE {predicate}",
        ]

    def render_create(self, dataset: DatasetId) -> str:
        """A statement creating the table from its registered model."""
        model = self.models.get(dataset)
        if model is None:
            raise KeyError(f"no model registered for {dataset}")
        return f"CREATE OR REPLACE TABLE {self.table_name(dataset)} AS {model}"

    def apply(self, dataset: DatasetId, partitions: Iterable[KeyPredicate]) -> list[str]:
        """Execute a partition-scoped rebuild. Returns the statements that ran."""
        statements = self.render_rebuild(dataset, partitions)
        conn = self.connect()
        conn.execute("BEGIN TRANSACTION")
        try:
            for statement in statements:
                conn.execute(statement)
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return statements

    def full_rebuild(self, dataset: DatasetId) -> str:
        """Rebuild the whole dataset. Used as shadow mode's ground truth."""
        statement = self.render_create(dataset)
        self.connect().execute(statement)
        return statement

    # -- reads -----------------------------------------------------------------

    def erase(
        self,
        dataset: DatasetId,
        *,
        key_column: str,
        subject: Any,
        partitions: Sequence[KeyPredicate],
    ) -> int:
        """Destroy a subject's rows, or re-derive a table so they stop appearing.

        These are different operations and conflating them is a real source of
        incomplete erasures. A source table holds the subject's rows directly, so
        it gets a DELETE. A derived table holds *aggregates* of them, often with no
        subject column at all — deleting there is impossible, and the correct action
        is to rebuild the affected partitions from the already-erased upstream.

        That makes ordering load-bearing: erase sources before rebuilding anything
        downstream, which is exactly the order the plan hands you.
        """
        spec = self.describe_partitioning(dataset)
        table = self.table_name(dataset)
        predicate = render_predicate(spec, partitions)

        if dataset in self.models:
            before = self._row_count(dataset)
            self.apply(dataset, partitions)
            return max(0, before - self._row_count(dataset))

        if key_column not in self.columns_of(dataset):
            raise ValueError(
                f"{dataset} has no column {key_column!r} and no model to rebuild from; "
                "the subject's data cannot be located here"
            )

        conn = self.connect()
        before = self._row_count(dataset)
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                f"DELETE FROM {table} WHERE {_identifier(key_column)} = "
                f"{_literal(subject)} AND ({predicate})"
            )
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return max(0, before - self._row_count(dataset))

    def _row_count(self, dataset: DatasetId) -> int:
        row = self.connect().execute(f"SELECT COUNT(*) FROM {self.table_name(dataset)}").fetchone()
        return int(row[0]) if row else 0

    def rows(self, dataset: DatasetId) -> list[tuple[Any, ...]]:
        """Every row of a table, as tuples."""
        cur = self.connect().execute(f"SELECT * FROM {self.table_name(dataset)}")
        return [tuple(r) for r in cur.fetchall()]

    def _key_from_row(self, spec: PartitionSpec, row: tuple[Any, ...]) -> KeyPredicate:
        """Coerce a result row's partition columns into a key at the declared grain."""
        from ...core.grains import truncate

        bindings: list[tuple[str, object]] = []
        for f, value in zip(spec.fields, row, strict=False):
            if f.kind == "time" and value is not None:
                assert f.grain is not None
                coerced = (
                    value
                    if isinstance(value, datetime)
                    else datetime.combine(value, datetime.min.time())
                )
                bindings.append((f.name, truncate(coerced, f.grain)))
            else:
                bindings.append((f.name, value))
        return KeyPredicate(bindings=tuple(bindings))

    def columns_of(self, dataset: DatasetId) -> list[str]:
        """Column names of a table, in declaration order."""
        cur = self.connect().execute(f"DESCRIBE {self.table_name(dataset)}")
        return [str(r[0]) for r in cur.fetchall()]

    def partitions_present(self, dataset: DatasetId) -> set[KeyPredicate]:
        """Distinct partition keys currently materialized in a table."""
        spec = self.describe_partitioning(dataset)
        if not spec.fields:
            return {KeyPredicate()}
        columns = ", ".join(f'"{f.name}"' for f in spec.fields)
        cur = self.connect().execute(f"SELECT DISTINCT {columns} FROM {self.table_name(dataset)}")
        return {self._key_from_row(spec, tuple(row)) for row in cur.fetchall()}

    def fingerprints(self, dataset: DatasetId) -> dict[KeyPredicate, str]:
        """A content hash per partition, for detecting what a rebuild actually changed.

        Order-independent: rows are sorted inside the aggregate, so two builds that
        produce the same rows in a different order hash identically. Without that,
        shadow mode would report every partition as changed on every run.
        """
        spec = self.describe_partitioning(dataset)
        table = self.table_name(dataset)
        columns = self.columns_of(dataset)
        if not columns:
            return {}

        # CHR(31) and CHR(30) are unit/record separators, so a value containing a
        # comma or pipe cannot forge a column boundary.
        row_expr = (
            "CONCAT_WS(CHR(31), "
            + ", ".join(f'COALESCE("{c}"::VARCHAR, CHR(0))' for c in columns)
            + ")"
        )
        digest = f"MD5(STRING_AGG({row_expr}, CHR(30) ORDER BY {row_expr}))"

        if spec.fields:
            group = ", ".join(f'"{f.name}"' for f in spec.fields)
            sql = f"SELECT {group}, {digest} AS fp FROM {table} GROUP BY {group}"
            width = len(spec.fields)
        else:
            sql = f"SELECT {digest} AS fp FROM {table}"
            width = 0

        out: dict[KeyPredicate, str] = {}
        for row in self.connect().execute(sql).fetchall():
            key = self._key_from_row(spec, tuple(row[:width])) if width else KeyPredicate()
            out[key] = str(row[width])
        return out
