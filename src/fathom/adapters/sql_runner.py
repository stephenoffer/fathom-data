"""Running SQL against a warehouse, without depending on its driver.

Every warehouse adapter talks to this instead of to `snowflake.connector` or
`databricks.sql` directly. Three reasons, in order of importance:

1. **Testability.** A warehouse adapter whose only test path is a live account is a
   warehouse adapter with no tests. `RecordedRunner` replays captured rows, so the
   query-shaping and result-parsing logic — where the bugs actually live — is
   covered offline.
2. **No hard dependencies.** Installing this library should not drag in three cloud
   SDKs. `DBAPIRunner` wraps anything DB-API 2.0, which all of them are.
3. **Observability.** One place to log, time, and cap queries, rather than three.

The cost is that adapters cannot use driver-specific features. So far none of them
need to.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.errors import FathomError

__all__ = [
    "DBAPIRunner",
    "QueryError",
    "QueryRunner",
    "RecordedRunner",
    "chunked",
    "quote_identifier",
]


class QueryError(FathomError):
    """A query failed. Carries the statement so the failure is diagnosable."""

    def __init__(self, sql: str, reason: str, hint: str = "") -> None:
        self.sql = sql
        self.reason = reason
        # Statements can be enormous; the head is what identifies them.
        head = " ".join(sql.split())[:240]
        message = f"query failed: {reason}\n  {head}"
        if hint:
            message += f"\n  {hint}"
        super().__init__(message)


@runtime_checkable
class QueryRunner(Protocol):
    """Executes SQL and returns rows as dicts keyed by column name."""

    def rows(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> list[dict[str, Any]]: ...


@dataclass
class DBAPIRunner:
    """Wraps any DB-API 2.0 connection.

    Works with `snowflake-connector-python`, `databricks-sql-connector`,
    `google-cloud-bigquery-dbapi`, `psycopg`, and anything else that follows PEP 249.
    """

    connection: Any
    name: str = "dbapi"
    max_rows: int | None = 500_000

    def rows(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> list[dict[str, Any]]:
        cursor = self.connection.cursor()
        try:
            try:
                cursor.execute(sql, params) if params else cursor.execute(sql)
            except Exception as exc:  # noqa: BLE001 - each driver has its own hierarchy
                raise QueryError(sql, f"{type(exc).__name__}: {exc}", _hint_for(exc)) from exc

            description = cursor.description or ()
            columns = [str(d[0]) for d in description]
            fetched = cursor.fetchall()
            if self.max_rows is not None and len(fetched) > self.max_rows:
                raise QueryError(
                    sql,
                    f"returned {len(fetched)} rows, over the {self.max_rows} cap",
                    "narrow the time window, or raise max_rows if you meant it",
                )
            return [dict(zip(columns, row, strict=False)) for row in fetched]
        finally:
            with _ignore_errors():
                cursor.close()


@dataclass
class RecordedRunner:
    """Replays captured responses. Used by tests and by `--dry-run` diagnostics.

    Matching is by substring against a normalized statement, so a test can key on
    the distinctive table name rather than reproducing whitespace exactly.
    """

    responses: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    name: str = "recorded"
    executed: list[str] = field(default_factory=list)
    strict: bool = True

    def rows(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> list[dict[str, Any]]:
        self.executed.append(sql)
        normalized = " ".join(sql.split()).upper()
        for key, rows in self.responses.items():
            if " ".join(key.split()).upper() in normalized:
                return [dict(r) for r in rows]
        if self.strict:
            raise QueryError(
                sql,
                "no recorded response matched",
                f"recorded keys: {sorted(self.responses)}",
            )
        return []


class _ignore_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


_HINTS = {
    "ProgrammingError": "check the object name and that the role can see it",
    "DatabaseError": "check connectivity and that the warehouse or cluster is running",
    "Forbidden": "the principal lacks permission on this object",
    "NotFound": "the object does not exist, or the role cannot see it",
}


def _hint_for(exc: BaseException) -> str:
    for cls in type(exc).__mro__:
        hint = _HINTS.get(cls.__name__)
        if hint:
            return hint
    return ""


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def quote_identifier(name: str, quote: str = '"') -> str:
    """Quote an identifier for interpolation.

    Identifiers cannot be bound as parameters, so anything reaching a statement by
    interpolation is validated first. A column name arriving from a config file is
    still untrusted input.
    """
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(
            f"{name!r} is not a valid identifier; identifiers cannot be parameterized, "
            "so only plain names are accepted here"
        )
    return f"{quote}{name}{quote}"


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Split a sequence for warehouses with statement or parameter limits."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
