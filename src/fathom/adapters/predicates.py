"""Rendering partition predicates as SQL.

Shared by every engine adapter, because a predicate that is right for DuckDB and
wrong for Snowflake would produce a rebuild that silently covers the wrong rows.

Two decisions worth stating:

**ANSI typed literals.** `TIMESTAMP '2026-03-14 00:00:00'` parses in DuckDB,
Snowflake, Databricks, BigQuery, Trino, and Postgres. Provider-specific spellings
(`TO_TIMESTAMP`, `::timestamp`) would need a dialect table and gain nothing.

**Half-open ranges, not equality.** A day partition may be stored as a full
timestamp, so `dt = '2026-03-14'` matches nothing while `dt >= '2026-03-14' AND
dt < '2026-03-15'` matches the day. Equality here is a whole class of silent
under-rebuilds.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from ..core.grains import step
from ..core.types import ANY, KeyPredicate, PartitionSpec

__all__ = ["identifier", "literal", "render_predicate"]


def identifier(name: str, *, quote: str = '"') -> str:
    """Quote a column or table name, escaping the quote character inside it.

    Values were escaped and identifiers were not, so a partition field or key column
    containing the quote character broke straight out of its own quoting. These names
    reach the renderer from `fathom.yml` and from `--key-column`, and the statements
    they land in include `DELETE`.
    """
    return quote + name.replace(quote, quote * 2) + quote


def literal(value: Any) -> str:
    """A portable SQL literal. Strings are escaped, never interpolated raw."""
    if isinstance(value, datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _key_sql(spec: PartitionSpec, key: KeyPredicate, quote: str) -> str:
    """SQL for one partition key. Unconstrained fields contribute nothing."""
    clauses: list[str] = []
    for f in spec.fields:
        value = key.get(f.name)
        column = identifier(f.name, quote=quote)
        if value is ANY:
            continue
        if value is None:
            clauses.append(f"{column} IS NULL")
            continue
        if f.kind == "time" and isinstance(value, datetime):
            assert f.grain is not None
            upper = step(value, 1, f.grain)
            clauses.append(f"{column} >= {literal(value)} AND {column} < {literal(upper)}")
        else:
            clauses.append(f"{column} = {literal(value)}")
    return "(" + " AND ".join(clauses) + ")" if clauses else "TRUE"


def render_predicate(spec: PartitionSpec, keys: Iterable[KeyPredicate], *, quote: str = '"') -> str:
    """A WHERE clause covering every key. Returns `TRUE` when anything is unbounded.

    `quote` is the identifier quoting character: `"` everywhere except Databricks
    and Spark, which use backticks.
    """
    parts: list[str] = []
    for key in sorted(keys, key=str):
        clause = _key_sql(spec, key, quote)
        if clause == "TRUE":
            # One unbounded key subsumes every narrower one; ORing them would produce
            # a predicate that is technically correct and needlessly enormous.
            return "TRUE"
        if clause not in parts:
            parts.append(clause)
    return " OR ".join(parts) if parts else "TRUE"
