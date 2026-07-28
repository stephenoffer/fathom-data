"""Lineage from SQL text.

The universal fallback. Native lineage tables and execution-plan listeners are both
better when available, but they require platform privileges — installing a JAR on a
Trino coordinator, enabling Unity Catalog. Parsing the query log needs nothing but
read access to history, so it is the path that always works.

sqlglot does the dialect normalization, which is what makes one extractor serve
every engine on the matrix instead of one per engine.

What this deliberately does not do is guess. An unqualified column with three
candidate tables, an opaque UDF, a MERGE with a correlated condition: all of these
produce a dataset-level edge with an `UNBOUNDED` partition mapping rather than a
plausible-looking column edge that might be wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

from ..ids import normalize_table
from ..partitions import UNBOUNDED, FieldMapping, PartitionMapping, Passthrough, TimeWindow
from ..types import UNPARTITIONED, DatasetId, PartitionSpec

__all__ = ["Extraction", "extract"]

# Truncation units as they appear across dialects, mapped to our grains.
_UNIT_TO_GRAIN = {
    "hour": "hour",
    "hh": "hour",
    "day": "day",
    "dd": "day",
    "dayofmonth": "day",
    "month": "month",
    "mm": "month",
    "mon": "month",
    "year": "year",
    "yyyy": "year",
    "yy": "year",
}


@dataclass
class Extraction:
    """What one statement told us."""

    target: DatasetId | None = None
    sources: tuple[DatasetId, ...] = ()
    column_edges: dict[DatasetId, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    mappings: dict[DatasetId, PartitionMapping] = field(default_factory=dict)
    evidence: str = "sql"
    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.target is None or not self.sources


def _table_id(
    table: exp.Table,
    *,
    system: str,
    instance: str | None,
    default_database: str | None,
    default_schema: str | None,
) -> DatasetId:
    parts = [p.name for p in (table.args.get("catalog"), table.args.get("db")) if p] + [table.name]
    return normalize_table(
        ".".join(parts),
        system=system,
        instance=instance,
        default_database=default_database,
        default_schema=default_schema,
    )


def _target_of(statement: Any) -> exp.Table | None:
    if isinstance(statement, exp.Create):
        this = statement.this
        if isinstance(this, exp.Table):
            return this
        if isinstance(this, exp.Schema) and isinstance(this.this, exp.Table):
            return this.this
    if isinstance(statement, exp.Insert):
        this = statement.this
        if isinstance(this, exp.Table):
            return this
        if isinstance(this, exp.Schema) and isinstance(this.this, exp.Table):
            return this.this
    if isinstance(statement, exp.Merge):
        this = statement.this
        if isinstance(this, exp.Table):
            return this
        if isinstance(this, exp.Alias) and isinstance(this.this, exp.Table):
            return this.this
    return None


def _outer_select(statement: Any) -> exp.Select | None:
    if isinstance(statement, exp.Select):
        return statement
    expression = statement.args.get("expression") or statement.args.get("this")
    if isinstance(expression, exp.Select):
        return expression
    found = statement.find(exp.Select)
    return found if isinstance(found, exp.Select) else None


def _unit_grain(expression: Any) -> str | None:
    """The grain a truncation reduces to, or None if this is not a truncation."""
    unit: exp.Expression | None = None
    if isinstance(expression, exp.DateTrunc | exp.TimestampTrunc):
        unit = expression.args.get("unit")
    elif isinstance(expression, exp.Anonymous) and expression.name.lower() in {
        "date_trunc",
        "datetrunc",
        "timestamp_trunc",
    }:
        args = expression.expressions
        # Dialects disagree on argument order; the literal is the unit either way.
        unit = next((a for a in args if isinstance(a, exp.Literal) and a.is_string), None)
    if unit is None:
        return None
    return _UNIT_TO_GRAIN.get(unit.name.strip("'\" ").lower())


def _source_column(expression: Any) -> exp.Column | None:
    """The single column an expression reads, or None if zero or many."""
    columns = list(expression.find_all(exp.Column))
    return columns[0] if len(columns) == 1 else None


def _infer_field(
    projection: Any,
    src_spec: PartitionSpec,
    dst_field_name: str,
    dst_spec: PartitionSpec,
) -> FieldMapping:
    """Classify how one output partition field derives from the input's."""
    dst_field = dst_spec.field(dst_field_name)
    if projection is None or dst_field is None:
        return UNBOUNDED

    inner = projection.unalias() if isinstance(projection, exp.Alias) else projection
    column = _source_column(inner)
    if column is None:
        return UNBOUNDED

    src_field = src_spec.field(column.name)
    if src_field is None:
        return UNBOUNDED

    if dst_field.kind == "value":
        # A value field must be carried through untouched to stay a passthrough.
        return Passthrough(column.name) if isinstance(inner, exp.Column) else UNBOUNDED

    assert dst_field.grain is not None
    if src_field.kind != "time" or src_field.grain is None:
        return UNBOUNDED

    if isinstance(inner, exp.Column):
        # Straight passthrough of a time column: grains come from the two specs.
        if dst_field.grain < src_field.grain:
            return UNBOUNDED  # refinement; no useful bound
        return TimeWindow(column.name, 0, 0, src_field.grain, dst_field.grain)

    grain_name = _unit_grain(inner)
    if grain_name is None:
        return UNBOUNDED

    from ..grains import Grain

    truncated = Grain.parse(grain_name)
    if truncated < src_field.grain or dst_field.grain < src_field.grain:
        return UNBOUNDED
    return TimeWindow(column.name, 0, 0, src_field.grain, dst_field.grain)


def extract(
    sql: str,
    *,
    dialect: str,
    system: str | None = None,
    instance: str | None = None,
    default_database: str | None = None,
    default_schema: str | None = None,
    specs: Mapping[DatasetId, PartitionSpec] | None = None,
) -> list[Extraction]:
    """Extract lineage from one or more statements.

    `specs` supplies known partition specs; without them every mapping is
    `UNBOUNDED`, which is correct but imprecise. Statements that fail to parse are
    skipped with a note rather than raising, because one malformed entry in a query
    log must not take down a whole ingest run.
    """
    system = system or dialect
    specs = specs or {}
    out: list[Extraction] = []

    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001 - query logs contain anything
        return [Extraction(notes=(f"unparseable ({type(exc).__name__}): {exc}",))]

    for statement in statements:
        if statement is None:
            continue

        target_table = _target_of(statement)
        if target_table is None:
            continue

        ids = {
            "system": system,
            "instance": instance,
            "default_database": default_database,
            "default_schema": default_schema,
        }
        target = _table_id(target_table, **ids)  # type: ignore[arg-type]

        select = _outer_select(statement)
        source_tables = [
            t
            for t in statement.find_all(exp.Table)
            if t is not target_table and not isinstance(t.parent, exp.Drop)
        ]
        sources: list[DatasetId] = []
        for t in source_tables:
            ds = _table_id(t, **ids)  # type: ignore[arg-type]
            if ds != target and ds not in sources:
                sources.append(ds)

        if not sources:
            continue

        extraction = Extraction(target=target, sources=tuple(sources))
        dst_spec = specs.get(target, UNPARTITIONED)
        ambiguous = len(sources) > 1

        # Column edges: only when a column's owning table is unambiguous.
        alias_by_table: dict[str, DatasetId] = {}
        for t in source_tables:
            ds = _table_id(t, **ids)  # type: ignore[arg-type]
            alias_by_table[t.alias_or_name.lower()] = ds

        edges: dict[DatasetId, list[tuple[str, str]]] = {ds: [] for ds in sources}
        if select is not None:
            for projection in select.expressions:
                out_name = projection.alias_or_name
                if not out_name or out_name == "*":
                    continue
                for column in projection.find_all(exp.Column):
                    qualifier = (column.table or "").lower()
                    owner = alias_by_table.get(qualifier)
                    if owner is None:
                        if ambiguous:
                            continue  # unqualified with several candidates: refuse to guess
                        owner = sources[0]
                    pair = (column.name, out_name)
                    if pair not in edges[owner]:
                        edges[owner].append(pair)

        notes: list[str] = []
        if ambiguous:
            notes.append("multiple source tables; unqualified columns were not attributed")
        if isinstance(statement, exp.Merge):
            notes.append("MERGE: row-level effects are not bounded, partitions widened")

        projections = {p.alias_or_name: p for p in (select.expressions if select else [])}
        for ds in sources:
            src_spec = specs.get(ds, UNPARTITIONED)
            if isinstance(statement, exp.Merge) or not dst_spec.fields:
                extraction.mappings[ds] = PartitionMapping.unknown(dst_spec)
            else:
                extraction.mappings[ds] = PartitionMapping(
                    fields=tuple(
                        (f.name, _infer_field(projections.get(f.name), src_spec, f.name, dst_spec))
                        for f in dst_spec.fields
                    )
                )
            extraction.column_edges[ds] = tuple(edges[ds])

        extraction.notes = tuple(notes)
        out.append(extraction)

    return out
