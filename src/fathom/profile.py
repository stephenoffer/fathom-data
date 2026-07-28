"""Profiles, and drift between them.

The cheap path matters more than the thorough one. Parquet and ORC footers carry
per-row-group min/max, null counts, and sizes — enough for most of a profile, at
metadata cost, without reading a single data page. Reading actual rows is reserved
for what footers cannot answer: value vocabularies and distributions.

Row-group statistics also give granularity *below* the directory partition, which
no catalog can offer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .types import DatasetId, KeyPredicate

__all__ = [
    "ColumnProfile",
    "Finding",
    "Profile",
    "Severity",
    "drift",
    "profile_parquet",
]


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class ColumnProfile:
    """What we know about one column of one partition."""

    name: str
    dtype: str
    row_count: int = 0
    null_count: int | None = None
    min: Any | None = None
    max: Any | None = None
    distinct_estimate: int | None = None
    byte_size: int | None = None

    @property
    def null_rate(self) -> float | None:
        if self.null_count is None or self.row_count == 0:
            return None
        return self.null_count / self.row_count


@dataclass(frozen=True)
class Profile:
    """A point-in-time statistical fingerprint of one dataset partition."""

    dataset: DatasetId
    partition: KeyPredicate = field(default_factory=KeyPredicate)
    row_count: int = 0
    columns: tuple[ColumnProfile, ...] = ()
    file_count: int = 0
    source: str = "footer"  # footer | scan | pushdown

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@dataclass(frozen=True)
class Finding:
    """One detected difference between two profiles."""

    column: str | None
    kind: str
    severity: Severity
    detail: str
    before: Any = None
    after: Any = None

    def __str__(self) -> str:
        where = f"{self.column}: " if self.column else ""
        return f"[{self.severity.value}] {where}{self.detail}"


def _stat(value: Any) -> Any:
    """Parquet statistics come back as native values already; normalize bytes."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return value


def _merge_min(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    try:
        return a if a <= b else b
    except TypeError:  # mixed types across files; give up rather than guess
        return None


def _merge_max(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    try:
        return a if a >= b else b
    except TypeError:
        return None


def profile_parquet(
    paths: Sequence[str | Path],
    *,
    dataset: DatasetId,
    partition: KeyPredicate | None = None,
    fs: Any = None,
) -> Profile:
    """Build a profile from Parquet footers alone. No data pages are read.

    Pass `fs` to profile object storage; without it, paths are opened directly from
    the local filesystem.

    Statistics are optional in the Parquet spec, so a column whose writer omitted
    them yields `None` rather than a fabricated value. Callers decide whether that
    warrants a scan.
    """
    import pyarrow.parquet as pq

    accum: dict[str, dict[str, Any]] = {}
    total_rows = 0
    files = 0

    for path in paths:
        if fs is not None:
            with fs.open(str(path)) as handle:
                md = pq.ParquetFile(handle).metadata
        else:
            md = pq.ParquetFile(str(path)).metadata
        files += 1
        total_rows += md.num_rows
        schema = md.schema.to_arrow_schema()

        for rg_index in range(md.num_row_groups):
            rg = md.row_group(rg_index)
            for col_index in range(rg.num_columns):
                col = rg.column(col_index)
                name = col.path_in_schema
                entry = accum.setdefault(
                    name,
                    {
                        "dtype": str(schema.field(name.split(".")[0]).type)
                        if name.split(".")[0] in schema.names
                        else "unknown",
                        "rows": 0,
                        "nulls": 0,
                        "nulls_known": True,
                        "min": None,
                        "max": None,
                        "bytes": 0,
                        "distinct": None,
                    },
                )
                entry["rows"] += col.num_values
                entry["bytes"] += col.total_compressed_size or 0

                stats = col.statistics
                if stats is None:
                    entry["nulls_known"] = False
                    continue
                if stats.null_count is None:
                    entry["nulls_known"] = False
                else:
                    entry["nulls"] += stats.null_count
                if stats.has_min_max:
                    entry["min"] = _merge_min(entry["min"], _stat(stats.min))
                    entry["max"] = _merge_max(entry["max"], _stat(stats.max))
                # distinct_count is legal but almost never populated by writers.
                distinct = getattr(stats, "distinct_count", None)
                if distinct:
                    entry["distinct"] = (entry["distinct"] or 0) + distinct

    columns = tuple(
        ColumnProfile(
            name=name,
            dtype=e["dtype"],
            row_count=e["rows"],
            null_count=e["nulls"] if e["nulls_known"] else None,
            min=e["min"],
            max=e["max"],
            distinct_estimate=e["distinct"],
            byte_size=e["bytes"],
        )
        for name, e in sorted(accum.items())
    )

    return Profile(
        dataset=dataset,
        partition=partition or KeyPredicate(),
        row_count=total_rows,
        columns=columns,
        file_count=files,
        source="footer",
    )


def drift(
    before: Profile,
    after: Profile,
    *,
    null_rate_tolerance: float = 0.05,
    row_count_tolerance: float = 0.25,
    min_rows: int = 1000,
) -> list[Finding]:
    """Compare two profiles of the same dataset.

    `min_rows` exists because a threshold without a sample-size guard is a noise
    generator: a partition of 40 rows will cross any null-rate tolerance by chance.
    Below the floor we downgrade to INFO rather than staying silent, so a genuinely
    tiny partition is still visible without paging anyone.
    """
    findings: list[Finding] = []
    underpowered = min(before.row_count, after.row_count) < min_rows

    def sev(level: Severity) -> Severity:
        return Severity.INFO if underpowered else level

    gone = set(before.column_names) - set(after.column_names)
    added = set(after.column_names) - set(before.column_names)
    for name in sorted(gone):
        findings.append(
            Finding(name, "column_removed", Severity.ERROR, "column disappeared", name, None)
        )
    for name in sorted(added):
        findings.append(
            Finding(name, "column_added", Severity.INFO, "new column appeared", None, name)
        )

    if before.row_count > 0:
        delta = (after.row_count - before.row_count) / before.row_count
        if abs(delta) > row_count_tolerance:
            findings.append(
                Finding(
                    None,
                    "row_count_shift",
                    sev(Severity.WARN),
                    f"row count moved {delta:+.1%} ({before.row_count} -> {after.row_count})",
                    before.row_count,
                    after.row_count,
                )
            )

    for name in sorted(set(before.column_names) & set(after.column_names)):
        b, a = before.column(name), after.column(name)
        assert b is not None and a is not None

        if b.dtype != a.dtype:
            findings.append(
                Finding(
                    name, "type_change", Severity.ERROR, f"{b.dtype} -> {a.dtype}", b.dtype, a.dtype
                )
            )

        br, ar = b.null_rate, a.null_rate
        if br is not None and ar is not None and abs(ar - br) > null_rate_tolerance:
            findings.append(
                Finding(
                    name,
                    "null_rate_shift",
                    sev(Severity.WARN),
                    f"null rate {br:.1%} -> {ar:.1%}",
                    br,
                    ar,
                )
            )

        # A shrinking range is the interesting direction: values that used to appear
        # and no longer do usually mean an upstream filter changed.
        if b.min is not None and a.min is not None:
            try:
                if a.min > b.min:
                    findings.append(
                        Finding(
                            name,
                            "min_raised",
                            sev(Severity.INFO),
                            f"min {b.min} -> {a.min}",
                            b.min,
                            a.min,
                        )
                    )
            except TypeError:
                pass
        if b.max is not None and a.max is not None:
            try:
                if a.max < b.max:
                    findings.append(
                        Finding(
                            name,
                            "max_lowered",
                            sev(Severity.INFO),
                            f"max {b.max} -> {a.max}",
                            b.max,
                            a.max,
                        )
                    )
            except TypeError:
                pass

    return findings


def summarize(findings: Iterable[Finding]) -> str:
    items = list(findings)
    if not items:
        return "no drift detected"
    order = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
    return "\n".join(str(f) for f in sorted(items, key=lambda f: order[f.severity]))
