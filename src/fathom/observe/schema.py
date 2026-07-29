"""Structural differences between two profiles of the same dataset.

Distinct from `profile.drift`, which asks whether the *values* moved. This asks
whether the *shape* moved, which is the thing that breaks a downstream query outright
rather than making its answer quietly wrong. Both matter and they fail differently,
so they stay apart rather than merging into one list whose entries mean two things.

Direction matters in the naming: `before` is the committed state, `after` is the
proposal, and `added` means present in `after` only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..core.types import DatasetId
from .profile import ColumnProfile, Profile, Severity

__all__ = [
    "ColumnChange",
    "SchemaDiff",
    "breaking_schema_changes",
    "diff_profiles",
    "diff_schemas",
    "is_breaking",
    "worst_severity",
]


@dataclass(frozen=True)
class ColumnChange:
    """One column whose type or nullability moved."""

    column: str
    before: ColumnProfile | None
    after: ColumnProfile | None

    @property
    def kind(self) -> str:
        """What sort of change this is, for grouping in a report."""
        if self.before is None:
            return "added"
        if self.after is None:
            return "removed"
        if self.before.dtype != self.after.dtype:
            return "retyped"
        return "changed"

    @property
    def is_breaking(self) -> bool:
        """A removal or a type change breaks a consumer; an addition does not."""
        return self.kind in {"removed", "retyped"}

    def __str__(self) -> str:
        if self.kind == "added":
            return f"+ {self.column}"
        if self.kind == "removed":
            return f"- {self.column}"
        b = self.before.dtype if self.before else "?"
        a = self.after.dtype if self.after else "?"
        return f"~ {self.column}: {b} => {a}"


@dataclass
class SchemaDiff:
    """Column-level differences between two profiles of the same dataset."""

    dataset: DatasetId
    changes: list[ColumnChange] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when the two schemas agree."""
        return not self.changes

    @property
    def breaking(self) -> list[ColumnChange]:
        """Changes that break a reader: columns removed, or types narrowed."""
        return [c for c in self.changes if c.is_breaking]

    @property
    def added(self) -> list[str]:
        """Columns present after and not before."""
        return [c.column for c in self.changes if c.kind == "added"]

    @property
    def removed(self) -> list[str]:
        """Columns present before and not after."""
        return [c.column for c in self.changes if c.kind == "removed"]

    @property
    def retyped(self) -> list[str]:
        """Columns whose type changed."""
        return [c.column for c in self.changes if c.kind == "retyped"]

    def summary(self) -> str:
        """The comparison as text, breaking changes first."""
        if self.is_empty:
            return f"{self.dataset}: schema unchanged"
        verdict = " [BREAKING]" if self.breaking else ""
        return f"{self.dataset}: {len(self.changes)} column change(s){verdict}\n" + "\n".join(
            f"  {c}" for c in self.changes
        )


def diff_schemas(before: Profile, after: Profile) -> SchemaDiff:
    """Structural comparison of two profiles, ignoring statistics.

    Distinct from `profile.drift`, which asks whether the *values* moved. This asks
    whether the *shape* moved, which is the thing that breaks a downstream query
    outright rather than making its answer wrong.
    """
    out = SchemaDiff(dataset=after.dataset)
    names = sorted(set(before.column_names) | set(after.column_names))
    for name in names:
        b, a = before.column(name), after.column(name)
        if b is None or a is None or b.dtype != a.dtype:
            out.changes.append(ColumnChange(name, b, a))
    return out


def breaking_schema_changes(before: Profile, after: Profile) -> list[ColumnChange]:
    """Only the changes that break a consumer outright."""
    return diff_schemas(before, after).breaking


def is_breaking(before: Profile, after: Profile) -> bool:
    """True when any column was removed or retyped."""
    return bool(breaking_schema_changes(before, after))


# -- profiles ------------------------------------------------------------------


def diff_profiles(
    before: Profile,
    after: Profile,
    *,
    null_rate_tolerance: float = 0.05,
    row_count_tolerance: float = 0.25,
) -> tuple[SchemaDiff, list[str]]:
    """Schema differences and value drift together, as one review artifact.

    Returns the structural diff plus rendered drift lines. Both matter and they fail
    for different reasons, so they stay separate rather than merging into one list
    whose entries mean different things.
    """
    from ..observe.profile import drift

    findings = drift(
        before,
        after,
        null_rate_tolerance=null_rate_tolerance,
        row_count_tolerance=row_count_tolerance,
    )
    ordered = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
    lines = [str(f) for f in sorted(findings, key=lambda f: ordered[f.severity])]
    return diff_schemas(before, after), lines


# -- plans ---------------------------------------------------------------------


def worst_severity(findings: Iterable[Any]) -> str:
    """The highest severity among some findings.

    Reads `Finding.severity` where it is there, and falls back to scanning rendered
    text only for callers that pass strings. Deriving severity purely by looking for
    `[error]` in formatted output made the answer depend on a display string: any
    change to how a finding renders silently turned every severity into "none",
    which reads as "nothing wrong".
    """
    levels = {"error": 0, "warn": 1, "info": 2}
    best = "none"
    best_rank = 99
    for finding in findings:
        level = getattr(finding, "severity", None)
        if level is not None:
            name = getattr(level, "value", str(level))
        else:
            text = str(finding)
            name = next((lvl for lvl in levels if f"[{lvl}]" in text), "none")
        rank = levels.get(name, 99)
        if rank < best_rank:
            best, best_rank = name, rank
    return best
