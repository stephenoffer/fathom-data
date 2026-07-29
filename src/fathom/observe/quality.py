"""Expectations over profiles, and generating them so nobody has to write them.

Data tests fail for two reasons. They are tedious to write, so there are never
enough of them; and the ones that exist encode what someone assumed on the day they
wrote them, which stops being true.

Both problems have the same answer here. Expectations are checked against a
`Profile`, which is already computed from footers at metadata cost — so a check costs
nothing beyond what profiling already did. And `learn` generates a whole suite from
an observed profile, with bounds widened by an explicit margin, so a team starts with
a hundred reasonable expectations instead of an empty file.

Two rules that keep generated suites from becoming noise:

- **Bounds widen, never tighten.** A learned range is the observed range plus a
  margin. A suite that fires the first time a value exceeds yesterday's maximum by
  one is a suite that gets disabled in a week.
- **A learned expectation carries `learned=True`.** Reviewing what a generator
  assumed is a different activity from reviewing what a person asserted, and a suite
  that hides which is which cannot be audited.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.types import DatasetId
from .profile import Finding, Profile, Severity

__all__ = [
    "Expectation",
    "Suite",
    "SuiteResult",
    "check",
    "column_count_between",
    "dtype_is",
    "from_dict",
    "in_range",
    "learn",
    "merge",
    "name_matches",
    "not_empty",
    "run",
    "to_dict",
    "max_below",
    "min_above",
    "not_null",
    "null_rate_below",
    "row_count_between",
    "schema_matches",
    "unique",
]

# How far a learned bound is widened past what was observed. 10% of the observed
# range, so a stable column stays stable and a volatile one is not fenced in tight.
DEFAULT_MARGIN = 0.1


@dataclass(frozen=True)
class Expectation:
    """One assertion about a profile.

    `kind` names the check, `params` carries its arguments. Kept data rather than
    closures so a suite round-trips through JSON and can live in a repository next to
    the models it guards.
    """

    kind: str
    column: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.ERROR
    learned: bool = False
    note: str = ""

    def __str__(self) -> str:
        where = f"{self.column}: " if self.column else ""
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        mark = " (learned)" if self.learned else ""
        return f"{where}{self.kind}({rendered}){mark}"


# -- builders ------------------------------------------------------------------


def not_null(column: str, *, severity: Severity = Severity.ERROR) -> Expectation:
    """The column has no nulls at all."""
    return Expectation("null_rate_below", column, {"max_rate": 0.0}, severity)


def null_rate_below(
    column: str, max_rate: float, *, severity: Severity = Severity.WARN
) -> Expectation:
    """The column's null rate stays under a ceiling."""
    return Expectation("null_rate_below", column, {"max_rate": max_rate}, severity)


def unique(column: str, *, severity: Severity = Severity.ERROR) -> Expectation:
    """Every value is distinct.

    Only checkable where the writer populated a distinct count, which most do not.
    An unverifiable expectation reports as skipped rather than passing.
    """
    return Expectation("unique", column, {}, severity)


def min_above(column: str, bound: float, *, severity: Severity = Severity.ERROR) -> Expectation:
    """The column's minimum stays at or above a floor."""
    return Expectation("min_above", column, {"bound": bound}, severity)


def max_below(column: str, bound: float, *, severity: Severity = Severity.ERROR) -> Expectation:
    """The column's maximum stays at or below a ceiling."""
    return Expectation("max_below", column, {"bound": bound}, severity)


def in_range(
    column: str, low: float, high: float, *, severity: Severity = Severity.ERROR
) -> Expectation:
    """The column's values stay inside a range."""
    return Expectation("in_range", column, {"low": low, "high": high}, severity)


def dtype_is(column: str, dtype: str, *, severity: Severity = Severity.ERROR) -> Expectation:
    """The column keeps its type. The cheapest check with the highest hit rate."""
    return Expectation("dtype_is", column, {"dtype": dtype}, severity)


def row_count_between(low: int, high: int, *, severity: Severity = Severity.WARN) -> Expectation:
    """The partition's row count stays in a band."""
    return Expectation("row_count_between", None, {"low": low, "high": high}, severity)


def column_count_between(low: int, high: int, *, severity: Severity = Severity.WARN) -> Expectation:
    """The dataset keeps roughly the columns it had."""
    return Expectation("column_count_between", None, {"low": low, "high": high}, severity)


def schema_matches(columns: Sequence[str], *, severity: Severity = Severity.ERROR) -> Expectation:
    """These columns are all present. Extra columns are allowed."""
    return Expectation("schema_matches", None, {"columns": sorted(columns)}, severity)


def name_matches(column: str, pattern: str, *, severity: Severity = Severity.WARN) -> Expectation:
    """The column's observed min and max both match a pattern.

    A weak but genuinely useful check on footer statistics alone: an id column whose
    bounds stop matching its format has had something else written into it.
    """
    return Expectation("name_matches", column, {"pattern": pattern}, severity)


def not_empty(*, severity: Severity = Severity.ERROR) -> Expectation:
    """The partition has at least one row."""
    return Expectation("row_count_between", None, {"low": 1, "high": None}, severity)


# -- evaluation ----------------------------------------------------------------


class _Skipped:
    """The profile carries nothing that could decide this expectation either way.

    Distinct from passing. A suite reporting PASS over expectations it could not
    evaluate is worse than one reporting nothing, because it is read as assurance.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SKIPPED"


SKIPPED = _Skipped()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluate(expectation: Expectation, profile: Profile) -> Finding | _Skipped | None:
    """Check one expectation.

    `None` means it passed. `SKIPPED` means the profile held nothing that could
    decide it — footers carry no distinct count, or no statistics for the column —
    which must not be reported as a pass.
    """
    params = expectation.params

    if expectation.kind == "row_count_between":
        low, high = params.get("low"), params.get("high")
        if low is not None and profile.row_count < low:
            return Finding(
                None,
                "row_count_below",
                expectation.severity,
                f"row count {profile.row_count} is below {low}",
                low,
                profile.row_count,
            )
        if high is not None and profile.row_count > high:
            return Finding(
                None,
                "row_count_above",
                expectation.severity,
                f"row count {profile.row_count} is above {high}",
                high,
                profile.row_count,
            )
        return None

    if expectation.kind == "column_count_between":
        count = len(profile.columns)
        low, high = params.get("low"), params.get("high")
        if (low is not None and count < low) or (high is not None and count > high):
            return Finding(
                None,
                "column_count",
                expectation.severity,
                f"column count {count} is outside [{low}, {high}]",
                None,
                count,
            )
        return None

    if expectation.kind == "schema_matches":
        missing = sorted(set(params.get("columns", ())) - set(profile.column_names))
        if missing:
            return Finding(
                None,
                "schema_mismatch",
                expectation.severity,
                f"missing column(s): {', '.join(missing)}",
                params.get("columns"),
                None,
            )
        return None

    column = profile.column(expectation.column or "")
    if column is None:
        return Finding(
            expectation.column,
            "column_missing",
            expectation.severity,
            "column is absent from the profile",
            expectation.column,
            None,
        )

    if expectation.kind == "null_rate_below":
        rate = column.null_rate
        if rate is None:
            return SKIPPED  # the writer omitted null counts
        ceiling = float(params.get("max_rate", 0.0))
        if rate > ceiling:
            return Finding(
                column.name,
                "null_rate",
                expectation.severity,
                f"null rate {rate:.1%} exceeds {ceiling:.1%}",
                ceiling,
                rate,
            )
        return None

    if expectation.kind == "unique":
        if column.distinct_estimate is None:
            return SKIPPED  # footers almost never carry a distinct count
        if column.distinct_estimate < column.row_count:
            return Finding(
                column.name,
                "not_unique",
                expectation.severity,
                f"{column.row_count - column.distinct_estimate} duplicate value(s)",
                column.row_count,
                column.distinct_estimate,
            )
        return None

    if expectation.kind == "dtype_is":
        expected = str(params.get("dtype", ""))
        if column.dtype != expected:
            return Finding(
                column.name,
                "dtype_change",
                expectation.severity,
                f"type is {column.dtype}, expected {expected}",
                expected,
                column.dtype,
            )
        return None

    if expectation.kind in {"min_above", "max_below", "in_range"}:
        low = _as_float(column.min)
        high = _as_float(column.max)
        if expectation.kind == "min_above":
            if low is None:
                return SKIPPED
            bound = float(params["bound"])
            if low < bound:
                return Finding(
                    column.name,
                    "min_below_bound",
                    expectation.severity,
                    f"minimum {low} is below {bound}",
                    bound,
                    low,
                )
        elif expectation.kind == "max_below":
            if high is None:
                return SKIPPED
            bound = float(params["bound"])
            if high > bound:
                return Finding(
                    column.name,
                    "max_above_bound",
                    expectation.severity,
                    f"maximum {high} is above {bound}",
                    bound,
                    high,
                )
        else:
            if low is None and high is None:
                return SKIPPED
            lo, hi = float(params["low"]), float(params["high"])
            if (low is not None and low < lo) or (high is not None and high > hi):
                return Finding(
                    column.name,
                    "out_of_range",
                    expectation.severity,
                    f"range [{low}, {high}] escapes [{lo}, {hi}]",
                    (lo, hi),
                    (low, high),
                )
        return None

    if expectation.kind == "name_matches":
        if column.min is None and column.max is None:
            return SKIPPED
        pattern = re.compile(str(params.get("pattern", ".*")))
        for value in (column.min, column.max):
            if value is not None and not pattern.search(str(value)):
                return Finding(
                    column.name,
                    "pattern_mismatch",
                    expectation.severity,
                    f"bound {value!r} does not match {pattern.pattern}",
                    pattern.pattern,
                    value,
                )
        return None

    # An expectation kind nobody implemented is not a passing expectation.
    return SKIPPED


@dataclass
class Suite:
    """A named set of expectations for one dataset."""

    dataset: DatasetId
    expectations: list[Expectation] = field(default_factory=list)
    name: str = ""

    def add(self, *expectations: Expectation) -> Suite:
        """Append expectations to this suite and return it, for chaining."""
        self.expectations.extend(expectations)
        return self

    @property
    def learned(self) -> list[Expectation]:
        """Expectations derived from an observed profile."""
        return [e for e in self.expectations if e.learned]

    @property
    def asserted(self) -> list[Expectation]:
        """Expectations a person wrote, as opposed to ones a generator proposed."""
        return [e for e in self.expectations if not e.learned]

    def summary(self) -> str:
        """The suite as text: how many expectations, and how many were learned."""
        return (
            f"{self.name or self.dataset}: {len(self.expectations)} expectation(s), "
            f"{len(self.learned)} learned"
        )


@dataclass
class SuiteResult:
    """What a suite found when run against a profile."""

    dataset: DatasetId
    findings: list[Finding] = field(default_factory=list)
    checked: int = 0
    # Expectations the profile could not decide either way. Kept separate from
    # `checked` so a suite cannot report assurance it never established.
    skipped: list[Expectation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when nothing failed at ERROR severity.

        Answers a different question from `complete`. A suite can pass because
        nothing failed, or because nothing could be checked.
        """
        return not any(f.severity is Severity.ERROR for f in self.findings)

    @property
    def complete(self) -> bool:
        """True when every expectation was actually evaluated.

        `passed` and `complete` answer different questions, and a green suite that is
        not complete is the one worth looking at: nothing failed because most of it
        was never checked.
        """
        return not self.skipped

    @property
    def errors(self) -> list[Finding]:
        """Findings at ERROR severity."""
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        """Findings at WARN severity."""
        return [f for f in self.findings if f.severity is Severity.WARN]

    def summary(self) -> str:
        """The suite as text: how many expectations, and how many were learned."""
        verdict = "PASS" if self.passed else "FAIL"
        line = (
            f"{self.dataset}: {verdict} — {self.checked} expectation(s), "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )
        if self.skipped:
            line += (
                f", {len(self.skipped)} unverifiable "
                "(the profile carries nothing that could decide them)"
            )
        return line


def check(profile: Profile, expectations: Iterable[Expectation]) -> SuiteResult:
    """Run expectations against a profile."""
    result = SuiteResult(dataset=profile.dataset)
    for expectation in expectations:
        outcome = _evaluate(expectation, profile)
        if isinstance(outcome, _Skipped):
            result.skipped.append(expectation)
            continue
        result.checked += 1
        if outcome is not None:
            result.findings.append(outcome)
    return result


def run(suite: Suite, profile: Profile) -> SuiteResult:
    """Run a whole suite. The named form of `check`."""
    return check(profile, suite.expectations)


# -- generation ----------------------------------------------------------------


def learn(
    profile: Profile,
    *,
    margin: float = DEFAULT_MARGIN,
    include_ranges: bool = True,
    include_nulls: bool = True,
) -> Suite:
    """Generate a suite from an observed profile.

    Every generated expectation is marked `learned`, and every bound is widened by
    `margin` past what was seen. This is a starting point for a person to prune, not
    a claim about what the data should be — a distinction the flag preserves so the
    two never blur together in review.
    """
    suite = Suite(dataset=profile.dataset, name=f"learned:{profile.dataset.name}")
    suite.expectations.append(
        Expectation(
            "schema_matches",
            None,
            {"columns": sorted(profile.column_names)},
            Severity.ERROR,
            learned=True,
            note="columns present when the suite was generated",
        )
    )
    if profile.row_count > 0:
        span = max(1, int(profile.row_count * margin))
        suite.expectations.append(
            Expectation(
                "row_count_between",
                None,
                {"low": max(0, profile.row_count - span * 5), "high": profile.row_count + span * 5},
                Severity.WARN,
                learned=True,
                note=f"observed {profile.row_count} rows",
            )
        )

    for column in profile.columns:
        suite.expectations.append(
            Expectation(
                "dtype_is",
                column.name,
                {"dtype": column.dtype},
                Severity.ERROR,
                learned=True,
            )
        )
        if include_nulls and column.null_rate is not None:
            ceiling = min(1.0, round(column.null_rate + max(margin, 0.02), 4))
            suite.expectations.append(
                Expectation(
                    "null_rate_below",
                    column.name,
                    {"max_rate": ceiling},
                    Severity.WARN,
                    learned=True,
                    note=f"observed {column.null_rate:.1%}",
                )
            )
        if include_ranges:
            low, high = _as_float(column.min), _as_float(column.max)
            if low is not None and high is not None and high >= low:
                pad = max(abs(high - low) * margin, abs(high) * margin, 1e-9)
                suite.expectations.append(
                    Expectation(
                        "in_range",
                        column.name,
                        {"low": round(low - pad, 6), "high": round(high + pad, 6)},
                        Severity.WARN,
                        learned=True,
                        note=f"observed [{low}, {high}]",
                    )
                )
    return suite


def merge(*suites: Suite) -> Suite:
    """Combine suites for the same dataset, keeping asserted expectations over learned ones.

    The upgrade path: run `learn` again next month and merge, and anything a person
    wrote survives the regeneration.
    """
    if not suites:
        raise ValueError("merge needs at least one suite")
    out = Suite(dataset=suites[0].dataset, name=suites[0].name)
    seen: dict[tuple[str, str | None], Expectation] = {}
    for suite in suites:
        for expectation in suite.expectations:
            key = (expectation.kind, expectation.column)
            prior = seen.get(key)
            if prior is None or (prior.learned and not expectation.learned):
                seen[key] = expectation
    out.expectations = [seen[key] for key in sorted(seen, key=lambda k: (k[1] or "", k[0]))]
    return out


def to_dict(suite: Suite) -> dict[str, Any]:
    """Serialize a suite, so it can live in the repository beside the models."""
    return {
        "dataset": str(suite.dataset),
        "name": suite.name,
        "expectations": [
            {
                "kind": e.kind,
                "column": e.column,
                "params": e.params,
                "severity": e.severity.value,
                "learned": e.learned,
                "note": e.note,
            }
            for e in suite.expectations
        ],
    }


def from_dict(blob: dict[str, Any], dataset: DatasetId) -> Suite:
    """Rebuild a suite written by `to_dict`."""
    suite = Suite(dataset=dataset, name=blob.get("name", ""))
    for entry in blob.get("expectations", ()):
        suite.expectations.append(
            Expectation(
                kind=entry["kind"],
                column=entry.get("column"),
                params=dict(entry.get("params", {})),
                severity=Severity(entry.get("severity", "error")),
                learned=bool(entry.get("learned", False)),
                note=entry.get("note", ""),
            )
        )
    return suite
