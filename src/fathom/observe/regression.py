"""Did the rewrite change the numbers?

Someone tunes a query for performance. Someone migrates a model from one engine to
another. Someone refactors three CTEs into one. The pipeline succeeds, the tests pass,
and the output is subtly different — a join hint changed the row order feeding a
`LIMIT`, a cast lost a decimal place, a `DISTINCT` disappeared in the tidy-up.

Nothing catches it. `diff` compares the *graph*, and the graph is identical. `drift`
compares against yesterday, and yesterday is now the new version. Shadow mode compares
the planner's *decisions*, not a transform's output. The one comparison nobody has is
the one that matters: this build against the build the old code would have produced.

**What this does not and cannot say.** A refactor that fixes a bug is *supposed* to
change the output. So nothing here reports "wrong" — it reports **changed**, per
partition, and the reviewer declares which changes were intended. A tool that decided
for itself which differences were bugs would be wrong in both directions, and the
expensive direction is the one where it stays quiet.

**Coverage is the whole game.** Comparing three partitions out of four hundred and
reporting "no changes" is the failure this module has to avoid, so a report carries
the count it compared, refuses to call itself clean below a floor, and names the
partitions it could not compare at all. Fingerprints prove a change happened; nothing
proves a change did not happen in a partition nobody fingerprinted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeGuard

from ..core.types import DatasetId, KeyPredicate
from .profile import Finding, Profile, Severity
from .schema import diff_schemas

__all__ = [
    "Changed",
    "RegressionReport",
    "changed_partitions",
    "compare_fingerprints",
    "compare_outputs",
    "explain",
    "is_regression",
    "only_in",
    "summarize",
    "unexplained",
    "worst",
]

# Below this fraction of the expected partitions compared, a report will not call
# itself clean. A refactor check over 1% of a table proves nothing about the other 99%.
MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class Changed:
    """One partition whose contents differ between the two builds."""

    dataset: DatasetId
    partition: KeyPredicate
    before: str
    after: str
    findings: tuple[Finding, ...] = ()  # how it differs, when profiles were supplied

    @property
    def is_explained(self) -> bool:
        """True when profiles were supplied and said something about the difference."""
        return bool(self.findings)

    def __str__(self) -> str:
        head = f"{self.dataset} {self.partition}: contents differ"
        if not self.findings:
            return head + " (no profiles supplied, so how is unknown)"
        return head + " — " + "; ".join(f.detail for f in self.findings)


def compare_fingerprints(
    before: Mapping[KeyPredicate, str], after: Mapping[KeyPredicate, str]
) -> list[KeyPredicate]:
    """Partitions present in both whose content hash differs.

    Partitions missing from either side are *not* reported here — an absent partition
    is a different fact from a changed one, and `only_in` keeps them apart so a
    reviewer is not told a table shrank when a partition simply was not built yet.
    """
    shared = set(before) & set(after)
    return sorted((k for k in shared if before[k] != after[k]), key=str)


def only_in(
    before: Mapping[KeyPredicate, str], after: Mapping[KeyPredicate, str]
) -> tuple[list[KeyPredicate], list[KeyPredicate]]:
    """Partitions on one side only, as `(gone, new)`.

    Both are worth a reviewer's attention and neither is a content change. A partition
    that vanished may be a genuine regression or an unbuilt slice; this reports the
    fact and declines to guess which.
    """
    gone = sorted(set(before) - set(after), key=str)
    new = sorted(set(after) - set(before), key=str)
    return gone, new


def explain(before: Profile, after: Profile, *, tolerance: float = 0.0) -> list[Finding]:
    """How two profiles of the same partition differ.

    A digest difference says *that* something changed and nothing about *what*, which
    is not actionable. This turns it into row counts, null rates, and ranges — the
    three that account for most refactor regressions.

    `tolerance` is a relative allowance on numeric moves, for a refactor that changes
    a floating-point summation order and nothing else.
    """
    findings: list[Finding] = []

    if before.row_count != after.row_count:
        moved_by = _relative(before.row_count, after.row_count)
        if moved_by is None or abs(moved_by) > tolerance:
            findings.append(
                Finding(
                    column=None,
                    kind="regression_row_count",
                    severity=Severity.ERROR,
                    detail=f"row count moved from {before.row_count} to {after.row_count}",
                    before=before.row_count,
                    after=after.row_count,
                )
            )

    findings.extend(
        Finding(
            column=change.column,
            kind="regression_schema",
            severity=Severity.ERROR,
            detail=str(change),
            before=change.before,
            after=change.after,
        )
        for change in diff_schemas(before, after).breaking
    )

    for column in after.columns:
        was = before.column(column.name)
        if was is None:
            continue
        if was.null_rate is not None and column.null_rate is not None:
            moved = column.null_rate - was.null_rate
            if abs(moved) > max(tolerance, 1e-9):
                findings.append(
                    Finding(
                        column=column.name,
                        kind="regression_null_rate",
                        severity=Severity.ERROR,
                        detail=(
                            f"{column.name} null rate moved from {was.null_rate:.2%} "
                            f"to {column.null_rate:.2%}"
                        ),
                        before=round(was.null_rate, 6),
                        after=round(column.null_rate, 6),
                    )
                )
        for name, old, new in (
            ("min", was.min, column.min),
            ("max", was.max, column.max),
        ):
            if _numeric(old) and _numeric(new):
                moved_by = _relative(float(old), float(new))
                if moved_by is not None and abs(moved_by) > tolerance:
                    findings.append(
                        Finding(
                            column=column.name,
                            kind=f"regression_{name}",
                            severity=Severity.WARN,
                            detail=f"{column.name} {name} moved from {old} to {new}",
                            before=old,
                            after=new,
                        )
                    )
    return findings


def _numeric(value: object) -> TypeGuard[float]:
    """A narrowing check, so callers can use the value without a second assertion.

    `bool` is excluded deliberately: `True` is an `int` in Python, and reporting that
    a flag column's max "moved from False to True" as a numeric range change is noise
    dressed as a finding.
    """
    return isinstance(value, int | float) and not isinstance(value, bool)


def _relative(before: float, after: float) -> float | None:
    """Signed relative change, or `None` when the baseline is zero.

    Zero to anything is not an infinite regression; it is a change with no meaningful
    ratio, and the caller reports it on the absolute difference instead.
    """
    if before == 0:
        return None
    return (after - before) / before


@dataclass
class RegressionReport:
    """What a rewrite did to a dataset's output."""

    dataset: DatasetId
    compared: int = 0
    expected: int = 0
    changed: list[Changed] = field(default_factory=list)
    gone: list[KeyPredicate] = field(default_factory=list)
    new: list[KeyPredicate] = field(default_factory=list)
    accepted: list[KeyPredicate] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of the expected partitions that were actually compared."""
        if self.expected <= 0:
            return 0.0
        return min(1.0, self.compared / self.expected)

    @property
    def is_conclusive(self) -> bool:
        """True when enough was compared for a clean result to mean anything."""
        return self.compared > 0 and self.coverage >= MIN_COVERAGE

    @property
    def is_clean(self) -> bool:
        """No unaccepted change, no partition gained or lost, and enough compared.

        The coverage term is what stops this reading as a pass after comparing three
        partitions out of four hundred.
        """
        return self.is_conclusive and not self.changed and not self.gone and not self.new

    def summary(self) -> str:
        lines = [
            f"{self.dataset}: compared {self.compared} of {self.expected} partition(s) "
            f"({self.coverage:.0%})"
        ]
        if self.changed:
            lines.append(f"    {len(self.changed)} partition(s) changed:")
            lines.extend(f"        {c}" for c in self.changed[:10])
            if len(self.changed) > 10:
                lines.append(f"        +{len(self.changed) - 10} more")
        if self.gone:
            lines.append(f"    {len(self.gone)} partition(s) present before and not after")
        if self.new:
            lines.append(f"    {len(self.new)} partition(s) present after and not before")
        if self.accepted:
            lines.append(f"    {len(self.accepted)} change(s) accepted as intended")
        if not self.is_conclusive:
            lines.append(
                f"    NOT CONCLUSIVE — below {MIN_COVERAGE:.0%} coverage. A clean result "
                "here says nothing about the partitions nobody fingerprinted."
            )
        elif self.is_clean:
            lines.append("    no change detected in the partitions compared")
        lines.append(
            "    a changed partition is not a wrong one: a rewrite that fixes a bug is "
            "supposed to change the output, and deciding which is which is not this "
            "tool's call"
        )
        return "\n".join(lines)


def compare_outputs(
    dataset: DatasetId,
    before: Mapping[KeyPredicate, str],
    after: Mapping[KeyPredicate, str],
    *,
    profiles_before: Mapping[KeyPredicate, Profile] | None = None,
    profiles_after: Mapping[KeyPredicate, Profile] | None = None,
    expected: int | None = None,
    intended: Iterable[KeyPredicate] = (),
    tolerance: float = 0.0,
) -> RegressionReport:
    """Compare a rewritten transform's output against its predecessor's.

    `expected` is how many partitions the dataset has, so coverage can be reported
    honestly. Absent, it defaults to the number of partitions either side knew about,
    which is the best available answer and still an under-count if both builds skipped
    the same slice.

    `intended` lists partitions a reviewer has already accepted as legitimately
    changed. They are counted and reported, not hidden — an accepted change is still a
    change, and a review that cannot see how many were waved through is not a review.
    """
    accepted = set(intended)
    shared = set(before) & set(after)
    gone, new = only_in(before, after)

    report = RegressionReport(
        dataset=dataset,
        compared=len(shared),
        expected=expected if expected is not None else len(set(before) | set(after)),
        gone=gone,
        new=new,
    )

    for key in compare_fingerprints(before, after):
        if key in accepted:
            report.accepted.append(key)
            continue
        findings: tuple[Finding, ...] = ()
        if profiles_before and profiles_after:
            was, is_now = profiles_before.get(key), profiles_after.get(key)
            if was is not None and is_now is not None:
                findings = tuple(explain(was, is_now, tolerance=tolerance))
        report.changed.append(
            Changed(
                dataset=dataset,
                partition=key,
                before=before[key],
                after=after[key],
                findings=findings,
            )
        )
    return report


def changed_partitions(report: RegressionReport) -> list[KeyPredicate]:
    """Just the keys that changed, for feeding back into a plan."""
    return [c.partition for c in report.changed]


def is_regression(report: RegressionReport) -> bool:
    """True when anything unaccepted changed, or a partition was gained or lost.

    Named for what a merge gate should block on. It is deliberately *not* the negation
    of `is_clean`: an inconclusive report is neither a regression nor a pass, and
    collapsing the two would let a refactor through on the strength of having compared
    almost nothing.
    """
    return bool(report.changed or report.gone or report.new)


def unexplained(report: RegressionReport) -> list[Changed]:
    """Changed partitions with no profile evidence for how they differ.

    A digest mismatch with nothing behind it is the least actionable finding a review
    can receive, and the fix is to supply profiles rather than to ignore it.
    """
    return [c for c in report.changed if not c.is_explained]


def worst(report: RegressionReport, *, limit: int = 10) -> list[Changed]:
    """Changed partitions with the most findings, which is where to look first."""
    return sorted(report.changed, key=lambda c: (-len(c.findings), str(c.partition)))[:limit]


def summarize(reports: Sequence[RegressionReport]) -> str:
    """One line per dataset across a whole refactor, worst first."""
    if not reports:
        return "nothing compared"
    ranked = sorted(reports, key=lambda r: (-len(r.changed), str(r.dataset)))
    lines = [f"{len(reports)} dataset(s) compared"]
    for report in ranked:
        state = (
            "INCONCLUSIVE"
            if not report.is_conclusive
            else ("clean" if report.is_clean else f"{len(report.changed)} changed")
        )
        lines.append(f"    {report.dataset}: {state} ({report.coverage:.0%} covered)")
    inconclusive = [r for r in reports if not r.is_conclusive]
    if inconclusive:
        lines.append(f"    {len(inconclusive)} dataset(s) compared too little to conclude anything")
    return "\n".join(lines)
