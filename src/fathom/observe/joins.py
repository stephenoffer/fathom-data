"""Join keys, and the two ways one quietly ruins a table.

A join is the only operation in a warehouse that can make a table *larger* than its
inputs, and the only one that can silently drop rows. Both failures come from the same
place — the shape of the key changed — and neither raises:

- **Fan-out.** A key that was unique stops being unique. Every downstream row is
  duplicated, `SUM(revenue)` doubles, and the pipeline succeeds. Somebody notices in
  the monthly close.
- **Orphaning.** Keys on one side stop matching the other, usually after a format
  change or a late-arriving dimension. An inner join drops them. The table is smaller
  and nothing says why.

`drift` sees the *symptom* — row count moved, distinct count moved — and reports it as
one finding per column, which is how a real join failure arrives as thirty alerts with
no cause. This module names the cause, from the same profiles, at no extra cost.

**What is soundly computable without a scan, and what is not.** From a profile we know
a column's `distinct_estimate` and the table's `row_count`. Their ratio is the average
fan-out of that key, and a change in it is real evidence about that one dataset —
nothing inferred, nothing assumed.

What we cannot know without scanning is the *overlap* between two key sets. So this
module never estimates how many rows a join will emit. It does one thing across two
datasets, and only in the direction that is provable: if the left side has more
distinct keys than the right side has in total, then at least the difference **cannot**
match, whatever the overlap is. That is a lower bound on orphaned keys — it can prove
rows will be dropped and can never prove none will be. The same asymmetry
`reidentification` runs on, for the same reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..core.types import DatasetId
from .profile import Finding, Profile, Severity

__all__ = [
    "JoinRisk",
    "KeyShape",
    "amplification",
    "candidate_keys",
    "fan_out",
    "is_unique_key",
    "join_risks",
    "orphan_floor",
    "risky_keys",
    "shape_of",
    "shape_drift",
    "uniqueness_lost",
]

# Fan-out below this counts as effectively unique. Not exactly 1.0, because a
# `distinct_estimate` from a sketch is approximate and a hair over 1.0 on a genuinely
# unique key is a rounding artifact rather than a duplicate.
UNIQUE_TOLERANCE = 1.02

# Relative change in average fan-out worth reporting. A key going from 1.0 to 1.05 is
# noise; 1.0 to 2.0 has doubled every downstream row.
DEFAULT_FAN_OUT_TOLERANCE = 0.25


@dataclass(frozen=True)
class KeyShape:
    """What one profile says about one candidate join key."""

    dataset: DatasetId
    column: str
    rows: int
    distinct: int

    @property
    def fan_out(self) -> float:
        """Average rows per distinct key. 1.0 means the key is unique."""
        return self.rows / self.distinct if self.distinct else 0.0

    @property
    def is_unique(self) -> bool:
        """True when the key identifies at most one row, within sketch tolerance."""
        return 0 < self.fan_out <= UNIQUE_TOLERANCE

    def __str__(self) -> str:
        return (
            f"{self.dataset}#{self.column}: {self.distinct} distinct over {self.rows} rows "
            f"(fan-out {self.fan_out:.2f})"
        )


def shape_of(profile: Profile, column: str) -> KeyShape | None:
    """The shape of one key, or `None` when the profile cannot say.

    Returns `None` rather than a zero when the distinct count is missing, because
    "not profiled" and "no distinct values" would otherwise be the same answer and
    only one of them is a finding.
    """
    found = profile.column(column)
    if found is None or not found.distinct_estimate or profile.row_count <= 0:
        return None
    return KeyShape(
        dataset=profile.dataset,
        column=column,
        rows=profile.row_count,
        distinct=found.distinct_estimate,
    )


def fan_out(profile: Profile, column: str) -> float | None:
    """Average rows per distinct value of `column`, or `None` if unmeasurable."""
    shape = shape_of(profile, column)
    return shape.fan_out if shape else None


def is_unique_key(profile: Profile, column: str) -> bool:
    """True when a column is behaving as a unique key in this profile."""
    shape = shape_of(profile, column)
    return bool(shape and shape.is_unique)


def uniqueness_lost(before: Profile, after: Profile, column: str) -> bool:
    """True when a key that was unique no longer is.

    The single most common cause of a join silently doubling a revenue total, and the
    one worth its own predicate because the remedy is different: a key that lost
    uniqueness needs a deduplication upstream, not a wider tolerance here.
    """
    was, is_now = shape_of(before, column), shape_of(after, column)
    return bool(was and is_now and was.is_unique and not is_now.is_unique)


def shape_drift(
    before: Profile,
    after: Profile,
    keys: Sequence[str],
    *,
    tolerance: float = DEFAULT_FAN_OUT_TOLERANCE,
    severity: Severity = Severity.ERROR,
) -> list[Finding]:
    """Keys whose fan-out moved by more than `tolerance`, relatively.

    Reported as `join_key_fan_out` rather than as a generic distinct-count change,
    because the two have different remedies and a reader needs to tell them apart:
    a distinct count that fell is a data question, and a fan-out that rose is a join
    about to duplicate everything downstream.
    """
    findings: list[Finding] = []
    for column in keys:
        was, is_now = shape_of(before, column), shape_of(after, column)
        if was is None or is_now is None or was.fan_out <= 0:
            continue
        change = (is_now.fan_out - was.fan_out) / was.fan_out
        if abs(change) < tolerance:
            continue

        direction = "rose" if change > 0 else "fell"
        detail = f"{column} fan-out {direction} from {was.fan_out:.2f} to {is_now.fan_out:.2f}"
        if was.is_unique and not is_now.is_unique:
            detail += " — this key was unique and is not any more, so every join on it "
            detail += "now duplicates rows"
        findings.append(
            Finding(
                column=column,
                kind="join_key_fan_out",
                severity=severity if change > 0 else Severity.WARN,
                detail=detail,
                before=round(was.fan_out, 4),
                after=round(is_now.fan_out, 4),
            )
        )
    return findings


def orphan_floor(left: Profile, right: Profile, column: str) -> int | None:
    """Lower bound on left-side keys that cannot possibly match the right side.

    If the left has more distinct keys than the right has *in total*, the difference
    cannot match whatever the overlap is. That is provable without a scan.

    The direction is one-way and deliberate: this can prove rows will be dropped and
    can never prove none will be. A floor of zero means "no loss proven", not "no loss"
    — the two key sets may be the same size and entirely disjoint.

    Returns `None` when either side lacks a distinct count.
    """
    a, b = shape_of(left, column), shape_of(right, column)
    if a is None or b is None:
        return None
    return max(0, a.distinct - b.distinct)


def amplification(inputs: Iterable[Profile], output: Profile) -> float | None:
    """Output rows over the largest input's rows.

    Above 1.0 means the join emitted more rows than any single input held, which is
    the arithmetic signature of fan-out. Not proof — a legitimate cross join or an
    unnest does the same thing — which is why this returns a number rather than a
    verdict.
    """
    largest = max((p.row_count for p in inputs), default=0)
    if largest <= 0:
        return None
    return output.row_count / largest


@dataclass
class JoinRisk:
    """What can be proven about one join, from profiles alone."""

    output: DatasetId
    column: str
    findings: list[Finding] = field(default_factory=list)
    orphans: int = 0
    amplification: float | None = None
    unmeasurable: list[str] = field(default_factory=list)

    @property
    def is_clear(self) -> bool:
        """No risk proven. Not the same as safe — overlap needs a scan."""
        return not self.findings and self.orphans == 0

    def summary(self) -> str:
        lines: list[str] = []
        if self.is_clear:
            lines.append(f"{self.output} on {self.column}: no join risk proven")
        else:
            lines.append(f"{self.output} on {self.column}: {len(self.findings)} finding(s)")
            lines.extend(f"    {f}" for f in self.findings)
        if self.orphans:
            lines.append(
                f"    at least {self.orphans} key(s) cannot match — that many rows will "
                "be dropped by an inner join, whatever the overlap is"
            )
        if self.amplification is not None and self.amplification > 1.0:
            lines.append(
                f"    output holds {self.amplification:.2f}x the rows of its largest input"
            )
        if self.unmeasurable:
            lines.append(
                f"    not measurable, no distinct count profiled: "
                f"{', '.join(sorted(self.unmeasurable))}"
            )
        lines.append(
            "    a clear result means no risk was proven; key overlap needs a scan this does not do"
        )
        return "\n".join(lines)


def join_risks(
    left: Profile,
    right: Profile,
    output: Profile,
    column: str,
    *,
    previous: Profile | None = None,
    tolerance: float = DEFAULT_FAN_OUT_TOLERANCE,
) -> JoinRisk:
    """Everything provable about one join on one key.

    `previous` is an earlier profile of `output`, used to detect that its key shape
    moved. Without it the check is limited to what one moment shows, which still
    catches the two cases that matter: a non-unique key on a side expected to be
    unique, and keys that provably cannot match.
    """
    risk = JoinRisk(output=output.dataset, column=column)

    for side in (left, right):
        if shape_of(side, column) is None:
            risk.unmeasurable.append(f"{side.dataset}#{column}")

    floor = orphan_floor(left, right, column)
    if floor:
        risk.orphans = floor

    risk.amplification = amplification([left, right], output)
    if risk.amplification is not None and risk.amplification > 1.0:
        risk.findings.append(
            Finding(
                column=column,
                kind="join_amplification",
                severity=Severity.WARN,
                detail=(
                    f"{output.dataset} holds {risk.amplification:.2f}x the rows of its "
                    "largest input, the arithmetic signature of fan-out"
                ),
                before=1.0,
                after=round(risk.amplification, 4),
            )
        )

    for side, label in ((left, "left"), (right, "right")):
        shape = shape_of(side, column)
        if shape is not None and not shape.is_unique:
            risk.findings.append(
                Finding(
                    column=column,
                    kind="join_key_not_unique",
                    severity=Severity.WARN,
                    detail=(
                        f"{label} side {side.dataset} has fan-out {shape.fan_out:.2f} on "
                        f"{column}, so each matching key multiplies the output"
                    ),
                    before=1.0,
                    after=round(shape.fan_out, 4),
                )
            )

    if previous is not None:
        risk.findings.extend(shape_drift(previous, output, [column], tolerance=tolerance))

    return risk


def risky_keys(risks: Sequence[JoinRisk]) -> list[JoinRisk]:
    """Only the joins with something proven, worst first by finding count."""
    flagged = [r for r in risks if not r.is_clear]
    return sorted(
        flagged, key=lambda r: (-len(r.findings) - (1 if r.orphans else 0), str(r.output))
    )


def candidate_keys(profile: Profile) -> list[str]:
    """Columns behaving as unique keys, which are the ones a join is likely on.

    A heuristic for suggesting what to check, never for deciding what to join. The
    graph's column edges are the real answer where they exist.
    """
    return [c.name for c in profile.columns if is_unique_key(profile, c.name)]
