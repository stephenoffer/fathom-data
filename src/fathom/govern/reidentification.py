"""Columns that identify nobody alone and everybody together.

`policy` finds direct identifiers: a column named `email` holding string values that
look like addresses. That catches the easy half. The hard half is that a birth date
identifies nobody, a postcode identifies nobody, and a gender identifies nobody,
while the three together identify most of a population. A dataset can pass every
direct-identifier check and still be personal data, and nothing in a per-column
label can see it, because the property is not a property of any column.

The classic result is Sweeney's: date of birth, five-digit ZIP, and sex uniquely
identify around 87% of the United States. Every governance stack that labels columns
one at a time is blind to exactly this, which is why a "de-identified" export is the
most common source of an accidental disclosure.

**Which direction this errs, and why.** It over-reports risk and never under-reports
it — the mirror of the planner, and for the same reason: the expensive failure is
one-directional. A false alarm costs a review. A missed one is a disclosure that
cannot be recalled.

That commitment is what makes the arithmetic honest. From per-column profiles we
know each quasi-identifier's distinct count `dᵢ` and the row count `N`, but not the
number of distinct *combinations* `C`, which would need a scan. What we do know is
`C ≥ max(dᵢ)`, and therefore that the average group size is at most `N / max(dᵢ)`.

So this module can **prove a dataset is risky** and can never prove one is safe. If
the bound comes back below the threshold, the average group is genuinely too small.
If it comes back above, the minimum group may still be one person hiding under a
comfortable average. `assess` says exactly that rather than returning a clean bill,
and `RiskReport.is_clear` is named for the absence of proven risk, not for safety.

Cross-dataset linkage is the same argument one level up. Two exports that are each
defensible become identifying when a common ancestor means their rows can be joined,
and neither export's own review can see the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..core.types import ColumnRef, DatasetId
from ..graph.model import Graph
from ..graph.query import common_ancestors
from ..observe.profile import Profile, Severity
from .policy import LabelSet

__all__ = [
    "DIRECT_IDENTIFIERS",
    "QUASI_IDENTIFIERS",
    "LinkageRisk",
    "RiskFinding",
    "RiskReport",
    "assess",
    "direct_identifiers",
    "k_upper_bound",
    "linkable_columns",
    "linkage_risks",
    "quasi_identifier_kinds",
    "quasi_identifiers",
    "risky_datasets",
    "singling_out",
]

# Labels that identify a person on their own. `policy` already implies `pii` from
# most of these; they are repeated here because this module needs to tell a direct
# identifier from a quasi one, which `pii` alone does not distinguish.
DIRECT_IDENTIFIERS = frozenset({"email", "phone", "national_id", "person_name", "user_identifier"})

# Labels that identify nobody alone and a great many people in combination.
QUASI_IDENTIFIERS = frozenset(
    {"date_of_birth", "postal_address", "latitude", "longitude", "ip_address", "device_id"}
)

# Below this average group size a dataset is reported as re-identifiable. Five is the
# common regulatory floor for published statistics; it is a parameter everywhere it is
# used because the right value is a policy decision, not a mathematical one.
DEFAULT_K = 5

# A column whose distinct count reaches this fraction of the row count is behaving as
# an identifier whatever it is called — a salted hash, an order reference, a session id.
SINGLING_OUT_RATIO = 0.9


@dataclass(frozen=True)
class RiskFinding:
    """One proven re-identification risk within a single dataset."""

    dataset: DatasetId
    columns: tuple[str, ...]
    kind: str  # quasi_identifier_set | singling_out | direct_identifier
    severity: Severity
    detail: str
    k_upper: float | None = None

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.dataset}: {self.detail}"


@dataclass(frozen=True)
class LinkageRisk:
    """Two datasets that are each defensible and jointly identifying."""

    left: DatasetId
    right: DatasetId
    via: tuple[DatasetId, ...]  # common ancestors making the join possible
    combined: tuple[str, ...]  # the union of quasi-identifier labels across both
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass
class RiskReport:
    """What could be proven about one dataset, and what could not."""

    dataset: DatasetId
    findings: list[RiskFinding] = field(default_factory=list)
    quasi: list[str] = field(default_factory=list)
    unmeasurable: list[str] = field(default_factory=list)
    k_threshold: int = DEFAULT_K

    @property
    def is_clear(self) -> bool:
        """No risk was *proven*. Deliberately not named `is_safe`.

        Proving safety would need the distinct count of the quasi-identifier
        combination, which is a scan this module does not do.
        """
        return not self.findings

    def summary(self) -> str:
        """The report as text, with what could not be measured stated."""
        lines: list[str] = []
        if self.is_clear:
            lines.append(f"{self.dataset}: no re-identification risk proven")
        else:
            lines.append(
                f"{self.dataset}: {len(self.findings)} re-identification risk(s) proven "
                f"at k={self.k_threshold}"
            )
            lines.extend(f"    {f}" for f in self.findings)
        if self.quasi:
            lines.append(f"    quasi-identifiers present: {', '.join(sorted(self.quasi))}")
        if self.unmeasurable:
            lines.append(
                f"    not measurable, no distinct count profiled: "
                f"{', '.join(sorted(self.unmeasurable))}"
            )
        lines.append(
            "    a clear result means no risk was proven, not that the data is "
            "safe — the minimum group size needs a scan this does not do"
        )
        return "\n".join(lines)


def _labels_of(labels: LabelSet, dataset: DatasetId, column: str) -> set[str]:
    return {label.name for label in labels.get(ColumnRef(dataset, column), set())}


def quasi_identifiers(profile: Profile, labels: LabelSet) -> list[str]:
    """Columns of `profile` carrying a quasi-identifier label, in profile order."""
    return [
        column.name
        for column in profile.columns
        if _labels_of(labels, profile.dataset, column.name) & QUASI_IDENTIFIERS
    ]


def direct_identifiers(profile: Profile, labels: LabelSet) -> list[str]:
    """Columns carrying a label that identifies a person on its own."""
    return [
        column.name
        for column in profile.columns
        if _labels_of(labels, profile.dataset, column.name) & DIRECT_IDENTIFIERS
    ]


def k_upper_bound(profile: Profile, columns: Sequence[str]) -> float | None:
    """Upper bound on the average group size over `columns`.

    `C ≥ max(dᵢ)` for the distinct count of any single column, so the average group
    size `N / C` is at most `N / max(dᵢ)`. Returns `None` when no column in the set
    carries a distinct estimate, which is a refusal to answer rather than a zero.

    An upper bound is the useful direction: it can show a group is too small, which
    is a proof of risk, and it can never show a group is large enough, which would be
    a claim of safety this cannot support.
    """
    if profile.row_count <= 0:
        return None
    widest = 0
    for name in columns:
        column = profile.column(name)
        if column is not None and column.distinct_estimate:
            widest = max(widest, column.distinct_estimate)
    if widest <= 0:
        return None
    return profile.row_count / widest


def singling_out(profile: Profile, *, ratio: float = SINGLING_OUT_RATIO) -> list[str]:
    """Columns near-unique across the rows, which single a person out whatever they are called.

    A salted hash carries no identifying label and identifies perfectly. So does an
    order reference, once joined to anything.
    """
    if profile.row_count <= 0:
        return []
    out: list[str] = []
    for column in profile.columns:
        if column.distinct_estimate and column.distinct_estimate / profile.row_count >= ratio:
            out.append(column.name)
    return out


def assess(
    profile: Profile,
    labels: LabelSet,
    *,
    k_threshold: int = DEFAULT_K,
    ratio: float = SINGLING_OUT_RATIO,
) -> RiskReport:
    """What can be proven about one dataset's re-identifiability.

    Three independent checks, reported separately because they are remediated
    differently: a direct identifier is dropped, a near-unique column is hashed or
    coarsened, and a quasi-identifier set is generalized.
    """
    quasi = quasi_identifiers(profile, labels)
    report = RiskReport(dataset=profile.dataset, quasi=quasi, k_threshold=k_threshold)

    for name in direct_identifiers(profile, labels):
        report.findings.append(
            RiskFinding(
                dataset=profile.dataset,
                columns=(name,),
                kind="direct_identifier",
                severity=Severity.ERROR,
                detail=f"{name} identifies a person on its own",
            )
        )

    for name in singling_out(profile, ratio=ratio):
        if name in report.quasi or any(name in f.columns for f in report.findings):
            continue
        column = profile.column(name)
        assert column is not None and column.distinct_estimate
        report.findings.append(
            RiskFinding(
                dataset=profile.dataset,
                columns=(name,),
                kind="singling_out",
                severity=Severity.WARN,
                detail=(
                    f"{name} has {column.distinct_estimate} distinct values across "
                    f"{profile.row_count} rows, so it singles a row out whatever it is named"
                ),
            )
        )

    if len(quasi) >= 2:
        bound = k_upper_bound(profile, quasi)
        if bound is None:
            report.unmeasurable.extend(quasi)
        elif bound < k_threshold:
            report.findings.append(
                RiskFinding(
                    dataset=profile.dataset,
                    columns=tuple(quasi),
                    kind="quasi_identifier_set",
                    severity=Severity.ERROR,
                    detail=(
                        f"{', '.join(quasi)} together give an average group of at most "
                        f"{bound:.2f} rows, below k={k_threshold}"
                    ),
                    k_upper=bound,
                )
            )
    return report


# -- across datasets -----------------------------------------------------------


def linkable_columns(left: Profile, right: Profile) -> list[str]:
    """Column names present in both profiles, which is what makes a join possible."""
    return sorted(set(left.column_names) & set(right.column_names))


def quasi_identifier_kinds(profile: Profile, labels: LabelSet) -> set[str]:
    """The *kinds* of quasi-identifier a dataset carries, as labels rather than columns.

    Linkage is about kinds, not names. Two datasets whose birth-date columns are called
    `dob` and `birth_date` carry the same attribute, and joining them adds nothing a
    reviewer needs to know about — whereas a birth date joined to a postcode does.
    Comparing column names would get both cases backwards.
    """
    found: set[str] = set()
    for column in profile.columns:
        found |= _labels_of(labels, profile.dataset, column.name) & QUASI_IDENTIFIERS
    return found


def linkage_risks(
    graph: Graph,
    profiles: Mapping[DatasetId, Profile],
    labels: LabelSet,
    *,
    k_threshold: int = DEFAULT_K,
) -> list[LinkageRisk]:
    """Pairs of datasets that are each defensible and jointly identifying.

    A pair qualifies when three things hold: they share an ancestor, so their rows can
    be joined; they share at least one column to join on; and the union of the quasi-
    identifier *kinds* they carry is larger than either one's own. Neither dataset's own review
    can see this, because the risk is not in either of them.
    """
    datasets = sorted(profiles)
    out: list[LinkageRisk] = []

    for index, left in enumerate(datasets):
        for right in datasets[index + 1 :]:
            shared = common_ancestors(graph, left, right)
            if not shared:
                continue
            joinable = linkable_columns(profiles[left], profiles[right])
            if not joinable:
                continue

            left_quasi = quasi_identifier_kinds(profiles[left], labels)
            right_quasi = quasi_identifier_kinds(profiles[right], labels)
            combined = left_quasi | right_quasi
            # Only interesting when joining actually adds something neither had alone.
            if (
                not left_quasi
                or not right_quasi
                or len(combined) <= max(len(left_quasi), len(right_quasi))
            ):
                continue

            out.append(
                LinkageRisk(
                    left=left,
                    right=right,
                    via=tuple(shared),
                    combined=tuple(sorted(combined)),
                    detail=(
                        f"{left} and {right} share {', '.join(joinable)} and an ancestor, "
                        f"so joining them yields {len(combined)} quasi-identifiers "
                        f"({', '.join(sorted(combined))}) where neither alone has more than "
                        f"{max(len(left_quasi), len(right_quasi))}"
                    ),
                )
            )
    return out


def risky_datasets(
    reports: Iterable[RiskReport],
) -> list[DatasetId]:
    """Datasets with at least one proven risk, worst first by finding count."""
    flagged = [r for r in reports if not r.is_clear]
    return [r.dataset for r in sorted(flagged, key=lambda r: (-len(r.findings), str(r.dataset)))]
