"""A promise one team makes to another about a dataset.

Everything needed to enforce a data contract already exists in this library and none
of it is bound to a promise. `observe.quality` checks a suite but nobody says whose
suite it is. `observe.schema` finds a breaking change but not who it breaks.
`observe.freshness` measures age against an SLA the producer never agreed to.
`graph.diff` gates a narrowing without knowing which consumer the narrowing hurts.

So a contract lives in a wiki, the producer never reads it, and it is discovered to
have been violated by the consumer, in production, on a Sunday.

A `Contract` is the missing object: one producer, named consumers, and the promises
that bind them. It adds no checking machinery — `verify` dispatches to the modules
above and collects what they say. What it adds is *attribution*: a violation names
who promised what to whom, which is the difference between an alert and an
escalation.

**Breaches are graded by who they hurt, not by how they read.** A removed column with
no consumer is a warning. The same removal with three consumers is an error, because
severity here is a property of the blast radius and not of the change. That is the
one piece of judgement this module makes on its own.

**What it deliberately does not do.** It does not version contracts (see
`graph.history` for the same problem solved for the graph), and it does not negotiate
or approve them. A contract here is a declaration that has been agreed elsewhere; the
value is that it is now declared somewhere a machine reads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from ..core.types import DatasetId
from ..observe.profile import Profile, Severity
from ..observe.quality import Suite, run
from ..observe.schema import diff_schemas

__all__ = [
    "Breach",
    "Contract",
    "ContractReport",
    "breaches",
    "consumers_of",
    "contracts_for",
    "producers",
    "unowned",
    "verify",
]


@dataclass(frozen=True)
class Contract:
    """What one team promises another about one dataset.

    Every field is optional except the dataset and the producer, because a contract
    nobody can write in five minutes is a contract nobody writes. A contract that
    promises only `columns` is still worth more than a wiki page, since it turns a
    column drop into an attributable breach.
    """

    dataset: DatasetId
    producer: str
    consumers: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()  # promised to exist; extras are permitted
    max_staleness: timedelta | None = None
    suite: Suite | None = None
    note: str = ""

    @property
    def is_unconsumed(self) -> bool:
        """A contract with no named consumer promises something to nobody."""
        return not self.consumers

    def __str__(self) -> str:
        to = ", ".join(self.consumers) if self.consumers else "nobody"
        return f"{self.dataset} by {self.producer} to {to}"


@dataclass(frozen=True)
class Breach:
    """One broken promise, and who it was made to."""

    dataset: DatasetId
    producer: str
    consumers: tuple[str, ...]
    kind: str  # missing_column | schema | staleness | expectation
    severity: Severity
    detail: str

    def __str__(self) -> str:
        who = ", ".join(self.consumers) if self.consumers else "no named consumer"
        return (
            f"[{self.severity.value}] {self.dataset}: {self.detail} "
            f"(owed by {self.producer} to {who})"
        )


@dataclass
class ContractReport:
    """Every breach found against one contract."""

    contract: Contract
    breaches: list[Breach] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)

    @property
    def is_met(self) -> bool:
        """True when every clause of the contract holds."""
        return not self.breaches

    @property
    def errors(self) -> list[Breach]:
        """Clauses that failed."""
        return [b for b in self.breaches if b.severity is Severity.ERROR]

    def summary(self) -> str:
        """The report as text, failures first."""
        if self.is_met and not self.unchecked:
            return f"{self.contract}: met"
        lines: list[str] = []
        if self.is_met:
            lines.append(f"{self.contract}: no breach found")
        else:
            lines.append(f"{self.contract}: {len(self.breaches)} breach(es)")
            lines.extend(f"    {b}" for b in self.breaches)
        if self.unchecked:
            lines.append(f"    not checked, nothing supplied: {', '.join(self.unchecked)}")
        return "\n".join(lines)


def _severity_for(contract: Contract, floor: Severity = Severity.WARN) -> Severity:
    """Escalate when somebody is actually relying on the promise.

    The single judgement this module makes: the same change is a warning against a
    dataset nobody consumes and an error against one three teams read.
    """
    return Severity.ERROR if contract.consumers else floor


def verify(
    contract: Contract,
    *,
    profile: Profile | None = None,
    previous: Profile | None = None,
    age: timedelta | None = None,
) -> ContractReport:
    """Check a contract against what is currently true.

    Each promise is checked only if the evidence for it was supplied; anything that
    could not be checked is listed in `unchecked` rather than passing silently. A
    report that looks met because the caller forgot to pass a profile is the failure
    mode this avoids.
    """
    report = ContractReport(contract=contract)

    if contract.columns:
        if profile is None:
            report.unchecked.append("columns (no profile supplied)")
        else:
            present = set(profile.column_names)
            for name in contract.columns:
                if name not in present:
                    report.breaches.append(
                        Breach(
                            dataset=contract.dataset,
                            producer=contract.producer,
                            consumers=contract.consumers,
                            kind="missing_column",
                            severity=_severity_for(contract, Severity.ERROR),
                            detail=f"promised column {name!r} is absent",
                        )
                    )

    if previous is not None and profile is not None:
        for change in diff_schemas(previous, profile).breaking:
            report.breaches.append(
                Breach(
                    dataset=contract.dataset,
                    producer=contract.producer,
                    consumers=contract.consumers,
                    kind="schema",
                    severity=_severity_for(contract),
                    detail=str(change),
                )
            )

    if contract.max_staleness is not None:
        if age is None:
            report.unchecked.append("staleness (no age supplied)")
        elif age > contract.max_staleness:
            report.breaches.append(
                Breach(
                    dataset=contract.dataset,
                    producer=contract.producer,
                    consumers=contract.consumers,
                    kind="staleness",
                    severity=_severity_for(contract),
                    detail=(
                        f"{_hours(age)} old, past the promised {_hours(contract.max_staleness)}"
                    ),
                )
            )

    if contract.suite is not None:
        if profile is None:
            report.unchecked.append("expectations (no profile supplied)")
        else:
            for finding in run(contract.suite, profile).errors:
                report.breaches.append(
                    Breach(
                        dataset=contract.dataset,
                        producer=contract.producer,
                        consumers=contract.consumers,
                        kind="expectation",
                        severity=_severity_for(contract, Severity.ERROR),
                        detail=str(finding),
                    )
                )

    return report


def _hours(span: timedelta) -> str:
    total = span.total_seconds() / 3600
    return f"{total:.1f}h" if total < 48 else f"{span.days}d"


def contracts_for(contracts: Iterable[Contract], dataset: DatasetId) -> list[Contract]:
    """Every contract covering one dataset. More than one is legitimate — two
    consumers may hold different promises about the same table."""
    return [c for c in contracts if c.dataset == dataset]


def consumers_of(contracts: Iterable[Contract], dataset: DatasetId) -> list[str]:
    """Everyone named as a consumer of one dataset, across all its contracts."""
    found: set[str] = set()
    for contract in contracts_for(contracts, dataset):
        found.update(contract.consumers)
    return sorted(found)


def unowned(datasets: Sequence[DatasetId], contracts: Iterable[Contract]) -> list[DatasetId]:
    """Datasets with no contract at all.

    Not a defect on its own — most tables need no contract. It becomes one for a
    dataset other teams read, which is why this takes the dataset list rather than
    deriving it, so a caller can pass only the ones that cross a team boundary.
    """
    covered = {c.dataset for c in contracts}
    return [ds for ds in datasets if ds not in covered]


def breaches(reports: Iterable[ContractReport]) -> list[Breach]:
    """Every breach across many reports, errors first."""
    everything = [b for report in reports for b in report.breaches]
    return sorted(everything, key=lambda b: (b.severity is not Severity.ERROR, str(b.dataset)))


def producers(contracts: Iterable[Contract]) -> Mapping[str, list[DatasetId]]:
    """Which datasets each team has promised something about."""
    out: dict[str, list[DatasetId]] = {}
    for contract in contracts:
        out.setdefault(contract.producer, []).append(contract.dataset)
    return {k: sorted(v) for k, v in sorted(out.items())}
