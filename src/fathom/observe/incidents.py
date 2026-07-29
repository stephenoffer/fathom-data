"""Turning a finding into tracked work, and learning something from it afterwards.

Every check in this package produces a finding and then stops. A finding that nobody
owns is a finding nobody fixes, and the second time it fires people already know it
is "just that alert".

The parts that earn their place:

**Grouping.** One upstream breakage produces a finding on every downstream dataset.
`correlate` groups findings that share a root through the lineage graph, so an
incident is one incident rather than fourteen tickets filed against fourteen teams.

**Blast radius from the graph, not from the finding.** `impact` walks reachability,
because the datasets affected by a breach are the ones downstream of it, and those
are exactly the ones whose owners have not noticed yet.

**Detection and start are different times.** `time_to_detect` is the gap between
them, and it is the number that says whether monitoring works. A team with a fast
`time_to_resolve` and a slow `time_to_detect` does not have good operations; it has
customers doing its monitoring.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from ..core.types import DatasetId

__all__ = [
    "Finding",
    "Incident",
    "IncidentState",
    "Postmortem",
    "Severity",
    "acknowledge",
    "correlate",
    "impact",
    "mean_time_to_detect",
    "mean_time_to_resolve",
    "open_incidents",
    "postmortem",
    "recurring",
    "resolve",
    "root_datasets",
    "severity_of",
    "summarize_incidents",
    "time_to_detect",
    "time_to_resolve",
    "timeline",
]


class Severity(StrEnum):
    """How much of a problem this is. Ordered by `rank`, not alphabetically."""

    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)


class IncidentState(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"

    @property
    def is_terminal(self) -> bool:
        return self in {IncidentState.RESOLVED, IncidentState.WONT_FIX}


@dataclass(frozen=True)
class Finding:
    """One check's output, with enough context to become work.

    `detected` and `started` are separate on purpose: a freshness breach detected at
    09:00 may have started at 02:00, and conflating them makes monitoring look six
    hours better than it is.
    """

    check: str
    dataset: DatasetId
    summary: str
    severity: Severity = Severity.MINOR
    detected: datetime | None = None
    started: datetime | None = None
    details: Mapping[str, str] = field(default_factory=dict)

    @property
    def detection_lag(self) -> timedelta | None:
        if self.detected is None or self.started is None:
            return None
        return self.detected - self.started


@dataclass
class Incident:
    """A group of findings with one cause and one owner."""

    identifier: str
    findings: list[Finding] = field(default_factory=list)
    state: IncidentState = IncidentState.OPEN
    owner: str = ""
    opened: datetime | None = None
    acknowledged: datetime | None = None
    resolved: datetime | None = None
    cause: str = ""
    affected: tuple[DatasetId, ...] = ()

    @property
    def severity(self) -> Severity:
        """The worst finding's severity, by rank.

        `max()` over a StrEnum compares strings, which puts "minor" above "critical".
        """
        if not self.findings:
            return Severity.INFO
        return max((f.severity for f in self.findings), key=lambda s: s.rank)

    @property
    def is_open(self) -> bool:
        return not self.state.is_terminal

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        seen: dict[DatasetId, None] = {}
        for finding in self.findings:
            seen.setdefault(finding.dataset, None)
        return tuple(seen)


def severity_of(findings: Iterable[Finding]) -> Severity:
    found = list(findings)
    if not found:
        return Severity.INFO
    return max((f.severity for f in found), key=lambda s: s.rank)


def root_datasets(
    findings: Sequence[Finding], upstream: Mapping[DatasetId, frozenset[DatasetId]]
) -> set[DatasetId]:
    """The findings' datasets that have no other affected dataset upstream of them.

    These are where to look. Everything else is a consequence.
    """
    affected = {f.dataset for f in findings}
    return {d for d in affected if not (upstream.get(d, frozenset()) & affected)}


def correlate(
    findings: Sequence[Finding],
    upstream: Mapping[DatasetId, frozenset[DatasetId]],
    *,
    window: timedelta = timedelta(hours=1),
) -> list[Incident]:
    """Group findings that plausibly share a cause.

    Two conditions, both required. They must be *connected through lineage* — one
    reachable from the other — and they must be *close in time*. Lineage alone would
    merge a genuine breach with an unrelated schema drift on the same table a month
    later; time alone would merge every check that happens to run at 09:00.
    """
    ordered = sorted(findings, key=lambda f: f.detected or datetime.min.replace(tzinfo=UTC))
    groups: list[list[Finding]] = []

    for finding in ordered:
        placed = False
        for group in groups:
            if _shares_cause(finding, group, upstream, window):
                group.append(finding)
                placed = True
                break
        if not placed:
            groups.append([finding])

    incidents: list[Incident] = []
    for index, group in enumerate(groups, start=1):
        roots = root_datasets(group, upstream)
        times = [f.detected for f in group if f.detected]
        incidents.append(
            Incident(
                identifier=f"INC-{index:04d}",
                findings=list(group),
                opened=min(times) if times else None,
                cause=(
                    f"originates at {', '.join(sorted(str(d) for d in roots))}" if roots else ""
                ),
            )
        )
    return incidents


def _shares_cause(
    finding: Finding,
    group: Sequence[Finding],
    upstream: Mapping[DatasetId, frozenset[DatasetId]],
    window: timedelta,
) -> bool:
    for other in group:
        connected = (
            finding.dataset in upstream.get(other.dataset, frozenset())
            or (other.dataset in upstream.get(finding.dataset, frozenset()))
            or finding.dataset == other.dataset
        )
        if not connected:
            continue
        if finding.detected is None or other.detected is None:
            return True  # no times to disprove it with; grouping is the safer error
        if abs(finding.detected - other.detected) <= window:
            return True
    return False


def impact(
    incident: Incident, downstream: Mapping[DatasetId, frozenset[DatasetId]]
) -> tuple[DatasetId, ...]:
    """Everything reachable from the incident's datasets.

    Taken from the graph rather than from the findings, because the datasets that
    matter are the ones nobody has checked yet.
    """
    reached: set[DatasetId] = set()
    for dataset in incident.datasets:
        reached.add(dataset)
        reached |= downstream.get(dataset, frozenset())
    return tuple(sorted(reached, key=str))


def acknowledge(incident: Incident, owner: str, *, at: datetime | None = None) -> Incident:
    """Give an incident an owner. Unowned incidents are not being worked on."""
    if not owner:
        raise ValueError(
            "an incident needs a named owner; acknowledging without one records that "
            "somebody saw it, which is not the same as somebody fixing it"
        )
    incident.owner = owner
    incident.state = IncidentState.ACKNOWLEDGED
    incident.acknowledged = at or datetime.now(UTC)
    return incident


def resolve(
    incident: Incident,
    *,
    cause: str = "",
    at: datetime | None = None,
    state: IncidentState = IncidentState.RESOLVED,
) -> Incident:
    if not state.is_terminal:
        raise ValueError(f"{state.value} is not a terminal state")
    incident.state = state
    incident.resolved = at or datetime.now(UTC)
    if cause:
        incident.cause = cause
    return incident


def time_to_detect(incident: Incident) -> timedelta | None:
    """How long the problem existed before anything noticed.

    The number that says whether monitoring works, as distinct from whether the team
    is fast at fixing things once a customer tells them.
    """
    lags = [f.detection_lag for f in incident.findings if f.detection_lag is not None]
    return max(lags) if lags else None


def time_to_resolve(incident: Incident) -> timedelta | None:
    if incident.opened is None or incident.resolved is None:
        return None
    return incident.resolved - incident.opened


def mean_time_to_detect(incidents: Iterable[Incident]) -> timedelta | None:
    values = [t for t in (time_to_detect(i) for i in incidents) if t is not None]
    return sum(values, timedelta()) / len(values) if values else None


def mean_time_to_resolve(incidents: Iterable[Incident]) -> timedelta | None:
    values = [t for t in (time_to_resolve(i) for i in incidents) if t is not None]
    return sum(values, timedelta()) / len(values) if values else None


def open_incidents(incidents: Iterable[Incident]) -> list[Incident]:
    """Still open, worst first."""
    return sorted(
        (i for i in incidents if i.is_open),
        key=lambda i: (-i.severity.rank, i.opened or datetime.max.replace(tzinfo=UTC)),
    )


def recurring(incidents: Iterable[Incident], *, threshold: int = 2) -> dict[str, int]:
    """Problems that have happened more than once.

    A recurrence means the previous fix addressed the symptom. This is usually a more
    useful list than the open one.
    """
    counts: dict[str, int] = {}
    for incident in incidents:
        for finding in incident.findings:
            key = f"{finding.check}:{finding.dataset}"
            counts[key] = counts.get(key, 0) + 1
    return {k: v for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v >= threshold}


def timeline(incident: Incident) -> list[tuple[datetime, str]]:
    """Everything that happened, in order. What a postmortem is written from."""
    events: list[tuple[datetime, str]] = []
    for finding in incident.findings:
        if finding.started:
            events.append((finding.started, f"{finding.dataset} broke ({finding.check})"))
        if finding.detected:
            events.append((finding.detected, f"{finding.check} detected it on {finding.dataset}"))
    if incident.acknowledged:
        events.append((incident.acknowledged, f"acknowledged by {incident.owner or 'nobody'}"))
    if incident.resolved:
        events.append((incident.resolved, f"{incident.state.value}"))
    return sorted(events, key=lambda e: e[0])


@dataclass(frozen=True)
class Postmortem:
    """What an incident should leave behind.

    `regression_checks` is the point of the whole exercise: an incident that produces
    prose and no check will happen again.
    """

    incident: str
    severity: Severity
    datasets: tuple[DatasetId, ...]
    detected_after: timedelta | None
    resolved_after: timedelta | None
    cause: str
    events: tuple[tuple[datetime, str], ...]
    regression_checks: tuple[str, ...] = ()

    def to_markdown(self) -> str:
        lines = [
            f"# {self.incident} ({self.severity.value})",
            "",
            f"- Datasets: {', '.join(str(d) for d in self.datasets) or 'none recorded'}",
            f"- Undetected for: {self.detected_after or 'unknown'}",
            f"- Open for: {self.resolved_after or 'still open'}",
            f"- Cause: {self.cause or 'not recorded'}",
            "",
            "## Timeline",
            "",
        ]
        lines.extend(f"- {when:%Y-%m-%d %H:%M} — {what}" for when, what in self.events)
        lines.extend(["", "## Regression checks", ""])
        if self.regression_checks:
            lines.extend(f"- [ ] {check}" for check in self.regression_checks)
        else:
            lines.append(
                "- [ ] None proposed. An incident that produces prose and no check "
                "will happen again."
            )
        return "\n".join(lines)


def postmortem(incident: Incident) -> Postmortem:
    """Assemble the record, proposing a regression check per distinct failed check."""
    checks = sorted({f.check for f in incident.findings})
    return Postmortem(
        incident=incident.identifier,
        severity=incident.severity,
        datasets=incident.datasets,
        detected_after=time_to_detect(incident),
        resolved_after=time_to_resolve(incident),
        cause=incident.cause,
        events=tuple(timeline(incident)),
        regression_checks=tuple(
            f"{check} runs on every affected dataset and fails the build" for check in checks
        ),
    )


def summarize_incidents(incidents: Sequence[Incident]) -> str:
    if not incidents:
        return "no incidents"
    still_open = open_incidents(incidents)
    mttd, mttr = mean_time_to_detect(incidents), mean_time_to_resolve(incidents)
    lines = [
        f"{len(incidents)} incident(s), {len(still_open)} open, "
        f"worst is {severity_of(f for i in incidents for f in i.findings).value}"
    ]
    if mttd:
        lines.append(f"  mean time to detect: {mttd}")
    if mttr:
        lines.append(f"  mean time to resolve: {mttr}")
    repeats = recurring(incidents)
    if repeats:
        lines.append(f"  recurring: {', '.join(f'{k} ×{v}' for k, v in repeats.items())}")
    return "\n".join(lines)
