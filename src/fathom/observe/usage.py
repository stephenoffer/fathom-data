"""Who actually reads a dataset, and what follows from nobody reading it.

The graph answers what *could* consume a dataset. It cannot answer what *does*, and
the difference is where most of a warehouse bill goes: tables built nightly for a
dashboard that was decommissioned in 2023, kept because the structural question —
"does anything depend on this?" — keeps answering yes about a consumer that is itself
dead.

Structural orphans (`graph.query.isolated`) find the easy case, where nothing points
at a dataset at all. They are rare, because a dead chain still has edges. What is
needed is the observed case: a dataset nothing has *read* in ninety days, whose
descendants nothing has read either.

**The one thing this module refuses to say.** Absence of a read event is not evidence
of absence. Query logs have retention windows, some consumers do not log, and a table
read once a quarter looks identical to a dead one over thirty days. So nothing here
returns "unused". Everything returns "no reads observed", carries the `window` it
observed over, and puts that window in its own summary text. `retirement_candidates`
is named for what it produces — a list for a person to review — and not for a
conclusion.

That is not excessive caution. Deleting a table that is read annually for a
regulatory filing is the one mistake in this area that cannot be undone by rebuilding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.types import DatasetId
from ..core.util.clock import as_utc, now
from ..graph.model import Graph
from ..graph.query import descendants

__all__ = [
    "ReadEvent",
    "RetirementCandidate",
    "UsageStats",
    "busiest",
    "events_from",
    "never_observed",
    "principals_of",
    "read_counts",
    "read_ratio",
    "retirement_candidates",
    "summarize",
    "unread_since",
]


@dataclass(frozen=True)
class ReadEvent:
    """One observed read of a dataset.

    `principal` is whoever did the reading — a user, a service account, a dashboard
    id, a scheduled job. It is a free string because the useful grouping differs by
    platform, and normalizing it here would throw away the distinction between a
    person and a pipeline, which is the distinction that matters most.
    """

    dataset: DatasetId
    principal: str
    at: datetime
    kind: str = "query"  # query | export | dashboard | api
    query_id: str = ""

    def __str__(self) -> str:
        return f"{self.principal} read {self.dataset} at {as_utc(self.at).isoformat()}"


@dataclass
class UsageStats:
    """What was observed about one dataset, over a stated window."""

    dataset: DatasetId
    reads: int = 0
    principals: set[str] = field(default_factory=set)
    kinds: dict[str, int] = field(default_factory=dict)
    first_read: datetime | None = None
    last_read: datetime | None = None
    window: timedelta | None = None  # how far back the observation reached

    @property
    def human_principals(self) -> set[str]:
        """Principals that are not obviously a scheduled job.

        A table read only by the pipeline that populates its own downstream is not
        being *used*; it is being maintained. Heuristic, and named as one.
        """
        return {p for p in self.principals if not _looks_scheduled(p)}

    def age(self, *, at: datetime | None = None) -> timedelta | None:
        """How long since the last observed read."""
        if self.last_read is None:
            return None
        return (at or now()) - as_utc(self.last_read)

    def summary(self) -> str:
        """Usage as text: who read this, how often, and how recently."""
        if self.reads == 0:
            return f"{self.dataset}: no reads observed" + (
                f" in the last {_days(self.window)}" if self.window else ""
            )
        last = as_utc(self.last_read).date().isoformat() if self.last_read else "unknown"
        return (
            f"{self.dataset}: {self.reads} read(s) by {len(self.principals)} principal(s), "
            f"last {last}"
        )


_SCHEDULED_HINTS = ("airflow", "dagster", "dbt", "svc_", "service_", "-job", "_job", "system")


def _looks_scheduled(principal: str) -> bool:
    lowered = principal.lower()
    return any(hint in lowered for hint in _SCHEDULED_HINTS)


def _days(window: timedelta | None) -> str:
    return f"{window.days} day(s)" if window else "the observed window"


def summarize(
    events: Iterable[ReadEvent], *, window: timedelta | None = None
) -> dict[DatasetId, UsageStats]:
    """Aggregate read events per dataset.

    `window` is carried through onto every `UsageStats` rather than computed from the
    events, because the span of the events is how far back reads *happened* and the
    window is how far back the log *reached*. Inferring one from the other is what
    turns a short retention period into a false report of disuse.
    """
    out: dict[DatasetId, UsageStats] = {}
    for event in events:
        stats = out.setdefault(event.dataset, UsageStats(dataset=event.dataset, window=window))
        stats.reads += 1
        stats.principals.add(event.principal)
        stats.kinds[event.kind] = stats.kinds.get(event.kind, 0) + 1
        moment = as_utc(event.at)
        if stats.first_read is None or moment < as_utc(stats.first_read):
            stats.first_read = moment
        if stats.last_read is None or moment > as_utc(stats.last_read):
            stats.last_read = moment
    return out


def never_observed(
    graph: Graph, stats: Mapping[DatasetId, UsageStats], *, window: timedelta | None = None
) -> list[UsageStats]:
    """Datasets in the graph with no read event at all in the window.

    Returns `UsageStats` rather than bare ids so the window travels with the answer
    and a caller cannot report "unused" without also having been handed "over what
    period".
    """
    return [
        stats.get(ds) or UsageStats(dataset=ds, window=window)
        for ds in graph.datasets
        if stats.get(ds, UsageStats(dataset=ds)).reads == 0
    ]


def unread_since(stats: Mapping[DatasetId, UsageStats], *, since: datetime) -> list[UsageStats]:
    """Datasets whose most recent observed read predates `since`."""
    out = [s for s in stats.values() if s.last_read is not None and as_utc(s.last_read) < since]
    return sorted(out, key=lambda s: as_utc(s.last_read))  # type: ignore[arg-type]


@dataclass(frozen=True)
class RetirementCandidate:
    """A dataset worth *reviewing* for retirement, with the reason and the caveat."""

    dataset: DatasetId
    reason: str
    window: timedelta | None
    descendants_checked: int
    last_read: datetime | None = None

    def __str__(self) -> str:
        return (
            f"{self.dataset}: {self.reason} (checked {self.descendants_checked} descendant(s); "
            f"no reads observed in {_days(self.window)} — absence of a read event is not "
            f"evidence the dataset is unused)"
        )


def retirement_candidates(
    graph: Graph,
    stats: Mapping[DatasetId, UsageStats],
    *,
    window: timedelta | None = None,
    ignore_scheduled: bool = True,
) -> list[RetirementCandidate]:
    """Datasets nothing read, whose descendants nothing read either.

    A dataset is only a candidate when its whole downstream cone is also unread —
    otherwise it is not unused, it is one hop away from something that is used.

    `ignore_scheduled` discounts principals that look like a scheduler, so a table
    read solely by the job that maintains its own downstream is still a candidate.
    That is a heuristic on a free-text field and will occasionally be wrong, which is
    survivable precisely because the output is a review list.
    """

    def read_by_anyone(ds: DatasetId) -> bool:
        """True when any principal read this dataset in the window."""
        found = stats.get(ds)
        if found is None or found.reads == 0:
            return False
        return bool(found.human_principals) if ignore_scheduled else True

    out: list[RetirementCandidate] = []
    for ds in graph.datasets:
        if read_by_anyone(ds):
            continue
        below = descendants(graph, ds)
        if any(read_by_anyone(child) for child in below):
            continue
        found = stats.get(ds)
        out.append(
            RetirementCandidate(
                dataset=ds,
                reason=(
                    "no reads observed, and none across its downstream"
                    if below
                    else "no reads observed, and nothing derives from it"
                ),
                window=window if window is not None else (found.window if found else None),
                descendants_checked=len(below),
                last_read=found.last_read if found else None,
            )
        )
    return out


def read_counts(
    stats: Mapping[DatasetId, UsageStats], *, people_only: bool = False
) -> dict[DatasetId, int]:
    """Reads per dataset, as the plain mapping `graph.plan.lifetime.value` expects.

    `people_only` drops datasets whose only readers look like scheduled jobs, so a
    table maintained by its own pipeline reports zero rather than looking read.

    The choice is a parameter rather than a default because the two answers differ in
    a way that changes conclusions, and because `retirement_candidates` already
    discounts schedulers. Passing raw counts to `value` while `retirement_candidates`
    discounts them is a real trap; this exists so the decision is made visibly at the
    call site rather than inherited by accident.
    """
    out: dict[DatasetId, int] = {}
    for ds, found in stats.items():
        if people_only and not found.human_principals:
            out[ds] = 0
        else:
            out[ds] = found.reads
    return out


def busiest(
    stats: Mapping[DatasetId, UsageStats], *, limit: int = 10
) -> list[tuple[DatasetId, int]]:
    """The most-read datasets, which is where a caching or materialization budget goes."""
    ranked = sorted(stats.values(), key=lambda s: (-s.reads, str(s.dataset)))
    return [(s.dataset, s.reads) for s in ranked[:limit]]


def principals_of(stats: Mapping[DatasetId, UsageStats], dataset: DatasetId) -> list[str]:
    """Everyone observed reading one dataset."""
    found = stats.get(dataset)
    return sorted(found.principals) if found else []


def read_ratio(graph: Graph, stats: Mapping[DatasetId, UsageStats]) -> float:
    """Fraction of the graph's datasets with at least one observed read.

    A low number is ambiguous by design and worth stating as such: it means either
    that much of the warehouse is dead, or that read logging does not cover it. Both
    are worth knowing and they are not distinguishable from here.
    """
    if not graph.datasets:
        return 0.0
    seen = sum(1 for ds in graph.datasets if stats.get(ds, UsageStats(dataset=ds)).reads > 0)
    return seen / len(graph.datasets)


def events_from(
    rows: Sequence[tuple[DatasetId, str, datetime]], *, kind: str = "query"
) -> list[ReadEvent]:
    """Build read events from a query log's (dataset, principal, timestamp) rows."""
    return [ReadEvent(ds, principal, at, kind=kind) for ds, principal, at in rows]
