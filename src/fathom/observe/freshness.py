"""Freshness that accounts for what a dataset is built from.

A table rebuilt five minutes ago is not fresh if its input has not landed since
Tuesday. Every freshness check that looks only at build time gets this wrong, and it
gets it wrong in the reassuring direction: the dashboard is green and the numbers are
four days old.

The correct notion is transitive. A dataset's effective freshness is the *oldest*
freshness anywhere upstream of it, because that is the age of the information it
actually carries. `effective_age` computes it; `blame` names the input responsible,
which is what turns a red SLA into a ticket someone can act on.

The second thing here is the difference between *stale* and *late*. Stale means the
data is older than the SLA allows. Late means the build has not run when it should
have. They have different fixes — a slow upstream versus a broken scheduler — and
reporting them as one number is why freshness alerts get muted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ..core.types import DatasetId
from ..core.util.clock import as_utc
from ..graph.model import Graph
from ..graph.query import closure, descendants, fold_downstream, shortest_path

__all__ = [
    "FreshnessReport",
    "SLA",
    "age",
    "blame",
    "effective_age",
    "expected_next_build",
    "freshness_path",
    "slas_from",
    "effective_freshness",
    "is_fresh",
    "is_late",
    "propagate_freshness",
    "report",
    "sla_violations",
    "stale_closure",
    "worst_offenders",
]


@dataclass(frozen=True)
class SLA:
    """How fresh a dataset is required to be, and how often it should build.

    `max_age` is about the data. `expected_interval` is about the schedule. Both are
    optional and they fail differently, which is the point of keeping them apart.
    """

    dataset: DatasetId
    max_age: timedelta | None = None
    expected_interval: timedelta | None = None
    owner: str = ""

    def __str__(self) -> str:
        parts = []
        if self.max_age:
            parts.append(f"max age {self.max_age}")
        if self.expected_interval:
            parts.append(f"every {self.expected_interval}")
        return f"{self.dataset}: {', '.join(parts) or 'no SLA'}"


def age(last_built: datetime | None, *, now: datetime | None = None) -> timedelta | None:
    """Time since a dataset was last built, or None when it never was."""
    if last_built is None:
        return None
    return (now or datetime.now(UTC)) - as_utc(last_built)


def effective_age(
    graph: Graph,
    ds: DatasetId,
    last_built: Mapping[DatasetId, datetime],
    *,
    now: datetime | None = None,
) -> timedelta | None:
    """The age of the oldest information this dataset carries.

    Takes the maximum age across the dataset and everything upstream of it, which is
    the only freshness figure that means what people assume freshness means.

    Returns None when nothing in the closure has a build time at all — unknown, not
    fresh. Reporting an unmeasured dataset as current is the failure this module
    exists to prevent.
    """
    reference = now or datetime.now(UTC)
    ages = [
        reference - as_utc(last_built[node]) for node in closure(graph, ds) if node in last_built
    ]
    return max(ages) if ages else None


def effective_freshness(
    graph: Graph, last_built: Mapping[DatasetId, datetime], *, now: datetime | None = None
) -> dict[DatasetId, timedelta | None]:
    """Effective age for every dataset in the graph."""
    return {ds: effective_age(graph, ds, last_built, now=now) for ds in graph.datasets}


def propagate_freshness(
    graph: Graph, last_built: Mapping[DatasetId, datetime]
) -> dict[DatasetId, datetime]:
    """The effective build timestamp of each dataset, oldest-upstream-wins.

    The timestamp form of `effective_age`, folded downstream so a long chain costs
    one pass rather than one traversal per node. Oldest wins at every join, because a
    dataset carries information no fresher than its stalest input.
    """
    return fold_downstream(
        graph,
        {ds: as_utc(stamp) for ds, stamp in last_built.items()},
        combine=min,
    )


def blame(
    graph: Graph,
    ds: DatasetId,
    last_built: Mapping[DatasetId, datetime],
    *,
    now: datetime | None = None,
) -> tuple[DatasetId, timedelta] | None:
    """The upstream dataset responsible for this one's effective age.

    What a freshness alert should say instead of naming the dashboard: the dashboard
    is fine, and `raw.fx_rates` has not landed since Tuesday.
    """
    reference = now or datetime.now(UTC)
    worst: tuple[DatasetId, timedelta] | None = None
    for node in closure(graph, ds):
        stamp = last_built.get(node)
        if stamp is None:
            continue
        node_age = reference - as_utc(stamp)
        if worst is None or node_age > worst[1]:
            worst = (node, node_age)
    return worst


def is_fresh(
    graph: Graph,
    sla: SLA,
    last_built: Mapping[DatasetId, datetime],
    *,
    now: datetime | None = None,
) -> bool:
    """True when the data is within its age budget, upstream included.

    An SLA with no `max_age` is trivially satisfied. A dataset with no build time
    anywhere in its closure is not fresh, because nothing established that it is.
    """
    if sla.max_age is None:
        return True
    current = effective_age(graph, sla.dataset, last_built, now=now)
    return False if current is None else current <= sla.max_age


def is_late(
    sla: SLA, last_built: Mapping[DatasetId, datetime], *, now: datetime | None = None
) -> bool:
    """True when this dataset's own build has not run on schedule.

    Distinct from staleness: a late build with a fresh upstream is a scheduler
    problem, and a stale dataset with an on-time build is an upstream problem.
    """
    if sla.expected_interval is None:
        return False
    own = age(last_built.get(sla.dataset), now=now)
    return True if own is None else own > sla.expected_interval


def stale_closure(
    graph: Graph,
    ds: DatasetId,
    last_built: Mapping[DatasetId, datetime],
    *,
    max_age: timedelta,
    now: datetime | None = None,
) -> list[DatasetId]:
    """Upstream datasets older than the budget — every contributor to the breach."""
    reference = now or datetime.now(UTC)
    out: list[DatasetId] = []
    for node in closure(graph, ds):
        stamp = last_built.get(node)
        if stamp is None or (reference - as_utc(stamp)) > max_age:
            out.append(node)
    return sorted(out, key=str)


@dataclass
class FreshnessReport:
    """Freshness across a set of SLAs, with causes attached."""

    stale: list[tuple[DatasetId, timedelta | None, DatasetId | None]] = field(default_factory=list)
    late: list[DatasetId] = field(default_factory=list)
    unmeasured: list[DatasetId] = field(default_factory=list)
    ok: list[DatasetId] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when nothing breached its freshness budget."""
        return not (self.stale or self.late or self.unmeasured)

    def summary(self) -> str:
        """The report as text, stalest first."""
        if self.is_clean:
            return f"freshness: all {len(self.ok)} dataset(s) within SLA"
        lines = [
            f"freshness: {len(self.stale)} stale, {len(self.late)} late, "
            f"{len(self.unmeasured)} unmeasured, {len(self.ok)} ok"
        ]
        for ds, current, cause in self.stale:
            because = f" — oldest input {cause}" if cause and cause != ds else ""
            lines.append(f"  STALE {ds}: {current or 'never built'}{because}")
        for ds in self.late:
            lines.append(f"  LATE  {ds}: build has not run on schedule")
        for ds in self.unmeasured:
            lines.append(f"  ?     {ds}: no build time recorded anywhere upstream")
        return "\n".join(lines)


def report(
    graph: Graph,
    slas: Iterable[SLA],
    last_built: Mapping[DatasetId, datetime],
    *,
    now: datetime | None = None,
) -> FreshnessReport:
    """Evaluate every SLA, separating stale from late from unmeasured."""
    out = FreshnessReport()
    for sla in slas:
        current = effective_age(graph, sla.dataset, last_built, now=now)
        if current is None:
            out.unmeasured.append(sla.dataset)
            continue
        if is_late(sla, last_built, now=now):
            out.late.append(sla.dataset)
        if sla.max_age is not None and current > sla.max_age:
            cause = blame(graph, sla.dataset, last_built, now=now)
            out.stale.append((sla.dataset, current, cause[0] if cause else None))
        elif sla.dataset not in out.late:
            out.ok.append(sla.dataset)
    return out


def sla_violations(
    graph: Graph,
    slas: Iterable[SLA],
    last_built: Mapping[DatasetId, datetime],
    *,
    now: datetime | None = None,
) -> list[str]:
    """Rendered violations, one line each, cause included."""
    result = report(graph, slas, last_built, now=now)
    lines = []
    for ds, current, cause in result.stale:
        because = f", oldest input {cause}" if cause and cause != ds else ""
        lines.append(f"{ds}: stale ({current}{because})")
    lines.extend(f"{ds}: build has not run on schedule" for ds in result.late)
    lines.extend(f"{ds}: freshness unmeasured" for ds in result.unmeasured)
    return lines


def worst_offenders(
    graph: Graph,
    last_built: Mapping[DatasetId, datetime],
    *,
    now: datetime | None = None,
    limit: int = 10,
) -> list[tuple[DatasetId, timedelta, int]]:
    """Stale datasets ranked by age multiplied by how much they hold back.

    A stale leaf inconveniences one report. A stale source three levels down holds
    back everything derived from it, and this ordering says so.
    """
    reference = now or datetime.now(UTC)
    scored: list[tuple[DatasetId, timedelta, int]] = []
    for ds, stamp in last_built.items():
        node_age = reference - as_utc(stamp)
        scored.append((ds, node_age, len(descendants(graph, ds))))
    scored.sort(key=lambda item: (-(item[1].total_seconds() * max(1, item[2])), str(item[0])))
    return scored[:limit]


def freshness_path(
    graph: Graph,
    ds: DatasetId,
    last_built: Mapping[DatasetId, datetime],
    *,
    now: datetime | None = None,
) -> list[DatasetId]:
    """The route from the oldest upstream contributor to the dataset in question.

    The lineage a freshness incident write-up needs, without anybody clicking through
    a UI to reconstruct it.
    """
    cause = blame(graph, ds, last_built, now=now)
    if cause is None or cause[0] == ds:
        return [ds]
    return shortest_path(graph, cause[0], ds) or [cause[0], ds]


def expected_next_build(sla: SLA, last_built: Mapping[DatasetId, datetime]) -> datetime | None:
    """When this dataset should next build, from its own schedule."""
    if sla.expected_interval is None:
        return None
    stamp = last_built.get(sla.dataset)
    return None if stamp is None else as_utc(stamp) + sla.expected_interval


def slas_from(graph: Graph, *, max_age: timedelta) -> list[SLA]:
    """A uniform SLA over every leaf dataset.

    A starting point, not a policy. Leaves are what people consume, so they are where
    a freshness budget is worth having before anyone has written one down.
    """
    from ..graph.query import leaves

    return [SLA(dataset=ds, max_age=max_age) for ds in leaves(graph)]
