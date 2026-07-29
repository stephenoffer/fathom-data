"""What a dataset has cost since it was created, against what it is worth.

`cost` prices a plan: this run, this many partitions, this much money. That answers
"should I run this backfill" and cannot answer the question a platform owner actually
has to answer once a quarter, which is "should this table exist at all".

A table costing $40 a night is invisible. Three hundred of them is the budget. The
reason nobody culls them is that the two facts needed to justify it live in different
systems and are never divided by one another: what a dataset has cost over its life,
and whether anybody reads it.

This module holds the first. `observe.usage` holds the second. `value` puts them
together, and is the only function here that produces a judgement.

**Cost is accumulated, never modelled.** A lifetime figure derived by multiplying a
current per-run cost by an assumed age would be a guess wearing a number's clothing,
and it would always flatter recently-created tables. `accumulate` takes the runs that
actually happened. A dataset with no recorded runs has no lifetime cost — `None`, not
zero — because "we never measured this" and "this was free" are different facts and
only one of them justifies keeping a table.

**What `value` will not do.** It returns a `Verdict`, never a decision, and every
verdict that recommends looking at a dataset carries the same caveat `observe.usage`
carries: no reads observed is not no reads. Cost is the reliable half of this ratio.
Usage is the half with a blind spot, and the output says which is which.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from ...core.types import DatasetId
from ...core.util.clock import as_utc
from .cost import CostModel, PartitionCost

__all__ = [
    "LifetimeCost",
    "RunRecord",
    "ValueFinding",
    "Verdict",
    "accumulate",
    "actionable",
    "burn_rate",
    "most_expensive_lifetime",
    "summarize",
    "total_spend",
    "unmeasured",
    "value",
]


@dataclass(frozen=True)
class RunRecord:
    """One build of one dataset that actually happened.

    Deliberately the same three bases `cost.PartitionCost` uses, so a run recorded by
    an orchestrator prices under the caller's existing `CostModel` with no second
    model to keep in step.
    """

    dataset: DatasetId
    at: datetime
    partitions: int = 0
    bytes_scanned: int = 0
    tokens: int = 0
    seconds: float = 0.0

    def as_partition_cost(self) -> PartitionCost:
        """This measurement expressed as a `PartitionCost`."""
        return PartitionCost(
            dataset=self.dataset,
            partitions=self.partitions,
            bytes_scanned=self.bytes_scanned,
            tokens=self.tokens,
            seconds=self.seconds,
        )


@dataclass
class LifetimeCost:
    """Everything one dataset has cost across its recorded runs."""

    dataset: DatasetId
    runs: int = 0
    spend: float = 0.0
    partitions: int = 0
    bytes_scanned: int = 0
    tokens: int = 0
    seconds: float = 0.0
    first_run: datetime | None = None
    last_run: datetime | None = None

    @property
    def is_measured(self) -> bool:
        """False when no run was ever recorded, which is not the same as free."""
        return self.runs > 0

    @property
    def span(self) -> timedelta | None:
        """Period this measurement covers."""
        if self.first_run is None or self.last_run is None:
            return None
        return as_utc(self.last_run) - as_utc(self.first_run)

    @property
    def per_run(self) -> float | None:
        """Cost of a single run."""
        return self.spend / self.runs if self.runs else None

    def summary(self) -> str:
        """The verdict as text, with the reasoning behind it."""
        if not self.is_measured:
            return f"{self.dataset}: no runs recorded — not measured, which is not free"
        window = f" over {self.span.days} day(s)" if self.span else ""
        return (
            f"{self.dataset}: {self.spend:,.2f} across {self.runs} run(s){window} "
            f"({self.partitions} partition(s))"
        )


def accumulate(records: Iterable[RunRecord], model: CostModel) -> dict[DatasetId, LifetimeCost]:
    """Total what each dataset has cost across the runs that actually happened.

    Only datasets appearing in `records` are present in the result. A dataset absent
    from the log is unmeasured rather than free, and inventing a zero entry for it
    would make it look like the cheapest thing in the warehouse.
    """
    out: dict[DatasetId, LifetimeCost] = {}
    for record in records:
        total = out.setdefault(record.dataset, LifetimeCost(dataset=record.dataset))
        total.runs += 1
        total.spend += model.cost_of(record.as_partition_cost())
        total.partitions += record.partitions
        total.bytes_scanned += record.bytes_scanned
        total.tokens += record.tokens
        total.seconds += record.seconds
        moment = as_utc(record.at)
        if total.first_run is None or moment < as_utc(total.first_run):
            total.first_run = moment
        if total.last_run is None or moment > as_utc(total.last_run):
            total.last_run = moment
    return out


def burn_rate(total: LifetimeCost, *, per: timedelta = timedelta(days=30)) -> float | None:
    """Spend projected forward over `per`, from the observed run history.

    Returns `None` for a dataset whose runs all fall at one instant, since a rate
    needs a span to divide by and one point does not have one.
    """
    span = total.span
    if span is None or span.total_seconds() <= 0:
        return None
    return total.spend * (per.total_seconds() / span.total_seconds())


def total_spend(totals: Mapping[DatasetId, LifetimeCost]) -> float:
    """What the measured part of the warehouse has cost."""
    return sum(t.spend for t in totals.values())


def most_expensive_lifetime(
    totals: Mapping[DatasetId, LifetimeCost], *, limit: int = 10
) -> list[LifetimeCost]:
    """The datasets that have cost the most, which is where a cull starts."""
    return sorted(totals.values(), key=lambda t: -t.spend)[:limit]


class Verdict(StrEnum):
    """What the cost-against-usage ratio suggests, never what to do."""

    EARNING = "earning"  # read, at any cost
    UNMEASURED = "unmeasured"  # no cost history, no basis for a view
    CHEAP_AND_QUIET = "cheap_and_quiet"  # unread but costs little; not worth the review
    REVIEW = "review"  # unread and expensive


@dataclass(frozen=True)
class ValueFinding:
    """One dataset's cost set against whether anyone reads it."""

    dataset: DatasetId
    verdict: Verdict
    spend: float | None
    reads: int
    window: timedelta | None

    @property
    def is_actionable(self) -> bool:
        """True when the evidence supports acting rather than reviewing."""
        return self.verdict is Verdict.REVIEW

    def __str__(self) -> str:
        if self.verdict is Verdict.UNMEASURED:
            return f"{self.dataset}: no cost history — unmeasured, not free"
        spend = f"{self.spend:,.2f}" if self.spend is not None else "?"
        if self.verdict is Verdict.EARNING:
            return f"{self.dataset}: {spend} spent, {self.reads} read(s) — earning it"
        caveat = (
            " — no reads observed"
            + (f" in {self.window.days} day(s)" if self.window else "")
            + ", which is not the same as no reads"
        )
        if self.verdict is Verdict.CHEAP_AND_QUIET:
            return f"{self.dataset}: {spend} spent, unread but cheap{caveat}"
        return f"{self.dataset}: {spend} spent and unread{caveat}"


def value(
    totals: Mapping[DatasetId, LifetimeCost],
    reads: Mapping[DatasetId, int],
    *,
    threshold: float,
    window: timedelta | None = None,
) -> list[ValueFinding]:
    """Set each dataset's lifetime cost against whether anybody reads it.

    `threshold` is the spend above which an unread dataset is worth a person's time.
    It has no default: the right number is a fraction of a budget this module cannot
    see, and a default would be a made-up figure that people would leave alone.

    `reads` is a plain count per dataset, and **which count is the caller's choice**.
    `observe.usage.read_counts(stats, people_only=True)` discounts scheduled jobs,
    matching what `retirement_candidates` does; passing raw counts here while
    discounting them there is how an intermediate table maintained by its own pipeline
    ends up looking read.

    Sorted most-expensive-first among the actionable ones, because a review list
    nobody reaches the bottom of should have the money at the top.
    """
    out: list[ValueFinding] = []
    for ds in sorted(set(totals) | set(reads), key=str):
        total = totals.get(ds)
        read_count = reads.get(ds, 0)

        if total is None or not total.is_measured:
            verdict = Verdict.UNMEASURED
        elif read_count > 0:
            verdict = Verdict.EARNING
        elif total.spend >= threshold:
            verdict = Verdict.REVIEW
        else:
            verdict = Verdict.CHEAP_AND_QUIET

        out.append(
            ValueFinding(
                dataset=ds,
                verdict=verdict,
                spend=total.spend if total and total.is_measured else None,
                reads=read_count,
                window=window,
            )
        )
    return sorted(out, key=lambda f: (not f.is_actionable, -(f.spend or 0.0), str(f.dataset)))


def actionable(findings: Sequence[ValueFinding]) -> list[ValueFinding]:
    """Only the findings recommending a review."""
    return [f for f in findings if f.is_actionable]


def unmeasured(findings: Sequence[ValueFinding]) -> list[DatasetId]:
    """Datasets with no cost history, which is the gap to close before trusting the rest."""
    return [f.dataset for f in findings if f.verdict is Verdict.UNMEASURED]


def summarize(findings: Sequence[ValueFinding], *, limit: int = 10) -> str:
    """A review list, money first, with the blind spot stated once rather than per line."""
    review = actionable(findings)
    missing = unmeasured(findings)
    if not review:
        head = "no dataset is both unread and above the threshold"
    else:
        head = f"{len(review)} dataset(s) unread and above the threshold:"
    lines = [head]
    lines.extend(f"    {f}" for f in review[:limit])
    if len(review) > limit:
        lines.append(f"    +{len(review) - limit} more")
    if missing:
        lines.append(
            f"    {len(missing)} dataset(s) have no cost history and were not judged; "
            "unmeasured is not free"
        )
    lines.append(
        "    cost is measured, usage is observed — a table read once a year for a "
        "filing looks identical here to a dead one"
    )
    return "\n".join(lines)
