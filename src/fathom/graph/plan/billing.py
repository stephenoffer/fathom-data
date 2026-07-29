"""What the warehouse actually charged, against what the model said it would.

`cost` prices a plan from declared rates. Every savings figure this library produces
rests on those rates, and nothing checks them. A model that is three times off does
not fail — it produces confident numbers in the wrong currency, and the first person
to notice is whoever compares a quarterly saving to the actual invoice and finds they
disagree.

So: read the bill.

Every major warehouse exposes one. Snowflake has `WAREHOUSE_METERING_HISTORY`,
BigQuery has `INFORMATION_SCHEMA.JOBS` with `total_bytes_billed`, Databricks has
`system.billing.usage`. This module does not query any of them — that is an adapter's
job, and a library that needs credentials to be imported is a library people vendor
around. It takes the rows and does the arithmetic.

**The attribution problem, stated rather than papered over.** Warehouse billing is
almost always per-warehouse or per-job, not per-dataset. Splitting a warehouse's bill
across the datasets built on it needs query tags nobody maintains, so this module
reconciles at the level the billing data actually supports:

- **Aggregate reconciliation is sound.** Total modelled against total billed over the
  same window is a real comparison, and a bias factor derived from it is a real
  correction.
- **Per-dataset reconciliation requires attribution** and is only produced when the
  billing rows carry it. A per-dataset variance computed by apportioning an
  unattributed total would be the cost model's own assumptions handed back as
  evidence for themselves.

`calibrate` returns a corrected model rather than mutating one, and it refuses on too
little data. A bias factor from a single day is a rounding artifact with a decimal
point.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ...core.types import DatasetId
from ...core.util.clock import as_utc
from .cost import CostModel

__all__ = [
    "BillingRecord",
    "Reconciliation",
    "attributed_share",
    "bias",
    "by_period",
    "by_source",
    "calibrate",
    "coverage_days",
    "drifted_datasets",
    "in_window",
    "reconcile",
    "total_billed",
    "unattributed",
    "within_tolerance",
]

# Below this many billing periods, a bias factor is noise. Two weeks of daily rows is
# the smallest window where a weekday/weekend split does not dominate the average.
MIN_PERIODS_TO_CALIBRATE = 14

# How far modelled may sit from billed before the model is worth revisiting. Cost
# models are estimates and 20% is a useful estimate; 200% is a different currency.
DEFAULT_TOLERANCE = 0.2


@dataclass(frozen=True)
class BillingRecord:
    """One line of what the warehouse actually charged.

    `dataset` is present only when the billing row carries attribution — a query tag,
    a job label, a dedicated warehouse. Absent means the charge is real and cannot be
    assigned, which is the common case and is reported rather than apportioned.
    """

    period: datetime
    amount: float
    source: str = ""  # warehouse, project, or workspace the charge came from
    dataset: DatasetId | None = None

    @property
    def is_attributed(self) -> bool:
        return self.dataset is not None

    def __str__(self) -> str:
        where = f" [{self.source}]" if self.source else ""
        to = f" -> {self.dataset}" if self.dataset else " (unattributed)"
        return f"{as_utc(self.period).date().isoformat()}{where}: {self.amount:,.2f}{to}"


def total_billed(records: Iterable[BillingRecord]) -> float:
    """Everything charged across the given rows."""
    return sum(r.amount for r in records)


def unattributed(records: Iterable[BillingRecord]) -> list[BillingRecord]:
    """Charges that carry no dataset. Usually most of them."""
    return [r for r in records if not r.is_attributed]


def attributed_share(records: Sequence[BillingRecord]) -> float:
    """Fraction of the billed amount that carries a dataset.

    The number that decides whether per-dataset reconciliation means anything. Low is
    the norm and is not a defect — it is a fact about warehouse billing, and the fix
    is query tagging, not arithmetic.
    """
    total = total_billed(records)
    if total <= 0:
        return 0.0
    return sum(r.amount for r in records if r.is_attributed) / total


def bias(modelled: float, billed: float) -> float | None:
    """How far the model sits from the bill, as a signed relative error.

    `0.0` means the model was right. `+0.5` means it predicted 50% more than was
    charged. Returns `None` when nothing was billed, because a ratio against zero is
    not a large error — it is no answer.
    """
    if billed <= 0:
        return None
    return (modelled - billed) / billed


@dataclass
class Reconciliation:
    """Modelled cost set against what was actually charged."""

    modelled: float
    billed: float
    periods: int
    attributed: float = 0.0
    per_dataset: dict[DatasetId, tuple[float, float]] = field(default_factory=dict)
    window: tuple[datetime, datetime] | None = None

    @property
    def bias(self) -> float | None:
        """Signed relative error of the model against the bill."""
        return bias(self.modelled, self.billed)

    @property
    def correction(self) -> float | None:
        """What to multiply the model's rates by to match the bill.

        `None` when nothing was modelled — a correction factor against zero would be
        infinite, and the honest reading is that the model priced nothing rather than
        that it was infinitely wrong.
        """
        if self.modelled <= 0:
            return None
        return self.billed / self.modelled

    @property
    def is_reliable(self) -> bool:
        """Enough periods to believe the comparison."""
        return self.periods >= MIN_PERIODS_TO_CALIBRATE

    def within(self, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        found = self.bias
        return found is not None and abs(found) <= tolerance

    def summary(self) -> str:
        lines = [
            f"modelled {self.modelled:,.2f} against {self.billed:,.2f} billed "
            f"over {self.periods} period(s)"
        ]
        found = self.bias
        if found is None:
            lines.append("    nothing was billed in this window; no comparison to make")
            return "\n".join(lines)

        direction = "over" if found > 0 else "under"
        lines.append(f"    the model {direction}-predicts by {abs(found):.0%}")
        if self.correction is not None and not self.within():
            lines.append(
                f"    multiply the model's rates by {self.correction:.3f} to match the bill"
            )
        if not self.is_reliable:
            lines.append(
                f"    only {self.periods} period(s) — below {MIN_PERIODS_TO_CALIBRATE}, a "
                "bias factor here is a rounding artifact with a decimal point"
            )
        if self.attributed < 1.0:
            lines.append(
                f"    {self.attributed:.0%} of the bill carries a dataset. The rest is "
                "real and unassignable; per-dataset figures cover only the attributed "
                "part, because apportioning the rest would hand the model's own "
                "assumptions back as evidence for themselves"
            )
        return "\n".join(lines)


def reconcile(
    modelled: float,
    records: Sequence[BillingRecord],
    *,
    per_dataset_modelled: Mapping[DatasetId, float] | None = None,
) -> Reconciliation:
    """Compare a modelled total against billing rows covering the same window.

    `per_dataset_modelled` enables the per-dataset half, which is produced **only for
    datasets the billing rows actually attribute**. A dataset the model priced but the
    bill never named is absent from `per_dataset` rather than shown against zero.
    """
    periods = {as_utc(r.period).date() for r in records}
    result = Reconciliation(
        modelled=modelled,
        billed=total_billed(records),
        periods=len(periods),
        attributed=attributed_share(records),
    )
    if records:
        moments = sorted(as_utc(r.period) for r in records)
        result.window = (moments[0], moments[-1])

    if per_dataset_modelled:
        billed_by_dataset: dict[DatasetId, float] = {}
        for record in records:
            if record.dataset is not None:
                billed_by_dataset[record.dataset] = (
                    billed_by_dataset.get(record.dataset, 0.0) + record.amount
                )
        for dataset, actual in billed_by_dataset.items():
            predicted = per_dataset_modelled.get(dataset)
            if predicted is not None:
                result.per_dataset[dataset] = (predicted, actual)

    return result


def within_tolerance(result: Reconciliation, *, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """True when the model is close enough to leave alone."""
    return result.within(tolerance)


def calibrate(
    model: CostModel, result: Reconciliation, *, tolerance: float = DEFAULT_TOLERANCE
) -> CostModel | None:
    """A copy of `model` with every rate scaled to match the bill.

    Returns `None` — rather than an uncorrected copy — in the three cases where a
    correction would be worse than none: too few periods to believe, nothing modelled
    to scale, or a model already inside tolerance. A caller that gets `None` should
    keep what it has, and the distinct reasons are on the `Reconciliation`.

    Scaling every rate by one factor is deliberately blunt. The bill does not say
    which basis was wrong, so apportioning the correction across per-partition and
    per-byte rates would be a guess dressed as a calibration.
    """
    if not result.is_reliable:
        return None
    correction = result.correction
    if correction is None or result.within(tolerance):
        return None
    return CostModel(
        price_per_partition=model.price_per_partition * correction,
        price_per_tb_scanned=model.price_per_tb_scanned * correction,
        price_per_million_tokens=model.price_per_million_tokens * correction,
        price_per_compute_hour=model.price_per_compute_hour * correction,
        kwh_per_tb_scanned=model.kwh_per_tb_scanned,
        grid_intensity=model.grid_intensity,
    )


def drifted_datasets(
    result: Reconciliation, *, tolerance: float = DEFAULT_TOLERANCE
) -> list[tuple[DatasetId, float]]:
    """Attributed datasets whose modelled cost is outside tolerance, worst first."""
    out: list[tuple[DatasetId, float]] = []
    for dataset, (predicted, actual) in result.per_dataset.items():
        found = bias(predicted, actual)
        if found is not None and abs(found) > tolerance:
            out.append((dataset, found))
    out.sort(key=lambda pair: (-abs(pair[1]), str(pair[0])))
    return out


def by_period(records: Iterable[BillingRecord]) -> dict[datetime, float]:
    """Charges totalled per billing period, for a trend."""
    out: dict[datetime, float] = {}
    for record in records:
        moment = as_utc(record.period)
        out[moment] = out.get(moment, 0.0) + record.amount
    return dict(sorted(out.items()))


def by_source(records: Iterable[BillingRecord]) -> dict[str, float]:
    """Charges totalled per warehouse, project, or workspace."""
    out: dict[str, float] = {}
    for record in records:
        key = record.source or "(unnamed)"
        out[key] = out.get(key, 0.0) + record.amount
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def in_window(
    records: Iterable[BillingRecord], *, start: datetime, end: datetime
) -> list[BillingRecord]:
    """Rows inside a window, so a reconciliation compares like with like.

    The most common way an aggregate reconciliation goes wrong is comparing a month of
    modelled cost against six weeks of billing, which shows a 50% bias that is entirely
    the window.
    """
    lower, upper = as_utc(start), as_utc(end)
    return [r for r in records if lower <= as_utc(r.period) <= upper]


def coverage_days(records: Sequence[BillingRecord]) -> timedelta | None:
    """How long the billing rows span, for checking they match the modelled window."""
    if not records:
        return None
    moments = sorted(as_utc(r.period) for r in records)
    return moments[-1] - moments[0]
