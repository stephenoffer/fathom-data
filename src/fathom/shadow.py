"""Shadow mode: run the planner alongside a full rebuild and grade it.

This is the adoption strategy, not a debugging aid. Nobody should let a new tool
decide what *not* to rebuild on the strength of a README. So run both, and publish
two numbers per dataset:

    savings  how many partitions the planner would have skipped
    missed   how many it called clean that the full rebuild proved dirty

`missed` must be zero. It is the direct, empirical test of the soundness invariant,
and reporting it honestly — including when it is not zero — is the whole point. A
tool that publishes its own failure rate is one people will eventually trust to
apply.

Ordering matters. The target tables must still hold the *pre-change* build when this
runs, so the sequence is: source data lands, plan, shadow, and only then apply. If
you apply first there is nothing left to compare against.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from .graph import InvalidationPlan
from .store import ShadowObservation, Store
from .types import DatasetId, KeyPredicate, covered_by

__all__ = ["Fingerprinter", "ShadowReport", "ShadowResult", "compare", "run"]


@runtime_checkable
class Fingerprinter(Protocol):
    """What shadow mode needs from an engine: content hashes and a full rebuild."""

    def fingerprints(self, dataset: DatasetId) -> dict[KeyPredicate, str]: ...

    def full_rebuild(self, dataset: DatasetId) -> str: ...


@dataclass(frozen=True)
class ShadowResult:
    """How the plan graded against ground truth for one dataset."""

    dataset: DatasetId
    planned: frozenset[KeyPredicate] = frozenset()
    actual: frozenset[KeyPredicate] = frozenset()
    missed: frozenset[KeyPredicate] = frozenset()
    wasted: frozenset[KeyPredicate] = frozenset()
    total: int = 0

    @property
    def is_sound(self) -> bool:
        """False means the planner would have served stale data. Nothing else matters."""
        return not self.missed

    @property
    def savings(self) -> float:
        """Fraction of the dataset's partitions the plan avoided rebuilding."""
        if self.total <= 0:
            return 0.0
        return max(0.0, 1.0 - len(self.planned) / self.total)

    @property
    def precision(self) -> float:
        """Fraction of planned partitions that genuinely needed rebuilding."""
        if not self.planned:
            return 1.0
        return 1.0 - len(self.wasted) / len(self.planned)

    def __str__(self) -> str:
        verdict = "SOUND" if self.is_sound else f"UNSOUND ({len(self.missed)} missed)"
        return (
            f"{self.dataset}: {verdict}  "
            f"planned={len(self.planned)} actual={len(self.actual)} "
            f"total={self.total} savings={self.savings:.0%} precision={self.precision:.0%}"
        )


def compare(
    dataset: DatasetId,
    *,
    planned: Iterable[KeyPredicate],
    actual: Iterable[KeyPredicate],
    total: int,
) -> ShadowResult:
    """Grade one plan against the partitions that actually changed."""
    planned_set = frozenset(planned)
    actual_set = frozenset(actual)

    # A partition is missed when no planned predicate covers it. Coverage, not
    # equality: a planned `dt=ANY` legitimately covers every concrete partition.
    missed = frozenset(k for k in actual_set if not covered_by(planned_set, k))
    # Wasted the other way round: a planned predicate covering nothing that changed.
    wasted = frozenset(p for p in planned_set if not any(covered_by([p], a) for a in actual_set))

    return ShadowResult(
        dataset=dataset,
        planned=planned_set,
        actual=actual_set,
        missed=missed,
        wasted=wasted,
        total=max(total, len(actual_set)),
    )


@dataclass
class ShadowReport:
    """All results from one shadow run."""

    results: list[ShadowResult] = field(default_factory=list)
    observed: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_sound(self) -> bool:
        return all(r.is_sound for r in self.results)

    @property
    def missed_total(self) -> int:
        return sum(len(r.missed) for r in self.results)

    @property
    def savings(self) -> float:
        total = sum(r.total for r in self.results)
        planned = sum(len(r.planned) for r in self.results)
        return 0.0 if total == 0 else max(0.0, 1.0 - planned / total)

    def summary(self) -> str:
        if not self.results:
            return "shadow: nothing compared"
        lines = [str(r) for r in self.results]
        headline = (
            f"shadow: {'SOUND' if self.is_sound else 'UNSOUND'} across "
            f"{len(self.results)} dataset(s), {self.savings:.0%} of partitions skipped"
        )
        if not self.is_sound:
            headline += f", {self.missed_total} partition(s) wrongly called clean"
        return "\n".join([headline, *lines])

    def persist(self, store: Store) -> None:
        for r in self.results:
            store.record_shadow(
                ShadowObservation(
                    dataset=r.dataset,
                    observed=self.observed,
                    planned=len(r.planned),
                    actual=len(r.actual),
                    missed=len(r.missed),
                    total=r.total,
                )
            )


def run(
    engine: Fingerprinter,
    plan: InvalidationPlan,
    datasets: Sequence[DatasetId],
    *,
    store: Store | None = None,
) -> ShadowReport:
    """Fingerprint, full-rebuild, fingerprint again, and grade the plan.

    `datasets` should be the models the engine can rebuild — typically everything
    downstream of the sources, not the sources themselves.
    """
    report = ShadowReport()

    for dataset in datasets:
        before = engine.fingerprints(dataset)
        engine.full_rebuild(dataset)
        after = engine.fingerprints(dataset)

        # Changed, appeared, or vanished. All three are things a plan had to predict.
        seen = set(before) | set(after)
        touched = {key for key in seen if before.get(key) != after.get(key)}

        report.results.append(
            compare(
                dataset,
                planned=plan.partitions(dataset),
                actual=touched,
                total=len(seen),
            )
        )

    if store is not None:
        report.persist(store)
    return report
