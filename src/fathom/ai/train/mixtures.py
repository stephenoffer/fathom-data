"""Corpus sampling weights: the highest-leverage pretraining decision nobody versions.

Deciding a corpus is 40% web, 30% code, 20% books, 10% math determines more about the
resulting model than most architecture choices, and it lives in a YAML file that is
never linked to the run it produced or the eval it moved.

What this makes answerable:

**Where a weight came from.** `Component.rationale` and `provenance` carry the run or
ablation that argued for a number. A weight with no rationale is not a decision, it is
an inheritance — and the honest report says so rather than implying somebody chose it.

**What a re-mixture actually costs.** `remix_plan` prices a weight change in tokens
that must be re-sampled, and separates the components that merely change proportion
from the ones that need more data than exists. Upweighting math to 20% when the math
corpus holds 3% of the tokens means *epoching* it seven times, which is a different
decision from sampling it more.

**Whether the mixture was ever tested.** `untested` names components no ablation ever
varied. Those are the weights carried forward from whatever the last team did.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ...core.types import DatasetId
from ..assets import mixture as mixture_asset

__all__ = [
    "Component",
    "EpochPressure",
    "Mixture",
    "RemixPlan",
    "WeightChange",
    "component_edges",
    "effective_epochs",
    "epoch_pressure",
    "mixture_edges",
    "normalize",
    "remix_plan",
    "token_budget",
    "unattributed",
    "untested",
    "validate",
]


@dataclass(frozen=True)
class Component:
    """One source and the share of training tokens it takes.

    `available_tokens` is what the source actually holds. Without it a mixture cannot
    tell "sample this more" from "read this seven times", and those have different
    effects on a model.
    """

    source: str
    weight: float
    available_tokens: int = 0
    rationale: str = ""
    decided_by: str = ""  # the run, ablation, or person that argued for this weight
    dataset: DatasetId | None = None

    @property
    def is_attributed(self) -> bool:
        """A weight with no rationale and no author is an inheritance, not a decision."""
        return bool(self.rationale or self.decided_by)


@dataclass(frozen=True)
class Mixture:
    """A named set of sampling weights."""

    name: str
    components: tuple[Component, ...] = ()
    total_tokens: int = 0
    registry: str = "local"
    parent: str = ""  # the mixture this was derived from

    @property
    def asset(self) -> DatasetId:
        return mixture_asset(self.name, registry=self.registry)

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.components)

    def component(self, source: str) -> Component | None:
        return next((c for c in self.components if c.source == source), None)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(c.source for c in self.components)


def normalize(mixture: Mixture) -> Mixture:
    """Rescale weights to sum to one, preserving proportions.

    Weights are written by hand and rarely sum to exactly 1.0. Normalising is right;
    silently treating 0.95 as 1.0 is not, because the missing 5% has to come from
    somewhere and where it comes from changes the model.
    """
    total = mixture.total_weight
    if total <= 0:
        raise ValueError(
            f"mixture {mixture.name!r} has total weight {total}; there is nothing to "
            "normalise and nothing would be sampled"
        )
    return Mixture(
        name=mixture.name,
        components=tuple(
            Component(
                source=c.source,
                weight=c.weight / total,
                available_tokens=c.available_tokens,
                rationale=c.rationale,
                decided_by=c.decided_by,
                dataset=c.dataset,
            )
            for c in mixture.components
        ),
        total_tokens=mixture.total_tokens,
        registry=mixture.registry,
        parent=mixture.parent,
    )


def validate(mixture: Mixture, *, tolerance: float = 1e-6) -> list[str]:
    """Problems that make a mixture wrong rather than merely unusual."""
    problems: list[str] = []
    if not mixture.components:
        problems.append("mixture has no components; nothing would be sampled")
        return problems

    total = mixture.total_weight
    if abs(total - 1.0) > tolerance:
        problems.append(
            f"weights sum to {total:.6g}, not 1.0 — normalise explicitly rather than "
            "leaving the remainder to whatever the sampler does with it"
        )
    for component in mixture.components:
        if component.weight < 0:
            problems.append(f"{component.source} has negative weight {component.weight}")
        elif component.weight == 0:
            problems.append(
                f"{component.source} has weight 0 and contributes nothing; remove it "
                "rather than leaving it to imply it was considered"
            )

    duplicates = {s for s in mixture.sources if mixture.sources.count(s) > 1}
    problems.extend(
        f"{source} appears more than once; its shares would silently add"
        for source in sorted(duplicates)
    )
    return problems


def token_budget(mixture: Mixture, total_tokens: int | None = None) -> dict[str, int]:
    """How many tokens each component contributes at a given training budget."""
    budget = total_tokens if total_tokens is not None else mixture.total_tokens
    return {c.source: round(c.weight * budget) for c in mixture.components}


# -- epoching ------------------------------------------------------------------


@dataclass(frozen=True)
class EpochPressure:
    """How many times a component must be re-read to hit its weight.

    Anything above one means repetition, which is a genuine decision with a known
    cost, not a rounding detail.
    """

    source: str
    required_tokens: int
    available_tokens: int

    @property
    def epochs(self) -> float:
        if self.available_tokens <= 0:
            return float("inf")
        return self.required_tokens / self.available_tokens

    @property
    def repeats(self) -> bool:
        return self.epochs > 1.0

    @property
    def is_unknown(self) -> bool:
        """No token count recorded — we cannot say, and must not imply otherwise."""
        return self.available_tokens <= 0


def effective_epochs(component: Component, total_tokens: int) -> EpochPressure:
    return EpochPressure(
        source=component.source,
        required_tokens=round(component.weight * total_tokens),
        available_tokens=component.available_tokens,
    )


def epoch_pressure(
    mixture: Mixture, total_tokens: int | None = None, *, limit: float = 1.0
) -> list[EpochPressure]:
    """Components that must be repeated to reach their weight, worst first.

    Components with no recorded size are reported too, marked unknown, because a
    silent omission reads as "this one is fine".
    """
    budget = total_tokens if total_tokens is not None else mixture.total_tokens
    pressures = [effective_epochs(c, budget) for c in mixture.components]
    return sorted(
        (p for p in pressures if p.is_unknown or p.epochs > limit),
        key=lambda p: -p.epochs,
    )


# -- provenance ----------------------------------------------------------------


def unattributed(mixture: Mixture) -> list[str]:
    """Components whose weight nobody has justified.

    Not a defect on its own — but a mixture where most weights are unattributed is
    one nobody chose, and saying so is more useful than implying deliberation.
    """
    return [c.source for c in mixture.components if not c.is_attributed]


def untested(mixture: Mixture, ablations: Iterable[Mapping[str, float]]) -> list[str]:
    """Components no ablation ever varied.

    An ablation that holds a weight fixed says nothing about it. These are the numbers
    carried forward from whatever the previous team happened to use.
    """
    varied: set[str] = set()
    seen: dict[str, set[float]] = {}
    for ablation in ablations:
        for source, weight in ablation.items():
            seen.setdefault(source, set()).add(weight)
    for source, weights in seen.items():
        if len(weights) > 1:
            varied.add(source)
    return [c.source for c in mixture.components if c.source not in varied]


def component_edges(mixture: Mixture) -> list[tuple[DatasetId, DatasetId]]:
    """Each component's dataset feeds the mixture.

    Components with no `dataset` produce no edge — a mixture naming a source fathom
    cannot identify is exactly the gap worth seeing, not one to paper over with a
    synthesised identity.
    """
    target = mixture.asset
    return [(c.dataset, target) for c in mixture.components if c.dataset is not None]


def mixture_edges(
    mixture: Mixture, trained: Iterable[DatasetId]
) -> list[tuple[DatasetId, DatasetId]]:
    """The mixture feeds every model trained under it."""
    source = mixture.asset
    return [(source, model) for model in trained]


# -- re-mixture ----------------------------------------------------------------


@dataclass(frozen=True)
class WeightChange:
    """One component's weight before and after."""

    source: str
    before: float
    after: float
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def relative(self) -> float:
        return 0.0 if self.before == 0 else self.delta / self.before

    @property
    def added(self) -> bool:
        return self.before == 0 and self.after > 0

    @property
    def dropped(self) -> bool:
        return self.before > 0 and self.after == 0


@dataclass(frozen=True)
class RemixPlan:
    """What changing a mixture costs.

    `resample_tokens` is the honest price: the tokens that must be drawn differently,
    not the size of the diff in the config file.
    """

    before: str
    after: str
    changes: tuple[WeightChange, ...]
    resample_tokens: int
    newly_repeated: tuple[EpochPressure, ...] = ()

    @property
    def unchanged(self) -> bool:
        return not any(c.delta for c in self.changes)

    @property
    def significant(self) -> tuple[WeightChange, ...]:
        return tuple(sorted(self.changes, key=lambda c: -abs(c.delta)))

    def summary(self) -> str:
        if self.unchanged:
            return f"{self.before} -> {self.after}: identical weights"
        lines = [f"{self.before} -> {self.after}: {self.resample_tokens:,} token(s) resampled"]
        for change in self.significant:
            if not change.delta:
                continue
            mark = " (new)" if change.added else " (dropped)" if change.dropped else ""
            lines.append(
                f"  {change.source}: {change.before:.3f} -> {change.after:.3f} "
                f"({change.delta:+.3f}){mark}"
            )
        for pressure in self.newly_repeated:
            if pressure.is_unknown:
                lines.append(f"  {pressure.source}: size unrecorded, so its epoch count is unknown")
            else:
                lines.append(
                    f"  {pressure.source} now needs {pressure.epochs:.1f} epochs — this "
                    "repeats data rather than sampling more of it, which is a different "
                    "decision with a different effect"
                )
        return "\n".join(lines)


def remix_plan(before: Mixture, after: Mixture, total_tokens: int | None = None) -> RemixPlan:
    """Price a weight change in tokens that must be re-drawn."""
    budget = total_tokens if total_tokens is not None else after.total_tokens or before.total_tokens
    old_budget = token_budget(before, budget)
    new_budget = token_budget(after, budget)

    changes: list[WeightChange] = []
    for source in sorted(set(before.sources) | set(after.sources)):
        old, new = before.component(source), after.component(source)
        changes.append(
            WeightChange(
                source=source,
                before=old.weight if old else 0.0,
                after=new.weight if new else 0.0,
                tokens_before=old_budget.get(source, 0),
                tokens_after=new_budget.get(source, 0),
            )
        )

    # Only the increases need new sampling; a component that shrank simply contributes
    # less, which costs nothing to do.
    resample = sum(max(0, c.tokens_after - c.tokens_before) for c in changes)

    was_repeating = {p.source for p in epoch_pressure(before, budget)}
    newly = tuple(p for p in epoch_pressure(after, budget) if p.source not in was_repeating)

    return RemixPlan(
        before=before.name,
        after=after.name,
        changes=tuple(changes),
        resample_tokens=resample,
        newly_repeated=newly,
    )
