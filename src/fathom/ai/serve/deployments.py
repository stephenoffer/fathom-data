"""Where users actually meet a model: endpoints, traffic splits, and rollback.

The graph currently ends at the model. Everything between a model and a user —
which variant is deployed, what share of traffic it takes, whether it was quantized
on the way — changes what people see and is invisible.

Two things here are worth more than the bookkeeping.

**A quantization regression is narrow and aggregate metrics miss it.** Int4 drops
perplexity by half a point and destroys one capability. `regression_report`
partitions by capability and reports the collapsed slice, because a single number
moving 0.4% is exactly what ships a broken model.

**A rollback needs somewhere to roll back to.** `can_rollback` refuses when the
previous variant has been garbage-collected, which is the moment people discover it
rather than the moment they planned for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from ...core.types import DatasetId
from ..assets import deployment as deployment_asset
from ..assets import model

__all__ = [
    "CapabilityResult",
    "Deployment",
    "DeploymentState",
    "RegressionReport",
    "RolloutStrategy",
    "TrafficSplit",
    "Variant",
    "active_variants",
    "can_rollback",
    "deployment_edges",
    "is_canary",
    "promote",
    "regression_report",
    "rollback",
    "rollout_plan",
    "traffic_to",
    "validate_split",
]


class DeploymentState(StrEnum):
    PENDING = "pending"
    CANARY = "canary"
    LIVE = "live"
    DRAINING = "draining"
    RETIRED = "retired"

    @property
    def serves_traffic(self) -> bool:
        return self in {DeploymentState.CANARY, DeploymentState.LIVE, DeploymentState.DRAINING}


class RolloutStrategy(StrEnum):
    IMMEDIATE = "immediate"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    SHADOW = "shadow"  # receives traffic, serves nobody; the safest and the slowest


@dataclass(frozen=True)
class Variant:
    """One served artefact behind an endpoint."""

    name: str
    source_model: str
    state: DeploymentState = DeploymentState.PENDING
    weight: float = 0.0
    quantization: str = ""
    hardware: str = ""
    deployed: datetime | None = None
    retained: bool = True  # false once the artefact is garbage-collected

    @property
    def is_quantized(self) -> bool:
        return bool(self.quantization)


@dataclass(frozen=True)
class TrafficSplit:
    """How requests divide between variants."""

    weights: Mapping[str, float]

    @property
    def total(self) -> float:
        return sum(self.weights.values())

    def share(self, variant: str) -> float:
        return self.weights.get(variant, 0.0)


@dataclass
class Deployment:
    """An endpoint and the variants behind it."""

    name: str
    variants: list[Variant] = field(default_factory=list)
    environment: str = "prod"
    strategy: RolloutStrategy = RolloutStrategy.CANARY

    @property
    def dataset(self) -> DatasetId:
        return deployment_asset(self.name, environment=self.environment)

    def variant(self, name: str) -> Variant | None:
        return next((v for v in self.variants if v.name == name), None)


def active_variants(deployment: Deployment) -> list[Variant]:
    return [v for v in deployment.variants if v.state.serves_traffic]


def traffic_to(deployment: Deployment, variant: str) -> float:
    found = deployment.variant(variant)
    return found.weight if found and found.state.serves_traffic else 0.0


def is_canary(deployment: Deployment, variant: str) -> bool:
    found = deployment.variant(variant)
    return bool(found and found.state is DeploymentState.CANARY)


def validate_split(deployment: Deployment, *, tolerance: float = 1e-6) -> list[str]:
    """Problems that make a traffic split wrong rather than merely unusual."""
    problems: list[str] = []
    serving = active_variants(deployment)

    if not serving:
        problems.append("no variant is serving traffic")
        return problems

    total = sum(v.weight for v in serving)
    if abs(total - 1.0) > tolerance:
        problems.append(
            f"weights sum to {total:.6g}, not 1.0; some share of traffic is unrouted "
            "or double-counted"
        )
    for variant in serving:
        if variant.weight < 0:
            problems.append(f"{variant.name} has negative weight {variant.weight}")

    live = [v for v in serving if v.state is DeploymentState.LIVE]
    if len(live) > 1 and deployment.strategy is RolloutStrategy.BLUE_GREEN:
        problems.append(
            f"{len(live)} variants are LIVE under a blue/green strategy, which expects one"
        )
    return problems


def deployment_edges(deployment: Deployment) -> list[tuple[DatasetId, DatasetId]]:
    """Each serving variant's model feeds the endpoint.

    Without these the graph stops at the model, and "which model is answering
    production traffic" is unanswerable from lineage.
    """
    target = deployment.dataset
    return [(model(v.source_model), target) for v in deployment.variants if v.state.serves_traffic]


def rollout_plan(
    deployment: Deployment, variant: str, *, steps: Sequence[float] = (0.01, 0.1, 0.5, 1.0)
) -> list[Mapping[str, float]]:
    """Successive traffic splits for promoting one variant.

    Each step names the full split rather than a delta, so a rollout halted midway
    leaves a state somebody can read rather than one they have to reconstruct.
    """
    others = [v for v in deployment.variants if v.name != variant and v.state.serves_traffic]
    plan: list[Mapping[str, float]] = []
    for share in steps:
        remaining = max(0.0, 1.0 - share)
        split: dict[str, float] = {variant: round(share, 6)}
        if others:
            each = remaining / len(others)
            split.update({v.name: round(each, 6) for v in others})
        plan.append(split)
    return plan


def promote(deployment: Deployment, variant: str, *, at: datetime | None = None) -> Deployment:
    """Take a variant to full traffic, draining the rest."""
    found = deployment.variant(variant)
    if found is None:
        raise KeyError(f"no variant {variant!r} in {deployment.name}")

    updated: list[Variant] = []
    for v in deployment.variants:
        if v.name == variant:
            updated.append(
                Variant(
                    name=v.name,
                    source_model=v.source_model,
                    state=DeploymentState.LIVE,
                    weight=1.0,
                    quantization=v.quantization,
                    hardware=v.hardware,
                    deployed=at or datetime.now(UTC),
                    retained=v.retained,
                )
            )
        elif v.state.serves_traffic:
            updated.append(
                Variant(
                    name=v.name,
                    source_model=v.source_model,
                    state=DeploymentState.DRAINING,
                    weight=0.0,
                    quantization=v.quantization,
                    hardware=v.hardware,
                    deployed=v.deployed,
                    retained=v.retained,
                )
            )
        else:
            updated.append(v)
    deployment.variants = updated
    return deployment


def can_rollback(deployment: Deployment, to: str) -> tuple[bool, str]:
    """Whether a previous variant is still there to roll back to.

    Checked before a rollout rather than during a rollback. A retired variant whose
    artefact was garbage-collected is discovered at exactly the wrong moment.
    """
    found = deployment.variant(to)
    if found is None:
        return False, f"no variant {to!r} in this deployment"
    if not found.retained:
        return False, (
            f"{to} was garbage-collected; there is no artefact to roll back to. "
            "Retain the previous variant for the length of your rollback window."
        )
    if found.state is DeploymentState.RETIRED and not found.retained:
        return False, f"{to} is retired and no longer available"
    return True, ""


def rollback(deployment: Deployment, to: str, *, at: datetime | None = None) -> Deployment:
    """Return traffic to a previous variant, refusing when it is gone."""
    ok, why = can_rollback(deployment, to)
    if not ok:
        raise RuntimeError(why)
    return promote(deployment, to, at=at)


# -- quantization regression ---------------------------------------------------


@dataclass(frozen=True)
class CapabilityResult:
    """One capability's score before and after a change."""

    capability: str
    before: float
    after: float
    samples: int = 0

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def relative(self) -> float:
        return 0.0 if self.before == 0 else self.delta / abs(self.before)


@dataclass(frozen=True)
class RegressionReport:
    """Whether a compressed variant is safe to serve.

    Aggregate and per-capability are separated because the aggregate is what looks
    fine and the per-capability is what shipped broken.
    """

    variant: str
    baseline: str
    aggregate_delta: float
    capabilities: tuple[CapabilityResult, ...]
    threshold: float

    @property
    def collapsed(self) -> tuple[CapabilityResult, ...]:
        return tuple(c for c in self.capabilities if c.relative < -self.threshold)

    @property
    def safe(self) -> bool:
        return not self.collapsed

    def summary(self) -> str:
        lines = [
            f"{self.variant} vs {self.baseline}: aggregate {self.aggregate_delta:+.4f}, "
            f"{'SAFE' if self.safe else 'REGRESSED'}"
        ]
        if self.safe and self.capabilities:
            lines.append("  no capability fell more than the threshold")
        for result in self.collapsed:
            lines.append(
                f"  {result.capability}: {result.before:.4f} -> {result.after:.4f} "
                f"({result.relative:+.1%}) over {result.samples} sample(s)"
            )
        if not self.safe:
            lines.append(
                "  The aggregate can look fine while one capability collapses. That is "
                "the failure mode this check exists for."
            )
        return "\n".join(lines)


def regression_report(
    variant: str,
    baseline: str,
    results: Iterable[CapabilityResult],
    *,
    threshold: float = 0.05,
) -> RegressionReport:
    """Compare a variant against its baseline, per capability.

    `threshold` is a *relative* drop, so a capability scoring 0.9 and one scoring
    0.3 are held to the same proportional standard rather than the same absolute one.
    """
    found = tuple(results)
    weighted = sum(c.delta * max(c.samples, 1) for c in found)
    total = sum(max(c.samples, 1) for c in found)
    return RegressionReport(
        variant=variant,
        baseline=baseline,
        aggregate_delta=weighted / total if total else 0.0,
        capabilities=found,
        threshold=threshold,
    )
