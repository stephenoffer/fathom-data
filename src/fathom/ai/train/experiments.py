"""Training runs, sweeps, and the arithmetic that justifies a model's size.

`training.py` records what a model was trained *on*. This records the run: its
hyperparameters, the sweep it belonged to, the ablation it was measured against, and
the scaling-law fit that argued for spending the compute in the first place.

Three things here are worth more than the bookkeeping.

**Run comparison is a diff, not a spreadsheet.** When a model regresses, the first
question is what changed since the last run. Hyperparameters, data, code, and seed
are four different answers with four different remedies, and `compare` separates
them instead of reporting "something moved".

**Determinism is a claim that gets checked.** A run that says it is reproducible and
is not costs someone a week. `determinism_report` names the specific sources —
unseeded dataloader shuffling, non-deterministic kernels, mixed-precision reduction
order — rather than emitting a boolean.

**Scaling laws extrapolate with an error bar or not at all.** Fitting three points
and predicting two orders of magnitude out is how compute budgets get burned, so
`predict` reports the extrapolation factor and `is_extrapolating` is checked by the
CLI before anyone quotes the number.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from ...core.types import DatasetId
from ..assets import run as run_asset
from ..assets import sweep as sweep_asset

__all__ = [
    "Ablation",
    "DeterminismReport",
    "HyperparameterDiff",
    "Run",
    "RunComparison",
    "RunStatus",
    "ScalingLaw",
    "Sweep",
    "Trial",
    "ablation",
    "best_trial",
    "chinchilla_optimal",
    "compare",
    "compute_flops",
    "determinism_report",
    "diff_hyperparameters",
    "dominated",
    "fit_scaling_law",
    "gpu_hours",
    "is_regression",
    "loss_curve_summary",
    "nondeterminism_sources",
    "pareto_front",
    "predict_loss",
    "predict_tokens_needed",
    "run_edges",
    "sweep_summary",
    "throughput",
    "to_markdown",
    "tokens_seen",
    "trial_table",
]


class RunStatus(StrEnum):
    """Where a run ended up. Only `COMPLETED` runs are safe to compare against."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"  # spot reclaim; resumable, unlike failed
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }

    @property
    def is_comparable(self) -> bool:
        """A partial run's metrics are not comparable to a finished one's."""
        return self is RunStatus.COMPLETED


@dataclass(frozen=True)
class Run:
    """One training or evaluation execution."""

    name: str
    status: RunStatus = RunStatus.PENDING
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    inputs: tuple[DatasetId, ...] = ()
    outputs: tuple[DatasetId, ...] = ()
    seed: int | None = None
    code_version: str = ""
    started: datetime | None = None
    finished: datetime | None = None
    accelerator: str = ""
    accelerator_count: int = 0
    tokens: int = 0
    steps: int = 0
    sweep: str = ""
    tracker: str = "local"

    @property
    def dataset(self) -> DatasetId:
        return run_asset(self.name, tracker=self.tracker)

    @property
    def duration_hours(self) -> float:
        if self.started is None or self.finished is None:
            return 0.0
        return max(0.0, (self.finished - self.started).total_seconds() / 3600)

    def metric(self, name: str) -> float | None:
        return self.metrics.get(name)


@dataclass(frozen=True)
class Trial:
    """One run inside a sweep, with the varied parameters isolated."""

    run: Run
    varied: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.run.name


@dataclass
class Sweep:
    """A set of trials varying hyperparameters over a shared base."""

    name: str
    trials: list[Trial] = field(default_factory=list)
    objective: str = "loss"
    minimize: bool = True
    tracker: str = "local"

    @property
    def dataset(self) -> DatasetId:
        return sweep_asset(self.name, tracker=self.tracker)

    @property
    def completed(self) -> list[Trial]:
        return [t for t in self.trials if t.run.status.is_comparable]


def gpu_hours(run: Run) -> float:
    """Accelerator-hours consumed. The number every cost model starts from."""
    return run.duration_hours * max(run.accelerator_count, 0)


def tokens_seen(run: Run) -> int:
    return max(run.tokens, 0)


def throughput(run: Run) -> float:
    """Tokens per accelerator-second. Comparable across cluster sizes; raw tokens/s is not."""
    seconds = run.duration_hours * 3600 * max(run.accelerator_count, 1)
    return 0.0 if seconds <= 0 else run.tokens / seconds


def compute_flops(run: Run, *, parameters: int) -> float:
    """Approximate training FLOPs via the standard 6ND estimate.

    Six FLOPs per parameter per token covers forward and backward. It ignores
    attention's quadratic term, which is why this is an estimate and is named as one.
    """
    return 6.0 * max(parameters, 0) * max(run.tokens, 0)


# -- comparison ----------------------------------------------------------------


@dataclass(frozen=True)
class HyperparameterDiff:
    """What differed between two runs' configurations."""

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed: tuple[tuple[str, Any, Any], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def summary(self) -> str:
        if self.is_empty:
            return "identical hyperparameters"
        parts = []
        for key, before, after in self.changed:
            parts.append(f"{key}: {before} -> {after}")
        parts.extend(f"+{k}" for k in self.added)
        parts.extend(f"-{k}" for k in self.removed)
        return ", ".join(parts)


def diff_hyperparameters(before: Run, after: Run) -> HyperparameterDiff:
    """Compare two runs' configurations."""
    left, right = dict(before.hyperparameters), dict(after.hyperparameters)
    shared = set(left) & set(right)
    return HyperparameterDiff(
        added=tuple(sorted(set(right) - set(left))),
        removed=tuple(sorted(set(left) - set(right))),
        changed=tuple(sorted((k, left[k], right[k]) for k in shared if left[k] != right[k])),
    )


@dataclass(frozen=True)
class RunComparison:
    """Why two runs differ, separated by cause.

    The separation is the point. A metric change explained by a hyperparameter is a
    result; the same change with identical hyperparameters and different data is a
    data incident; with both identical it is nondeterminism, and the run that claims
    to be reproducible is lying.
    """

    before: Run
    after: Run
    hyperparameters: HyperparameterDiff
    metric_deltas: Mapping[str, float]
    inputs_changed: tuple[DatasetId, ...]
    code_changed: bool
    seed_changed: bool

    @property
    def unexplained(self) -> bool:
        """True when metrics moved and nothing that should explain it did."""
        moved = any(abs(v) > 1e-9 for v in self.metric_deltas.values())
        return (
            moved
            and self.hyperparameters.is_empty
            and not (self.inputs_changed or self.code_changed or self.seed_changed)
        )

    def summary(self) -> str:
        lines = [f"{self.before.name} -> {self.after.name}"]
        for name, delta in sorted(self.metric_deltas.items()):
            lines.append(f"  {name}: {delta:+.6g}")
        lines.append(f"  hyperparameters: {self.hyperparameters.summary()}")
        if self.inputs_changed:
            lines.append(f"  inputs changed: {len(self.inputs_changed)}")
        if self.code_changed:
            lines.append("  code changed")
        if self.seed_changed:
            lines.append("  seed changed")
        if self.unexplained:
            lines.append(
                "  UNEXPLAINED: metrics moved with identical config, data, code, and "
                "seed. Either the run is nondeterministic or something is untracked."
            )
        return "\n".join(lines)


def compare(before: Run, after: Run) -> RunComparison:
    """Diff two runs across every axis that could explain a metric change."""
    deltas = {
        name: after.metrics.get(name, 0.0) - value
        for name, value in before.metrics.items()
        if name in after.metrics
    }
    changed_inputs = tuple(sorted(set(after.inputs).symmetric_difference(before.inputs), key=str))
    return RunComparison(
        before=before,
        after=after,
        hyperparameters=diff_hyperparameters(before, after),
        metric_deltas=deltas,
        inputs_changed=changed_inputs,
        code_changed=bool(before.code_version and before.code_version != after.code_version),
        seed_changed=before.seed != after.seed,
    )


def is_regression(
    before: Run, after: Run, *, metric: str = "loss", minimize: bool = True, tolerance: float = 0.0
) -> bool:
    """Whether `after` is worse than `before` on one metric, beyond tolerance."""
    left, right = before.metrics.get(metric), after.metrics.get(metric)
    if left is None or right is None:
        return False
    delta = right - left
    return delta > tolerance if minimize else delta < -tolerance


# -- sweeps and ablations ------------------------------------------------------


def best_trial(sweep: Sweep) -> Trial | None:
    """The best completed trial by the sweep's objective."""
    finished = sweep.completed
    scored = [(t, t.run.metrics.get(sweep.objective)) for t in finished]
    usable = [(t, v) for t, v in scored if v is not None]
    if not usable:
        return None
    return (
        min(usable, key=lambda p: p[1])[0] if sweep.minimize else max(usable, key=lambda p: p[1])[0]
    )


def dominated(a: Mapping[str, float], b: Mapping[str, float], *, minimize: Sequence[str]) -> bool:
    """True when `b` is at least as good as `a` everywhere and better somewhere."""
    at_least_as_good = True
    strictly_better = False
    for name in minimize:
        left, right = a.get(name), b.get(name)
        if left is None or right is None:
            return False
        if right > left:
            at_least_as_good = False
        if right < left:
            strictly_better = True
    return at_least_as_good and strictly_better


def pareto_front(sweep: Sweep, *, objectives: Sequence[str]) -> list[Trial]:
    """Trials not dominated on every objective at once.

    Single-objective selection hides the trade-off that usually matters — the model
    that is 0.2% better and three times more expensive to serve.
    """
    finished = sweep.completed
    return [
        trial
        for trial in finished
        if not any(
            dominated(trial.run.metrics, other.run.metrics, minimize=objectives)
            for other in finished
            if other is not trial
        )
    ]


@dataclass(frozen=True)
class Ablation:
    """A controlled comparison: one change, everything else held."""

    name: str
    baseline: Run
    variant: Run
    changed: str

    @property
    def effect(self) -> Mapping[str, float]:
        return {
            metric: self.variant.metrics.get(metric, 0.0) - value
            for metric, value in self.baseline.metrics.items()
            if metric in self.variant.metrics
        }

    @property
    def is_controlled(self) -> bool:
        """True when exactly the named parameter differs.

        An uncontrolled ablation measures the sum of its changes and attributes it to
        one of them, which is how a team spends a quarter chasing the wrong lever.
        """
        diff = diff_hyperparameters(self.baseline, self.variant)
        touched = {k for k, _, _ in diff.changed} | set(diff.added) | set(diff.removed)
        return touched == {self.changed}


def ablation(name: str, baseline: Run, variant: Run, *, changed: str) -> Ablation:
    return Ablation(name=name, baseline=baseline, variant=variant, changed=changed)


def sweep_summary(sweep: Sweep) -> dict[str, Any]:
    """Counts, best trial, and objective spread."""
    finished = sweep.completed
    values = [v for v in (t.run.metrics.get(sweep.objective) for t in finished) if v is not None]
    best = best_trial(sweep)
    return {
        "sweep": sweep.name,
        "trials": len(sweep.trials),
        "completed": len(finished),
        "failed": sum(1 for t in sweep.trials if t.run.status is RunStatus.FAILED),
        "objective": sweep.objective,
        "best": best.name if best else None,
        "best_value": min(values)
        if values and sweep.minimize
        else (max(values) if values else None),
        "spread": (max(values) - min(values)) if len(values) > 1 else 0.0,
    }


def trial_table(sweep: Sweep) -> list[dict[str, Any]]:
    """A flat table of trials, for rendering or export."""
    rows = []
    for trial in sweep.trials:
        row: dict[str, Any] = {"trial": trial.name, "status": trial.run.status.value}
        row.update({f"hp.{k}": v for k, v in trial.varied.items()})
        row.update({f"metric.{k}": v for k, v in trial.run.metrics.items()})
        rows.append(row)
    return rows


# -- determinism ---------------------------------------------------------------

# Sources ordered by how often they are the actual cause.
_NONDETERMINISM = {
    "seed": "no seed recorded; the dataloader shuffle and init are unreproducible",
    "dataloader_workers": "multi-worker dataloading without a per-worker seed reorders batches",
    "cudnn_benchmark": "cuDNN autotuning picks different kernels per run",
    "tf32": "TF32 matmuls change reduction order and therefore results",
    "atomics": "atomic accumulation in custom kernels is order-dependent",
    "mixed_precision": "loss scaling interacts with reduction order",
    "distributed_allreduce": "allreduce order varies with topology and network timing",
    "code_version": "no code version recorded; the source is not pinned",
}


@dataclass(frozen=True)
class DeterminismReport:
    """Whether a run can be reproduced, and specifically what stops it."""

    run: str
    reproducible: bool
    sources: tuple[tuple[str, str], ...] = ()

    def summary(self) -> str:
        if self.reproducible:
            return f"{self.run}: reproducible"
        lines = [f"{self.run}: NOT reproducible"]
        lines.extend(f"  {name}: {why}" for name, why in self.sources)
        return "\n".join(lines)


def nondeterminism_sources(run: Run) -> list[tuple[str, str]]:
    """Named reasons this run cannot be reproduced bit-for-bit."""
    found: list[tuple[str, str]] = []
    if run.seed is None:
        found.append(("seed", _NONDETERMINISM["seed"]))
    if not run.code_version:
        found.append(("code_version", _NONDETERMINISM["code_version"]))
    for key, why in _NONDETERMINISM.items():
        if key in {"seed", "code_version"}:
            continue
        value = run.hyperparameters.get(key)
        if value:
            found.append((key, why))
    return found


def determinism_report(run: Run) -> DeterminismReport:
    sources = nondeterminism_sources(run)
    return DeterminismReport(run=run.name, reproducible=not sources, sources=tuple(sources))


# -- scaling laws --------------------------------------------------------------


@dataclass(frozen=True)
class ScalingLaw:
    """A power-law fit of loss against compute, parameters, or tokens.

    `L(x) = irreducible + coefficient * x ** -exponent`
    """

    variable: str
    coefficient: float
    exponent: float
    irreducible: float = 0.0
    points: int = 0
    fitted_range: tuple[float, float] = (0.0, 0.0)
    residual: float = 0.0

    def predict(self, x: float) -> float:
        if x <= 0:
            return float("inf")
        return float(self.irreducible + self.coefficient * (x**-self.exponent))

    def extrapolation_factor(self, x: float) -> float:
        """How far beyond the fitted range a prediction reaches."""
        top = self.fitted_range[1]
        return 0.0 if top <= 0 else max(0.0, x / top)

    def is_extrapolating(self, x: float, *, limit: float = 10.0) -> bool:
        """True when a prediction is more than `limit`× past the largest fitted point.

        Fitting three small runs and quoting a number two orders of magnitude out is
        how compute budgets get burned. The caller is told rather than stopped.
        """
        return self.extrapolation_factor(x) > limit

    def summary(self) -> str:
        low, high = self.fitted_range
        return (
            f"L({self.variable}) = {self.irreducible:.4g} + "
            f"{self.coefficient:.4g} * {self.variable}^-{self.exponent:.4g}  "
            f"[{self.points} points over {low:.3g}..{high:.3g}, residual {self.residual:.4g}]"
        )


def fit_scaling_law(
    observations: Sequence[tuple[float, float]],
    *,
    variable: str = "compute",
    irreducible: float = 0.0,
) -> ScalingLaw:
    """Least-squares power-law fit in log space.

    Needs at least two points, and reports how many it used — a two-point "law" is a
    line through two dots, and the caller should be able to see that.
    """
    usable = [(x, y - irreducible) for x, y in observations if x > 0 and y - irreducible > 0]
    if len(usable) < 2:
        raise ValueError(
            f"fitting a scaling law needs at least two points above the irreducible "
            f"loss; got {len(usable)}"
        )

    xs = [math.log(x) for x, _ in usable]
    ys = [math.log(y) for _, y in usable]
    n = len(usable)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = (
        0.0
        if denominator == 0
        else sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    )
    intercept = mean_y - slope * mean_x

    residual = math.sqrt(
        sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True)) / n
    )
    raw = [x for x, _ in usable]
    return ScalingLaw(
        variable=variable,
        coefficient=math.exp(intercept),
        exponent=-slope,
        irreducible=irreducible,
        points=n,
        fitted_range=(min(raw), max(raw)),
        residual=residual,
    )


def predict_loss(law: ScalingLaw, x: float) -> float:
    return law.predict(x)


def predict_tokens_needed(law: ScalingLaw, target_loss: float) -> float:
    """Invert the law: how much of the variable reaches a target loss.

    Returns infinity when the target is at or below the irreducible loss, which is
    the honest answer rather than an enormous finite number.
    """
    headroom = target_loss - law.irreducible
    if headroom <= 0 or law.exponent <= 0 or law.coefficient <= 0:
        return float("inf")
    return float((law.coefficient / headroom) ** (1.0 / law.exponent))


def chinchilla_optimal(
    budget_flops: float, *, tokens_per_parameter: float = 20.0
) -> dict[str, float]:
    """Compute-optimal parameter and token counts for a FLOP budget.

    Uses the 6ND approximation and a tokens-per-parameter ratio, defaulting to the
    Chinchilla finding of roughly 20. The ratio is a parameter because it is an
    empirical result that has moved and will move again.
    """
    if budget_flops <= 0:
        return {"parameters": 0.0, "tokens": 0.0, "ratio": tokens_per_parameter}
    parameters = math.sqrt(budget_flops / (6.0 * tokens_per_parameter))
    return {
        "parameters": parameters,
        "tokens": parameters * tokens_per_parameter,
        "ratio": tokens_per_parameter,
        "flops": budget_flops,
    }


def loss_curve_summary(losses: Sequence[float], *, window: int = 10) -> dict[str, float]:
    """Shape of a loss curve: final, best, and whether it is still improving.

    `plateau` is the useful one — a run whose trailing window has stopped moving is a
    run whose remaining compute is being wasted.
    """
    if not losses:
        return {"final": 0.0, "best": 0.0, "improvement": 0.0, "plateau": 1.0, "spikes": 0.0}
    # Clamp the window so head and tail cannot overlap. Without this a short series
    # compares itself to itself and reports zero improvement on a curve that plainly
    # improved, which reads as a plateau and stops a run that should keep going.
    span = max(1, min(window, len(losses) // 2)) if len(losses) > 1 else 1
    tail = list(losses[-span:])
    head = list(losses[:span])
    spread = (max(tail) - min(tail)) if len(tail) > 1 else 0.0
    scale = abs(sum(tail) / len(tail)) or 1.0
    mean_head = sum(head) / len(head)
    mean_tail = sum(tail) / len(tail)
    spikes = sum(
        1
        for previous, current in zip(losses, losses[1:], strict=False)
        if previous > 0 and current > previous * 1.5
    )
    return {
        "final": float(losses[-1]),
        "best": float(min(losses)),
        "improvement": float(mean_head - mean_tail),
        "plateau": float(spread / scale),
        "spikes": float(spikes),
    }


# -- graph integration ---------------------------------------------------------


def run_edges(run: Run) -> list[tuple[DatasetId, DatasetId]]:
    """Edges this run contributes: every input to the run, the run to every output.

    Routing through the run rather than input-to-output directly is what makes the
    hyperparameters attachable to something. An edge cannot hold a learning rate.
    """
    node = run.dataset
    return [(source, node) for source in run.inputs] + [
        (node, produced) for produced in run.outputs
    ]


def to_markdown(runs: Iterable[Run]) -> str:
    """A comparison table, for a pull request or a report."""
    rows = list(runs)
    if not rows:
        return "_no runs_"
    metrics = sorted({name for r in rows for name in r.metrics})
    header = ["run", "status", "tokens", "gpu_hours", *metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [
            r.name,
            r.status.value,
            f"{r.tokens:,}",
            f"{gpu_hours(r):.1f}",
            *[f"{r.metrics.get(m, float('nan')):.6g}" for m in metrics],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
