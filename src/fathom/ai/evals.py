"""Eval sets, and whether you can believe the numbers they produce.

An eval score is a claim about data the model has not seen. Lineage is how that
claim gets checked, because contamination is a reachability property: if the eval
set and the training set share an ancestor, or one is downstream of the other, the
model was trained on its own test.

That happens constantly, and rarely on purpose. A team builds an eval from the same
gold table the features come from. A synthetic-data job seeds itself from the
benchmark. A scraped corpus quietly includes the benchmark's public repository. None
of these look like cheating from inside the pipeline that does them — they look like
reuse, which is normally a virtue.

Three levels of evidence, weakest first, all reported separately because they mean
different things:

- **Shared ancestry.** The two sets have a common upstream. Suggestive, common, and
  frequently benign.
- **Reachability.** The eval set is downstream of the training set, or the reverse.
  Strong. The eval no longer measures generalization.
- **Identifier overlap.** The same record ids appear in both. Conclusive, and the
  only one of the three that needs actual data rather than the graph.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..core.types import DatasetId
from ..graph.model import Graph, link
from ..graph.query import ancestors, common_ancestors, has_path, shortest_path
from .assets import AssetKind, spec_for
from .training import training_inputs

__all__ = [
    "ContaminationReport",
    "EvalResult",
    "added_metrics",
    "audit_evals",
    "compare_results",
    "contaminated_models",
    "holdout_integrity",
    "contamination",
    "eval_sets_for",
    "identifier_overlap",
    "is_contaminated",
    "leakage_paths",
    "models_evaluated_by",
    "record_eval",
    "regressions",
    "removed_metrics",
    "shared_ancestry",
    "stale_results",
]


@dataclass(frozen=True)
class EvalResult:
    """One model graded against one eval set."""

    model: DatasetId
    eval_set: DatasetId
    metrics: Mapping[str, float] = field(default_factory=dict)
    model_version: str = ""
    eval_version: str = ""
    recorded: datetime = field(default_factory=lambda: datetime.now(UTC))

    def metric(self, name: str) -> float | None:
        """One metric's value, or None when this run did not report it."""
        return self.metrics.get(name)

    def __str__(self) -> str:
        rendered = ", ".join(f"{k}={v:.4g}" for k, v in sorted(self.metrics.items()))
        return f"{self.model}@{self.model_version or '?'} on {self.eval_set}: {rendered}"


def record_eval(graph: Graph, result: EvalResult) -> Graph:
    """Write an evaluation into the graph as an edge from the eval set to the model.

    The direction is deliberate and initially surprising: the eval set feeds the
    model, because grading is a use of data and this is what makes contamination
    checks and erasure walks find eval sets at all. An eval set holding a person's
    data is an obligation exactly like a training set is.
    """
    link(
        graph,
        result.eval_set,
        result.model,
        evidence="eval",
        src_spec=spec_for(AssetKind.EVAL_SET),
        dst_spec=spec_for(AssetKind.MODEL),
    )
    return graph


def eval_sets_for(graph: Graph, model: DatasetId) -> list[DatasetId]:
    """Eval sets this model was graded against."""
    from .assets import kind_of

    return sorted(
        {e.src for e in graph.in_edges(model) if kind_of(e.src) is AssetKind.EVAL_SET}, key=str
    )


def models_evaluated_by(graph: Graph, eval_set: DatasetId) -> list[DatasetId]:
    """Models graded against one eval set. Everything invalidated if the set is compromised."""
    from .assets import is_model

    return sorted({e.dst for e in graph.out_edges(eval_set) if is_model(e.dst)}, key=str)


# -- contamination -------------------------------------------------------------


def shared_ancestry(graph: Graph, train: DatasetId, eval_set: DatasetId) -> list[DatasetId]:
    """Datasets upstream of both the training data and the eval set."""
    return common_ancestors(graph, train, eval_set)


def leakage_paths(graph: Graph, train: DatasetId, eval_set: DatasetId) -> list[list[DatasetId]]:
    """Direct routes between the two sets, in either direction.

    A non-empty result means one set is derived from the other, which is the strong
    form of contamination.
    """
    found: list[list[DatasetId]] = []
    for src, dst in ((train, eval_set), (eval_set, train)):
        if has_path(graph, src, dst):
            path = shortest_path(graph, src, dst)
            if path:
                found.append(path)
    return found


def identifier_overlap(
    train_ids: Iterable[str], eval_ids: Iterable[str]
) -> tuple[int, float, list[str]]:
    """Records appearing in both sets: the count, the eval-side fraction, and examples.

    The fraction is against the eval set rather than the training set on purpose. Ten
    thousand shared records out of a million training rows sounds negligible; the same
    overlap covering half the eval set means the score is meaningless.

    It is the fraction of eval *rows* whose identifier was seen in training, not the
    fraction of distinct identifiers. Dividing a distinct-id count by a row count
    mixes units and understates: an eval set of five rows, four of them the same
    contaminated record, is 80% compromised and was previously reported as 20%.
    Understating is the one direction a contamination check must never round.
    """
    train = set(train_ids)
    evaluation = list(eval_ids)
    contaminated_rows = sum(1 for record in evaluation if record in train)
    shared = sorted({record for record in evaluation if record in train})
    ratio = contaminated_rows / len(evaluation) if evaluation else 0.0
    return len(shared), ratio, shared[:20]


@dataclass
class ContaminationReport:
    """Whether an eval score can be believed, and on what evidence."""

    model: DatasetId
    eval_set: DatasetId
    shared_ancestors: list[DatasetId] = field(default_factory=list)
    paths: list[list[DatasetId]] = field(default_factory=list)
    overlapping_ids: int = 0
    overlap_ratio: float = 0.0
    examples: list[str] = field(default_factory=list)

    @property
    def severity(self) -> str:
        """`clean`, `suspect`, or `contaminated`, in ascending order of certainty."""
        if self.overlapping_ids or self.paths:
            return "contaminated"
        if self.shared_ancestors:
            return "suspect"
        return "clean"

    @property
    def is_contaminated(self) -> bool:
        """True when graph or identifier evidence discredits the score."""
        return self.severity == "contaminated"

    def summary(self) -> str:
        """The report as text, strongest evidence first."""
        lines = [f"{self.model} on {self.eval_set}: {self.severity.upper()}"]
        if self.paths:
            for path in self.paths:
                lines.append("  path: " + " -> ".join(str(node) for node in path))
        if self.overlapping_ids:
            lines.append(
                f"  {self.overlapping_ids} shared record id(s), "
                f"{self.overlap_ratio:.1%} of the eval set"
            )
            if self.examples:
                lines.append("  e.g. " + ", ".join(self.examples[:5]))
        if self.shared_ancestors and not self.paths:
            lines.append(
                f"  {len(self.shared_ancestors)} shared upstream dataset(s): "
                + ", ".join(str(ds) for ds in self.shared_ancestors[:3])
            )
        if self.severity == "clean":
            lines.append("  no shared lineage found")
        return "\n".join(lines)


def contamination(
    graph: Graph,
    model: DatasetId,
    eval_set: DatasetId,
    *,
    train_ids: Iterable[str] = (),
    eval_ids: Iterable[str] = (),
) -> ContaminationReport:
    """Check one model's eval set against everything it was trained on.

    Identifier overlap is optional because it needs the data itself. Without it the
    verdict tops out at `suspect` on shared ancestry alone — which is the honest
    ceiling for a graph-only check, and is stated rather than rounded up.
    """
    report = ContaminationReport(model=model, eval_set=eval_set)

    for source in training_inputs(graph, model):
        if source == eval_set:
            continue
        report.shared_ancestors.extend(shared_ancestry(graph, source, eval_set))
        report.paths.extend(leakage_paths(graph, source, eval_set))

    report.shared_ancestors = sorted(set(report.shared_ancestors), key=str)

    if train_ids and eval_ids:
        count, ratio, examples = identifier_overlap(train_ids, eval_ids)
        report.overlapping_ids = count
        report.overlap_ratio = ratio
        report.examples = examples
    return report


def is_contaminated(graph: Graph, model: DatasetId, eval_set: DatasetId) -> bool:
    """True when graph evidence alone is enough to distrust the score."""
    return contamination(graph, model, eval_set).is_contaminated


def audit_evals(graph: Graph, model: DatasetId) -> list[ContaminationReport]:
    """Contamination reports for every eval set a model was graded against."""
    return [contamination(graph, model, eval_set) for eval_set in eval_sets_for(graph, model)]


def contaminated_models(graph: Graph) -> list[tuple[DatasetId, DatasetId]]:
    """Every (model, eval set) pair in the graph with provable leakage.

    The report to run across a whole model registry before publishing a leaderboard.
    """
    from .assets import is_model

    out: list[tuple[DatasetId, DatasetId]] = []
    for ds in graph.datasets:
        if not is_model(ds):
            continue
        for report in audit_evals(graph, ds):
            if report.is_contaminated:
                out.append((ds, report.eval_set))
    return sorted(out, key=lambda pair: (str(pair[0]), str(pair[1])))


# -- results over time ---------------------------------------------------------


def compare_results(before: EvalResult, after: EvalResult) -> dict[str, tuple[float, float, float]]:
    """Per-metric before, after, and delta, for metrics both runs reported.

    Metrics present on only one side are omitted rather than filled with zero. A
    suite that gains `f1` did not improve it from 0.0, and a suite that drops it did
    not regress to 0.0 — but substituting a default produced exactly those two
    fictions, so adding or removing any metric raised a phantom regression on the
    leaderboard. Use `added_metrics` and `removed_metrics` to see the difference in
    coverage, which is a separate fact from a change in score.
    """
    out: dict[str, tuple[float, float, float]] = {}
    for name in sorted(set(before.metrics) & set(after.metrics)):
        b, a = before.metrics[name], after.metrics[name]
        out[name] = (b, a, a - b)
    return out


def added_metrics(before: EvalResult, after: EvalResult) -> list[str]:
    """Metrics the later run reports and the earlier one did not."""
    return sorted(set(after.metrics) - set(before.metrics))


def removed_metrics(before: EvalResult, after: EvalResult) -> list[str]:
    """Metrics the earlier run reported and the later one dropped.

    Worth surfacing on its own: a metric that stops being reported looks like a clean
    run, and is often how a failing check quietly leaves a suite.
    """
    return sorted(set(before.metrics) - set(after.metrics))


def regressions(
    before: EvalResult,
    after: EvalResult,
    *,
    tolerance: float = 0.01,
    lower_is_better: Iterable[str] = ("loss", "error", "latency"),
) -> list[str]:
    """Metrics that moved the wrong way by more than `tolerance`.

    Direction is decided per metric name rather than assumed, because a suite mixing
    accuracy and loss will otherwise report every improvement as a regression.
    """
    inverted = {name.lower() for name in lower_is_better}
    out: list[str] = []
    for name, (b, a, delta) in compare_results(before, after).items():
        worse_when_lower = not any(token in name.lower() for token in inverted)
        regressed = (delta < -tolerance) if worse_when_lower else (delta > tolerance)
        if regressed:
            out.append(f"{name}: {b:.4g} -> {a:.4g} ({delta:+.4g})")
    return out


def stale_results(
    graph: Graph, results: Sequence[EvalResult], dirty: Mapping[DatasetId, Iterable[Any]]
) -> list[EvalResult]:
    """Eval results invalidated by a change to the model or the eval set.

    An eval score is a claim about a specific model version against a specific eval
    version. Either moving makes the number describe something that no longer exists,
    and leaderboards routinely keep showing it anyway.
    """
    plan = graph.invalidate({ds: list(keys) for ds, keys in dirty.items()})
    affected = set(plan.dirty)
    return [r for r in results if r.model in affected or r.eval_set in affected]


def holdout_integrity(graph: Graph, eval_set: DatasetId) -> list[str]:
    """Reasons this eval set may not be a genuine holdout.

    Phrased as findings rather than a boolean because each one has a different fix,
    and a single "contaminated: true" tells a team nothing about which.
    """
    findings: list[str] = []
    upstream = ancestors(graph, eval_set)
    if not upstream:
        findings.append("no lineage recorded for this eval set; its provenance is unverifiable")

    from .assets import is_model

    for source in upstream:
        if is_model(source):
            findings.append(
                f"derived from the model {source}; a set generated by a model cannot "
                "measure that model"
            )
    # `is_model` is a name check; `has_path` is a full traversal. Ordering the
    # cheap predicate first turns O(V) traversals into O(models) of them.
    consumers = [ds for ds in graph.datasets if is_model(ds) and has_path(graph, eval_set, ds)]
    if len(consumers) > 5:
        findings.append(
            f"used by {len(consumers)} models; repeated use against the same holdout "
            "overfits it even without any single leak"
        )
    return findings
