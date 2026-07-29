"""Training runs as graph edges, and what follows from that.

A model is produced from data the same way a gold table is, so a training run is an
ingest event and the resulting edges go in the same graph. Doing it this way buys
four things that are otherwise separate products:

- **Retraining plans.** When a source partition changes, the planner already knows
  which models it reaches. Retraining becomes an invalidation question rather than
  a cron job that runs weekly whether or not anything moved.
- **Reproducibility.** A run that records concrete input partitions can be replayed;
  one that records "the users table" cannot, because the users table has changed
  since. `unpinned_inputs` names exactly which inputs make a run unrepeatable.
- **Bills of material.** Regulators increasingly ask what a model was trained on,
  and the honest answer is a traversal of this graph, not a spreadsheet somebody
  maintains by hand.
- **Erasure that reaches weights.** If a subject's rows fed a training run, deleting
  the rows does not remove them from the model. Knowing which models were exposed is
  the precondition for doing anything about it — see `fathom.ai.unlearning`.

The `evidence` string on every edge this module writes starts with `training:`, so
model provenance stays distinguishable from SQL-derived lineage forever after.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..core.types import DatasetId, KeyPredicate, PartitionSpec
from ..core.util import digest as _digest
from ..core.util import markdown as md
from ..graph.model import Edge, Graph, InvalidationPlan, link
from ..graph.query import ancestors, descendants
from .assets import AssetKind, is_model, kind_of, spec_for

__all__ = [
    "BillOfMaterials",
    "InputPin",
    "TrainingRun",
    "compare_runs",
    "input_specs",
    "models",
    "pin_from_plan",
    "training_edges",
    "untracked_models",
    "data_bill_of_materials",
    "derived_models",
    "input_digest",
    "is_reproducible",
    "model_card",
    "models_trained_on",
    "provenance_gaps",
    "record_training_run",
    "retraining_plan",
    "run_digest",
    "stale_models",
    "training_inputs",
    "training_data_summary",
    "unpinned_inputs",
]


@dataclass(frozen=True)
class InputPin:
    """One training input, pinned as precisely as the caller could manage.

    `partitions` empty means the whole dataset as it stood at training time, which
    is not a pin at all — the dataset has moved since and nobody can say to what.
    `snapshot` is the adapter's own version handle when there is one: an Iceberg
    snapshot id, a Delta version, an S3 object version. That *is* a real pin.
    """

    dataset: DatasetId
    partitions: frozenset[KeyPredicate] = frozenset()
    snapshot: str = ""
    row_count: int | None = None
    columns: tuple[str, ...] = ()

    @property
    def is_pinned(self) -> bool:
        """True when this input can be reconstructed exactly."""
        return bool(self.snapshot) or bool(self.partitions)

    def __str__(self) -> str:
        keys = ", ".join(sorted(str(k) for k in self.partitions)[:2])
        detail = self.snapshot or (keys if self.partitions else "unpinned")
        return f"{self.dataset} [{detail}]"


@dataclass
class TrainingRun:
    """One training or fine-tuning run, and everything needed to explain it later."""

    model: DatasetId
    inputs: list[InputPin] = field(default_factory=list)
    version: str = ""
    base_model: DatasetId | None = None
    code_version: str = ""
    framework: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    started: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_id: str = ""

    def add_input(
        self,
        dataset: DatasetId,
        *,
        partitions: Iterable[KeyPredicate] = (),
        snapshot: str = "",
        columns: Sequence[str] = (),
        row_count: int | None = None,
    ) -> InputPin:
        """Record one input. Returns the pin so a caller can inspect what was captured."""
        pin = InputPin(
            dataset=dataset,
            partitions=frozenset(partitions),
            snapshot=snapshot,
            row_count=row_count,
            columns=tuple(columns),
        )
        self.inputs.append(pin)
        return pin

    @property
    def datasets(self) -> list[DatasetId]:
        """Every dataset that contributed to this training run."""
        return sorted({pin.dataset for pin in self.inputs}, key=str)

    @property
    def total_rows(self) -> int | None:
        """Rows across every contributing dataset, where counts are known."""
        counts = [pin.row_count for pin in self.inputs if pin.row_count is not None]
        return sum(counts) if len(counts) == len(self.inputs) and counts else None

    def summary(self) -> str:
        """The bill of materials as text, gaps stated rather than omitted."""
        pinned = sum(1 for pin in self.inputs if pin.is_pinned)
        base = f", fine-tuned from {self.base_model}" if self.base_model else ""
        return (
            f"{self.model}{'@' + self.version if self.version else ''}: "
            f"{len(self.inputs)} input(s), {pinned} pinned{base}"
        )


def input_digest(pin: InputPin) -> str:
    """A stable hash of one pinned input. Two runs agree here or they used different data."""
    return _digest.short(
        _digest.of_json(
            {
                "dataset": str(pin.dataset),
                "snapshot": pin.snapshot,
                "partitions": sorted(str(k) for k in pin.partitions),
                "columns": sorted(pin.columns),
            }
        )
    )


def run_digest(run: TrainingRun) -> str:
    """A hash over inputs, code, and hyperparameters.

    Deliberately excludes timestamps and the run id: two runs with the same digest
    should be expected to produce the same model, and *when* they ran does not bear
    on that. A digest that changed every run would be a timestamp with extra steps.
    """
    return _digest.of_json(
        {
            "model": str(run.model),
            "base_model": str(run.base_model) if run.base_model else "",
            "inputs": sorted(input_digest(pin) for pin in run.inputs),
            "code_version": run.code_version,
            "framework": run.framework,
            "hyperparameters": run.hyperparameters,
        }
    )


def unpinned_inputs(run: TrainingRun) -> list[DatasetId]:
    """Inputs recorded without a snapshot or partition set.

    Each one is a reason this run cannot be reproduced, and the list is what to hand
    a team asking why their retrain does not match.
    """
    return sorted({pin.dataset for pin in run.inputs if not pin.is_pinned}, key=str)


def is_reproducible(run: TrainingRun) -> bool:
    """True when every input is pinned and the code version is recorded."""
    return bool(run.inputs) and not unpinned_inputs(run) and bool(run.code_version)


def record_training_run(
    graph: Graph,
    run: TrainingRun,
    *,
    specs: Mapping[DatasetId, PartitionSpec] | None = None,
) -> Graph:
    """Write a training run into the graph as edges, and return the graph.

    Every input edge is `UNBOUNDED`: a model is a function of all its training data
    at once, so no partition of the model can be attributed to a partition of the
    input. That is not a limitation being papered over — it is the honest shape of
    the dependency, and it means a single changed day correctly invalidates the
    whole model rather than some fictional slice of it.
    """
    declared = dict(specs or {})
    model_spec = declared.get(run.model, spec_for(AssetKind.MODEL))
    evidence = f"training:{run.run_id}" if run.run_id else "training"

    for pin in run.inputs:
        link(
            graph,
            pin.dataset,
            run.model,
            evidence=evidence,
            columns=((column, "*") for column in pin.columns),
            src_spec=declared.get(pin.dataset),
            dst_spec=model_spec,
        )

    if run.base_model is not None:
        link(
            graph,
            run.base_model,
            run.model,
            evidence=f"{evidence}:base",
            src_spec=spec_for(AssetKind.MODEL),
            dst_spec=model_spec,
        )
    elif not run.inputs:
        graph.add_dataset(run.model, model_spec)
    return graph


def training_inputs(graph: Graph, model: DatasetId, *, transitive: bool = False) -> list[DatasetId]:
    """Datasets that fed a model. Direct inputs by default, the full closure optionally.

    The transitive form is what a bill of materials needs: a model trained on one
    gold table was, in every sense a regulator means, trained on the raw events
    behind it.
    """
    if transitive:
        return ancestors(graph, model)
    return sorted({e.src for e in graph.in_edges(model)}, key=str)


def models_trained_on(
    graph: Graph, dataset: DatasetId, *, transitive: bool = True
) -> list[DatasetId]:
    """Models this dataset reached. The question every data incident eventually asks."""
    reach = descendants(graph, dataset) if transitive else [e.dst for e in graph.out_edges(dataset)]
    return sorted({ds for ds in reach if is_model(ds)}, key=str)


def derived_models(graph: Graph, base: DatasetId) -> list[DatasetId]:
    """Models fine-tuned from a base model, transitively.

    A safety finding on a base model applies to everything in this list, which is
    usually longer than the team that owns the base model expects.
    """
    return sorted({ds for ds in descendants(graph, base) if is_model(ds)}, key=str)


def retraining_plan(
    graph: Graph, dirty: Mapping[DatasetId, Iterable[KeyPredicate]]
) -> InvalidationPlan:
    """Which models a set of changed source partitions obliges you to retrain.

    The same fixpoint walk the `plan` verb uses, so the same soundness invariant
    holds: it may name a model that did not really need retraining, never miss one
    that did.
    """
    return graph.invalidate(dirty)


def stale_models(
    graph: Graph, dirty: Mapping[DatasetId, Iterable[KeyPredicate]]
) -> list[DatasetId]:
    """Just the models from a retraining plan, without the intermediate tables."""
    plan = retraining_plan(graph, dirty)
    return sorted({ds for ds in plan.dirty if is_model(ds)}, key=str)


def provenance_gaps(graph: Graph, model: DatasetId) -> list[str]:
    """Everything about this model's lineage that cannot currently be answered.

    Written as sentences rather than codes because the audience is whoever has to
    sign the attestation, and they need to know what is missing, not what enum
    variant it corresponds to.
    """
    gaps: list[str] = []
    upstream = ancestors(graph, model)
    if not upstream:
        gaps.append("no training inputs recorded; this model's provenance is unknown")
        return gaps

    for edge in graph.in_edges(model):
        if not edge.columns:
            gaps.append(
                f"the edge from {edge.src} carries no column detail, so which fields "
                "entered training cannot be stated"
            )
        if edge.mapping.is_unbounded and graph.spec(edge.src).fields:
            gaps.append(
                f"{edge.src} is partitioned but the training edge does not record which "
                "partitions were used; the run cannot be reproduced from the graph alone"
            )
    orphan_sources = [
        ds
        for ds in upstream
        if kind_of(ds) is AssetKind.TABLE and not graph.in_edges(ds) and not graph.spec(ds).fields
    ]
    if orphan_sources:
        gaps.append(
            f"{len(orphan_sources)} upstream source(s) have neither a partition spec nor "
            "lineage of their own; the closure stops there rather than at a real origin"
        )
    return gaps


@dataclass
class BillOfMaterials:
    """Everything a model was built from, at every remove."""

    model: DatasetId
    direct: list[DatasetId] = field(default_factory=list)
    transitive: list[DatasetId] = field(default_factory=list)
    base_models: list[DatasetId] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True when every input's provenance is known.

        False is the normal answer and the useful one: it names what the filing
        cannot yet claim.
        """
        return not self.gaps

    def summary(self) -> str:
        """The bill of materials as text, gaps stated rather than omitted."""
        lines = [
            f"{self.model}: {len(self.direct)} direct input(s), "
            f"{len(self.transitive)} in the full closure"
        ]
        if self.base_models:
            lines.append(f"  built on: {', '.join(str(m) for m in self.base_models)}")
        for gap in self.gaps:
            lines.append(f"  gap: {gap}")
        return "\n".join(lines)


def data_bill_of_materials(graph: Graph, model: DatasetId) -> BillOfMaterials:
    """The full input closure of a model, with its own gaps stated.

    A bill of materials that hides what it could not determine is worse than none,
    because it will be read as complete.
    """
    transitive = ancestors(graph, model)
    return BillOfMaterials(
        model=model,
        direct=training_inputs(graph, model),
        transitive=transitive,
        base_models=sorted((ds for ds in transitive if is_model(ds)), key=str),
        gaps=provenance_gaps(graph, model),
    )


def training_data_summary(graph: Graph, model: DatasetId) -> str:
    """A prose summary of what a model was trained on.

    Shaped for the "sufficiently detailed summary of training content" that the EU
    AI Act asks general-purpose model providers to publish. It is generated from the
    graph, so it stays true as the graph changes rather than aging in a document.
    """
    bom = data_bill_of_materials(graph, model)
    by_namespace: dict[str, int] = {}
    for ds in bom.transitive:
        by_namespace[ds.namespace] = by_namespace.get(ds.namespace, 0) + 1

    lines = [
        f"# Training data summary — {model.name}",
        "",
        f"Generated from lineage on {datetime.now(UTC).date().isoformat()}.",
        "",
        f"This model draws on {len(bom.transitive)} upstream dataset(s) across "
        f"{len(by_namespace)} system(s).",
        "",
        "## Sources by system",
        "",
    ]
    lines.append(
        md.bullets(
            f"{md.code(namespace)} — {count} dataset(s)"
            for namespace, count in sorted(by_namespace.items())
        )
    )
    lines.extend(
        ["", "## Direct training inputs", "", md.bullets(md.code(ds) for ds in bom.direct)]
    )
    if bom.base_models:
        lines.extend(
            [
                "",
                "## Inherited from base models",
                "",
                md.bullets(md.code(ds) for ds in bom.base_models),
            ]
        )
    if bom.gaps:
        lines.extend(
            [
                "",
                "## Known gaps",
                "",
                "This summary is incomplete:",
                "",
                md.bullets(bom.gaps),
            ]
        )
    return "\n".join(lines)


def model_card(
    graph: Graph,
    model: DatasetId,
    *,
    run: TrainingRun | None = None,
    intended_use: str = "",
    limitations: str = "",
) -> str:
    """A model card whose provenance section is generated rather than written.

    The narrative sections stay a human's job. The lineage, input list, and
    reproducibility verdict come from the graph, which is the part that otherwise
    goes stale within a week of being written.
    """
    bom = data_bill_of_materials(graph, model)
    lines = [f"# Model card — {model.name}", "", f"- identity: `{model}`"]
    if run is not None:
        lines.extend(
            [
                f"- version: `{run.version or 'unversioned'}`",
                f"- framework: {run.framework or '—'}",
                f"- code version: `{run.code_version or 'unrecorded'}`",
                f"- trained: {run.started.isoformat()}",
                f"- run digest: `{run_digest(run)[:16]}`",
                f"- reproducible: {'yes' if is_reproducible(run) else 'no'}",
            ]
        )
    lines.extend(["", "## Intended use", "", intended_use or "_Not stated._"])
    lines.extend(["", "## Limitations", "", limitations or "_Not stated._"])
    lines.extend(
        [
            "",
            "## Training data",
            "",
            f"{len(bom.transitive)} dataset(s) upstream.",
            "",
            md.bullets(md.code(ds) for ds in bom.direct),
        ]
    )
    if run is not None and unpinned_inputs(run):
        lines.extend(
            [
                "",
                "### Unpinned inputs",
                "",
                "These were not captured precisely enough to reproduce the run:",
                "",
                md.bullets(md.code(ds) for ds in unpinned_inputs(run)),
            ]
        )
    if bom.gaps:
        lines.extend(["", "### Provenance gaps", "", md.bullets(bom.gaps)])
    return "\n".join(lines)


def compare_runs(before: TrainingRun, after: TrainingRun) -> dict[str, Any]:
    """What differs between two training runs of the same model.

    When a retrain regresses, this is the first thing to look at: exactly one of
    data, code, or hyperparameters changed, and the answer is usually here.
    """
    before_inputs = {str(pin.dataset): input_digest(pin) for pin in before.inputs}
    after_inputs = {str(pin.dataset): input_digest(pin) for pin in after.inputs}
    changed = sorted(
        name
        for name in set(before_inputs) & set(after_inputs)
        if before_inputs[name] != after_inputs[name]
    )
    hyper_changed = sorted(
        key
        for key in set(before.hyperparameters) | set(after.hyperparameters)
        if before.hyperparameters.get(key) != after.hyperparameters.get(key)
    )
    return {
        "same_digest": run_digest(before) == run_digest(after),
        "inputs_added": sorted(set(after_inputs) - set(before_inputs)),
        "inputs_removed": sorted(set(before_inputs) - set(after_inputs)),
        "inputs_changed": changed,
        "code_changed": before.code_version != after.code_version,
        "hyperparameters_changed": hyper_changed,
    }


def pin_from_plan(plan: InvalidationPlan, dataset: DatasetId) -> InputPin:
    """Build an input pin from a plan's partitions for one dataset.

    The natural bridge: whatever the planner said was fresh is exactly what a
    training run should record as the slice it consumed.
    """
    return InputPin(dataset=dataset, partitions=plan.partitions(dataset))


def training_edges(graph: Graph) -> list[Edge]:
    """Every edge written by a training run, across the whole graph."""
    return sorted(
        (e for e in graph.edges if e.evidence.startswith("training")),
        key=lambda e: (str(e.dst), str(e.src)),
    )


def models(graph: Graph) -> list[DatasetId]:
    """Every model, checkpoint, and adapter in the graph."""
    return sorted((ds for ds in graph.datasets if is_model(ds)), key=str)


def untracked_models(graph: Graph) -> list[DatasetId]:
    """Models with no recorded inputs at all — provenance that was never captured."""
    return sorted((ds for ds in models(graph) if not graph.in_edges(ds)), key=str)


def input_specs(graph: Graph, model: DatasetId) -> dict[DatasetId, PartitionSpec]:
    """Partition specs of a model's direct inputs, for building pins against."""
    return {ds: graph.spec(ds) for ds in training_inputs(graph, model)}
