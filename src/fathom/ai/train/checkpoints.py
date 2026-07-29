"""Checkpoints, sharding, and whether a run can actually resume from one.

A checkpoint is not one file. It is a set of shards written by a specific
parallelism configuration, and it is resumable only by a job whose topology can
consume that layout. Resuming a 512-GPU run into a 256-GPU cluster is a real
operation with a real failure mode, and treating a checkpoint as a single opaque
blob is how a team discovers at hour thirty that it cannot.

`can_resume` is the function this module exists for. It reports the specific
incompatibility — a tensor-parallel degree that does not divide, an optimizer state
sharded per-rank against a different world size, a mismatched framework version —
rather than a boolean, because every one of those has a different remedy and two of
them are fixable by resharding.

Retention is the other half. Checkpoints are enormous and mostly worthless: the
interesting ones are the latest, the best, and the ones someone else's model was
derived from. `retention_plan` will not propose deleting the third kind, because a
checkpoint with descendants in the graph is evidence, not storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from ...core.types import DatasetId
from ..assets import checkpoint as checkpoint_asset

__all__ = [
    "Checkpoint",
    "Parallelism",
    "ResumeVerdict",
    "Shard",
    "ShardingScheme",
    "RetentionPlan",
    "RetentionReason",
    "can_resume",
    "checkpoint_size",
    "compatible_topologies",
    "describe_topology",
    "is_resharding_required",
    "latest",
    "missing_shards",
    "optimizer_overhead",
    "parallelism_from",
    "reshard_plan",
    "retention_plan",
    "shard_count",
    "shard_index",
    "step_of",
    "topology_changed",
    "validate",
    "world_size",
]


class ShardingScheme(StrEnum):
    """How a checkpoint's tensors are split across files."""

    SINGLE = "single"  # one file, one rank; resumable anywhere
    PER_RANK = "per_rank"  # one file per rank; resumable only at the same world size
    SHARDED = "sharded"  # distributed checkpoint; resumable after resharding
    CONSOLIDATED = "consolidated"  # gathered to rank zero; portable but large


@dataclass(frozen=True)
class Parallelism:
    """The topology a checkpoint was written under.

    `world_size` is the product, and it is derived rather than stored so a config
    that does not multiply out cannot be represented at all.
    """

    data: int = 1
    tensor: int = 1
    pipeline: int = 1
    expert: int = 1
    context: int = 1

    def __post_init__(self) -> None:
        for name in ("data", "tensor", "pipeline", "expert", "context"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} parallelism must be at least 1")

    @property
    def size(self) -> int:
        return self.data * self.tensor * self.pipeline * self.expert * self.context

    def describe(self) -> str:
        parts = [
            f"{name[0]}p={value}"
            for name, value in (
                ("data", self.data),
                ("tensor", self.tensor),
                ("pipeline", self.pipeline),
                ("expert", self.expert),
                ("context", self.context),
            )
            if value > 1
        ]
        return " ".join(parts) if parts else "single device"


@dataclass(frozen=True)
class Shard:
    """One file of a checkpoint."""

    path: str
    rank: int = 0
    bytes: int = 0
    contains_optimizer: bool = False
    checksum: str = ""


@dataclass(frozen=True)
class Checkpoint:
    """One saved state of a training run."""

    name: str
    step: int = 0
    parallelism: Parallelism = field(default_factory=Parallelism)
    scheme: ShardingScheme = ShardingScheme.SINGLE
    shards: tuple[Shard, ...] = ()
    framework: str = ""
    framework_version: str = ""
    dtype: str = ""
    parameters: int = 0
    run: str = ""
    written: datetime | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    registry: str = "local"

    @property
    def dataset(self) -> DatasetId:
        return checkpoint_asset(self.name, registry=self.registry)


def world_size(checkpoint: Checkpoint) -> int:
    return checkpoint.parallelism.size


def shard_count(checkpoint: Checkpoint) -> int:
    return len(checkpoint.shards)


def checkpoint_size(checkpoint: Checkpoint) -> int:
    return sum(shard.bytes for shard in checkpoint.shards)


def optimizer_overhead(checkpoint: Checkpoint) -> float:
    """Fraction of the checkpoint that is optimizer state rather than weights.

    Usually two thirds for Adam in fp32. Worth surfacing, because a retention policy
    that drops optimizer state keeps the model and loses only resumability.
    """
    total = checkpoint_size(checkpoint)
    if total <= 0:
        return 0.0
    return sum(s.bytes for s in checkpoint.shards if s.contains_optimizer) / total


def shard_index(checkpoint: Checkpoint) -> dict[int, list[Shard]]:
    """Shards grouped by rank."""
    index: dict[int, list[Shard]] = {}
    for shard in checkpoint.shards:
        index.setdefault(shard.rank, []).append(shard)
    return index


def missing_shards(checkpoint: Checkpoint) -> list[int]:
    """Ranks with no shard present.

    A checkpoint missing one rank is not a slightly damaged checkpoint; it is
    unloadable, and finding out at resume time costs whatever the queue wait was.
    """
    if checkpoint.scheme in {ShardingScheme.SINGLE, ShardingScheme.CONSOLIDATED}:
        return [] if checkpoint.shards else [0]
    present = set(shard_index(checkpoint))
    return sorted(set(range(world_size(checkpoint))) - present)


def step_of(checkpoint: Checkpoint) -> int:
    return checkpoint.step


def describe_topology(checkpoint: Checkpoint) -> str:
    return (
        f"{checkpoint.parallelism.describe()} "
        f"(world size {world_size(checkpoint)}, {checkpoint.scheme.value})"
    )


def parallelism_from(mapping: Mapping[str, int]) -> Parallelism:
    """Build a topology from a config mapping, accepting the usual spellings."""

    def pick(*names: str) -> int:
        for name in names:
            if name in mapping:
                return int(mapping[name])
        return 1

    return Parallelism(
        data=pick("data", "dp", "data_parallel_size"),
        tensor=pick("tensor", "tp", "tensor_parallel_size", "tensor_model_parallel_size"),
        pipeline=pick("pipeline", "pp", "pipeline_parallel_size", "pipeline_model_parallel_size"),
        expert=pick("expert", "ep", "expert_parallel_size"),
        context=pick("context", "cp", "context_parallel_size", "sequence_parallel_size"),
    )


# -- resume --------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeVerdict:
    """Whether a checkpoint can be loaded into a target topology, and why not.

    `resharding_required` is the interesting middle state: not directly loadable,
    but recoverable without retraining. Reporting it as a plain failure sends people
    to restart a run they could have resumed.
    """

    can_resume: bool
    resharding_required: bool = False
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        if self.can_resume and not self.warnings:
            return "resumable"
        lines = [
            "resumable after resharding"
            if self.resharding_required
            else ("resumable" if self.can_resume else "NOT resumable")
        ]
        lines.extend(f"  blocker: {b}" for b in self.blockers)
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def can_resume(
    checkpoint: Checkpoint,
    target: Parallelism,
    *,
    framework: str = "",
    framework_version: str = "",
    parameters: int = 0,
) -> ResumeVerdict:
    """Whether a job with `target` topology can load this checkpoint.

    Distinguishes three outcomes rather than two: loadable, loadable after a
    resharding pass, and genuinely blocked. Only the third means retraining.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    reshard = False

    absent = missing_shards(checkpoint)
    if absent:
        blockers.append(
            f"{len(absent)} shard(s) missing for rank(s) {absent[:8]}; the checkpoint "
            "is incomplete and cannot be loaded at any topology"
        )

    source = checkpoint.parallelism
    if source != target:
        if checkpoint.scheme is ShardingScheme.PER_RANK and source.size != target.size:
            blockers.append(
                f"per-rank checkpoint written at world size {source.size} cannot load "
                f"at {target.size}; consolidate it first"
            )
        elif checkpoint.scheme in {ShardingScheme.SHARDED, ShardingScheme.CONSOLIDATED}:
            reshard = True
            warnings.append(
                f"topology changed {source.describe()} -> {target.describe()}; resharding required"
            )
        elif checkpoint.scheme is ShardingScheme.SINGLE:
            warnings.append("single-file checkpoint will be broadcast to all ranks")

    # A tensor-parallel degree that does not divide the parameter count evenly cannot
    # be resharded without knowing the model's layer shapes.
    if target.tensor > 1 and parameters and parameters % target.tensor != 0:
        warnings.append(
            f"parameter count {parameters:,} is not divisible by tensor parallel "
            f"degree {target.tensor}; resharding needs per-layer shapes"
        )

    if framework and checkpoint.framework and framework != checkpoint.framework:
        blockers.append(f"checkpoint was written by {checkpoint.framework}, target is {framework}")
    if (
        framework_version
        and checkpoint.framework_version
        and framework_version != checkpoint.framework_version
    ):
        warnings.append(
            f"framework version differs ({checkpoint.framework_version} -> "
            f"{framework_version}); state dict keys may have moved"
        )

    has_optimizer = any(s.contains_optimizer for s in checkpoint.shards)
    if not has_optimizer and checkpoint.shards:
        warnings.append(
            "no optimizer state present; the run will resume with a fresh optimizer, "
            "which is a different experiment"
        )

    return ResumeVerdict(
        can_resume=not blockers,
        resharding_required=reshard and not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def is_resharding_required(checkpoint: Checkpoint, target: Parallelism) -> bool:
    return can_resume(checkpoint, target).resharding_required


def topology_changed(checkpoint: Checkpoint, target: Parallelism) -> bool:
    return checkpoint.parallelism != target


def compatible_topologies(checkpoint: Checkpoint, *, limit: int = 16) -> list[Parallelism]:
    """Topologies this checkpoint loads into without resharding.

    For a per-rank checkpoint that is exactly its own; for a sharded one it is every
    factorisation of the world size, which is what makes "can I fit this on the
    cluster I actually have" answerable.
    """
    size = world_size(checkpoint)
    if checkpoint.scheme is ShardingScheme.PER_RANK:
        return [checkpoint.parallelism]
    if checkpoint.scheme is ShardingScheme.SINGLE:
        return [Parallelism(data=n) for n in range(1, min(size, limit) + 1)]

    found: list[Parallelism] = []
    for tensor in _divisors(size):
        remainder = size // tensor
        for pipeline in _divisors(remainder):
            data = remainder // pipeline
            found.append(Parallelism(data=data, tensor=tensor, pipeline=pipeline))
            if len(found) >= limit:
                return found
    return found


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def reshard_plan(checkpoint: Checkpoint, target: Parallelism) -> dict[str, object]:
    """What a resharding pass would have to do."""
    source = checkpoint.parallelism
    return {
        "from": source.describe(),
        "to": target.describe(),
        "from_world_size": source.size,
        "to_world_size": target.size,
        "shards_in": shard_count(checkpoint),
        "shards_out": target.size if checkpoint.scheme is not ShardingScheme.SINGLE else 1,
        "bytes": checkpoint_size(checkpoint),
        "tensor_split_changed": source.tensor != target.tensor,
        "pipeline_split_changed": source.pipeline != target.pipeline,
        "expert_split_changed": source.expert != target.expert,
        "gather_required": source.tensor > target.tensor or source.pipeline > target.pipeline,
    }


def validate(checkpoint: Checkpoint) -> list[str]:
    """Structural problems, checked before a resume rather than during one."""
    problems: list[str] = []
    absent = missing_shards(checkpoint)
    if absent:
        problems.append(f"missing shards for ranks {absent[:8]}")
    if checkpoint.scheme is ShardingScheme.PER_RANK and shard_count(checkpoint) != world_size(
        checkpoint
    ):
        problems.append(
            f"per-rank checkpoint has {shard_count(checkpoint)} shards for world size "
            f"{world_size(checkpoint)}"
        )
    if checkpoint.shards and not any(s.checksum for s in checkpoint.shards):
        problems.append("no checksums recorded; silent corruption would be undetectable")
    if checkpoint.step < 0:
        problems.append("negative step")
    if not checkpoint.framework:
        problems.append("no framework recorded; resume compatibility cannot be checked")
    return problems


# -- retention -----------------------------------------------------------------


class RetentionReason(StrEnum):
    """Why a checkpoint is kept. Ordered by how hard it is to argue with."""

    HAS_DESCENDANTS = "has_descendants"  # something was derived from it; it is evidence
    LATEST = "latest"
    BEST = "best"
    MILESTONE = "milestone"
    WITHIN_WINDOW = "within_window"
    EXPENDABLE = "expendable"


@dataclass(frozen=True)
class RetentionPlan:
    """Which checkpoints to keep, which to drop, and why."""

    keep: tuple[tuple[str, RetentionReason], ...] = ()
    drop: tuple[str, ...] = ()
    bytes_freed: int = 0

    def summary(self) -> str:
        lines = [
            f"keep {len(self.keep)}, drop {len(self.drop)}, free {self.bytes_freed / 1e9:.1f} GB"
        ]
        for name, reason in self.keep:
            lines.append(f"  keep {name}: {reason.value}")
        return "\n".join(lines)


def latest(checkpoints: Iterable[Checkpoint]) -> Checkpoint | None:
    ordered = sorted(checkpoints, key=lambda c: c.step)
    return ordered[-1] if ordered else None


def retention_plan(
    checkpoints: Sequence[Checkpoint],
    *,
    keep_last: int = 3,
    metric: str = "loss",
    minimize: bool = True,
    milestone_every: int = 0,
    referenced: Iterable[str] = (),
) -> RetentionPlan:
    """Decide which checkpoints to keep.

    `referenced` names checkpoints something in the graph was derived from. Those are
    never proposed for deletion regardless of age: a checkpoint with descendants is
    the evidence for how they came to exist, and deleting it makes a training-data
    question permanently unanswerable.
    """
    if not checkpoints:
        return RetentionPlan()

    ordered = sorted(checkpoints, key=lambda c: c.step)
    protected = set(referenced)
    keep: dict[str, RetentionReason] = {}

    for c in ordered[-max(keep_last, 0) :]:
        keep[c.name] = RetentionReason.WITHIN_WINDOW
    if ordered:
        keep[ordered[-1].name] = RetentionReason.LATEST

    scored = [(c, c.metrics.get(metric)) for c in ordered]
    usable = [(c, v) for c, v in scored if v is not None]
    if usable:
        best = (
            min(usable, key=lambda p: p[1])[0] if minimize else max(usable, key=lambda p: p[1])[0]
        )
        keep[best.name] = RetentionReason.BEST

    if milestone_every > 0:
        for c in ordered:
            if c.step and c.step % milestone_every == 0:
                keep.setdefault(c.name, RetentionReason.MILESTONE)

    for c in ordered:
        if c.name in protected:
            keep[c.name] = RetentionReason.HAS_DESCENDANTS

    drop = [c for c in ordered if c.name not in keep]
    return RetentionPlan(
        keep=tuple(sorted(keep.items())),
        drop=tuple(c.name for c in drop),
        bytes_freed=sum(checkpoint_size(c) for c in drop),
    )
