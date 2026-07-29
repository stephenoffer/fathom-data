"""Derivation between models: adapters, merges, distillation, and preference data.

A deployed model is rarely a thing that was trained once. It is a base model, plus a
LoRA adapter, merged, quantized, and served — four derivations deep, each with its
own inputs and its own licence. "Which base model is inside this endpoint" is a
traversal, and until the derivations are edges it is an archaeology exercise.

The two things this module gets right on purpose:

**Licences compound restrictively.** A permissively-licenced adapter over a
research-only base is research-only. Teams get this backwards constantly, because
the adapter is the part they wrote. `effective_restrictions` unions the constraints
rather than the permissions.

**Preference data has annotators in it.** A DPO run is downstream of ranked pairs,
which are downstream of people who were paid to produce them and whose consent has a
scope. That chain is a real erasure obligation and it is three edges long, so
nothing finds it by accident.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ...core.types import DatasetId
from ..assets import adapter as adapter_asset
from ..assets import annotation_set, model, preference_set

__all__ = [
    "Adaptation",
    "AdaptationKind",
    "Annotator",
    "Derivation",
    "Distillation",
    "Merge",
    "PreferencePair",
    "PreferenceSet",
    "Quantization",
    "QuantizationFormat",
    "adapter_edges",
    "annotator_diversity",
    "base_of",
    "derivation_chain",
    "distillation_edges",
    "effective_restrictions",
    "inter_annotator_agreement",
    "is_derived_from",
    "lineage_depth",
    "merge_edges",
    "preference_edges",
    "quantization_edges",
    "rank_stability",
    "trainable_fraction",
    "validate_merge",
]


class AdaptationKind(StrEnum):
    """How a model was specialised from a base."""

    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    PREFIX = "prefix"
    PROMPT_TUNING = "prompt_tuning"
    FULL = "full"  # every parameter updated; not really an adapter
    IA3 = "ia3"
    BITFIT = "bitfit"

    @property
    def is_parameter_efficient(self) -> bool:
        return self is not AdaptationKind.FULL


@dataclass(frozen=True)
class Adaptation:
    """A fine-tuning delta over a base model."""

    name: str
    base: str
    kind: AdaptationKind = AdaptationKind.LORA
    rank: int = 0
    alpha: float = 0.0
    target_modules: tuple[str, ...] = ()
    trainable_parameters: int = 0
    base_parameters: int = 0
    training_data: tuple[DatasetId, ...] = ()
    licence: str = ""
    registry: str = "local"

    @property
    def dataset(self) -> DatasetId:
        return adapter_asset(self.name, registry=self.registry)

    @property
    def base_dataset(self) -> DatasetId:
        return model(self.base, registry=self.registry)


def trainable_fraction(adaptation: Adaptation) -> float:
    """What share of the model this adaptation actually moves.

    A LoRA at rank 8 touches well under a percent, which is why the base model
    dominates the result and therefore the licence.
    """
    if adaptation.base_parameters <= 0:
        return 0.0
    return adaptation.trainable_parameters / adaptation.base_parameters


def adapter_edges(adaptation: Adaptation) -> list[tuple[DatasetId, DatasetId]]:
    """Two parents: the base model, and the data that tuned it.

    Only the second is yours, which is exactly why both have to be edges.
    """
    target = adaptation.dataset
    return [(adaptation.base_dataset, target)] + [
        (source, target) for source in adaptation.training_data
    ]


@dataclass(frozen=True)
class Merge:
    """A model produced by combining others."""

    name: str
    sources: tuple[str, ...]
    method: str = "linear"
    weights: tuple[float, ...] = ()
    registry: str = "local"

    @property
    def dataset(self) -> DatasetId:
        return model(self.name, registry=self.registry)


def merge_edges(merge: Merge) -> list[tuple[DatasetId, DatasetId]]:
    target = merge.dataset
    return [(model(s, registry=merge.registry), target) for s in merge.sources]


def validate_merge(merge: Merge) -> list[str]:
    """Problems that make a merge unreproducible or wrong."""
    problems: list[str] = []
    if not merge.sources:
        problems.append("a merge needs at least one source")
    if merge.weights and len(merge.weights) != len(merge.sources):
        problems.append(f"{len(merge.weights)} weights for {len(merge.sources)} sources")
    if merge.weights:
        total = sum(merge.weights)
        if merge.method == "linear" and abs(total - 1.0) > 1e-6:
            problems.append(
                f"linear merge weights sum to {total:.6g}, not 1.0; the result is scaled"
            )
    if len(set(merge.sources)) != len(merge.sources):
        problems.append("a source appears twice; its weight is effectively doubled")
    return problems


@dataclass(frozen=True)
class Distillation:
    """A student trained to imitate a teacher."""

    student: str
    teacher: str
    corpus: tuple[DatasetId, ...] = ()
    temperature: float = 1.0
    method: str = "logit"  # logit | sequence | feature
    registry: str = "local"


def distillation_edges(distillation: Distillation) -> list[tuple[DatasetId, DatasetId]]:
    """The teacher is an input to the student, and so is the corpus it spoke about.

    This edge is the one people forget, and it is the one that matters: a student
    distilled from a teacher inherits the teacher's training data obligations even
    though it never saw that data.
    """
    target = model(distillation.student, registry=distillation.registry)
    return [(model(distillation.teacher, registry=distillation.registry), target)] + [
        (source, target) for source in distillation.corpus
    ]


class QuantizationFormat(StrEnum):
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    FP8 = "fp8"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"


@dataclass(frozen=True)
class Quantization:
    """A compressed variant of a model. A derivation, not a deployment detail."""

    name: str
    source: str
    format: QuantizationFormat = QuantizationFormat.INT8
    group_size: int = 0
    calibration_data: tuple[DatasetId, ...] = ()
    registry: str = "local"

    @property
    def bits(self) -> int:
        return {
            QuantizationFormat.INT8: 8,
            QuantizationFormat.FP8: 8,
            QuantizationFormat.INT4: 4,
            QuantizationFormat.NF4: 4,
            QuantizationFormat.GPTQ: 4,
            QuantizationFormat.AWQ: 4,
            QuantizationFormat.GGUF: 4,
        }[self.format]


def quantization_edges(quantization: Quantization) -> list[tuple[DatasetId, DatasetId]]:
    """Calibration data is an input. It is small, and it is still training data."""
    target = model(quantization.name, registry=quantization.registry)
    return [(model(quantization.source, registry=quantization.registry), target)] + [
        (source, target) for source in quantization.calibration_data
    ]


# -- preference data -----------------------------------------------------------


@dataclass(frozen=True)
class Annotator:
    """Whoever produced a label, and under what terms."""

    identifier: str
    cohort: str = ""
    consent_scope: str = ""
    locale: str = ""


@dataclass(frozen=True)
class PreferencePair:
    """One ranked comparison."""

    prompt_digest: str
    chosen_digest: str
    rejected_digest: str
    annotator: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class PreferenceSet:
    """Ranked pairs for DPO or RLHF."""

    name: str
    pairs: tuple[PreferencePair, ...] = ()
    annotators: tuple[Annotator, ...] = ()
    source_annotations: tuple[str, ...] = ()
    store: str = "local"

    @property
    def dataset(self) -> DatasetId:
        return preference_set(self.name, store=self.store)


def preference_edges(preferences: PreferenceSet) -> list[tuple[DatasetId, DatasetId]]:
    """Preference sets are downstream of the annotation sets that produced them.

    Three edges from a model to a person: model -> preferences -> annotations ->
    annotator cohort. Nothing finds that chain unless it is recorded.
    """
    target = preferences.dataset
    return [
        (annotation_set(name, store=preferences.store), target)
        for name in preferences.source_annotations
    ]


def annotator_diversity(preferences: PreferenceSet) -> dict[str, float]:
    """How concentrated the labelling is.

    A preference set where three annotators produced 80% of the pairs encodes three
    people's taste, and the model will too.
    """
    counts: dict[str, int] = {}
    for pair in preferences.pairs:
        if pair.annotator:
            counts[pair.annotator] = counts.get(pair.annotator, 0) + 1
    total = sum(counts.values())
    if not total:
        return {"annotators": 0.0, "gini": 0.0, "top_share": 0.0, "pairs": 0.0}

    shares = sorted(c / total for c in counts.values())
    n = len(shares)
    # Gini over annotator workload: 0 is perfectly even, approaching 1 is one person.
    cumulative = sum((i + 1) * s for i, s in enumerate(shares))
    gini = (2 * cumulative) / (n * sum(shares)) - (n + 1) / n if n > 1 else 0.0
    return {
        "annotators": float(n),
        "gini": round(gini, 4),
        "top_share": round(max(shares), 4),
        "pairs": float(total),
    }


def inter_annotator_agreement(
    first: Sequence[PreferencePair], second: Sequence[PreferencePair]
) -> float:
    """Fraction of shared prompts where two annotators chose the same response.

    Below about 0.7 the signal is mostly noise, and a DPO run on it is fitting
    disagreement.
    """
    left = {p.prompt_digest: p.chosen_digest for p in first}
    right = {p.prompt_digest: p.chosen_digest for p in second}
    shared = set(left) & set(right)
    if not shared:
        return 0.0
    return sum(1 for k in shared if left[k] == right[k]) / len(shared)


def rank_stability(preferences: PreferenceSet) -> dict[str, float]:
    """Whether the same prompt gets ranked consistently.

    Contradictory pairs are not noise to be averaged out; they are prompts where the
    preference is genuinely contested, and training on both teaches nothing.
    """
    seen: dict[str, set[str]] = {}
    for pair in preferences.pairs:
        seen.setdefault(pair.prompt_digest, set()).add(pair.chosen_digest)
    repeated = {k: v for k, v in seen.items() if len(v) > 1}
    total = len(seen)
    return {
        "prompts": float(total),
        "contradictory": float(len(repeated)),
        "stability": round(1.0 - (len(repeated) / total), 4) if total else 1.0,
    }


# -- derivation traversal ------------------------------------------------------


@dataclass(frozen=True)
class Derivation:
    """One step in how a model came to exist."""

    source: str
    target: str
    kind: str
    detail: str = ""


def derivation_chain(
    derivations: Iterable[Derivation], target: str, *, max_depth: int = 32
) -> list[Derivation]:
    """Walk back from a model to its roots.

    Cycle-safe and depth-bounded, because a merge of two models that were themselves
    merged from each other is a config people write.
    """
    index: dict[str, list[Derivation]] = {}
    for d in derivations:
        index.setdefault(d.target, []).append(d)

    chain: list[Derivation] = []
    seen: set[str] = set()
    frontier = [target]
    for _ in range(max_depth):
        if not frontier:
            break
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        for d in index.get(current, ()):
            chain.append(d)
            frontier.append(d.source)
    return chain


def base_of(derivations: Iterable[Derivation], target: str) -> list[str]:
    """The root models a target ultimately derives from."""
    chain = derivation_chain(derivations, target)
    sources = {d.source for d in chain}
    targets = {d.target for d in chain}
    return sorted(sources - targets)


def lineage_depth(derivations: Iterable[Derivation], target: str) -> int:
    """How many derivation steps deep a model is."""
    index: dict[str, list[Derivation]] = {}
    for d in derivations:
        index.setdefault(d.target, []).append(d)

    def walk(name: str, seen: frozenset[str]) -> int:
        if name in seen:
            return 0
        parents = index.get(name, ())
        if not parents:
            return 0
        return 1 + max(walk(p.source, seen | {name}) for p in parents)

    return walk(target, frozenset())


def is_derived_from(derivations: Iterable[Derivation], target: str, ancestor: str) -> bool:
    chain = derivation_chain(derivations, target)
    return any(d.source == ancestor for d in chain)


def effective_restrictions(
    licences: Mapping[str, Iterable[str]], components: Iterable[str]
) -> set[str]:
    """The union of every component's restrictions.

    Restrictively, not permissively. A permissive adapter over a research-only base
    is research-only, and teams get this backwards because the adapter is the part
    they wrote.
    """
    combined: set[str] = set()
    for name in components:
        combined.update(licences.get(name, ()))
    return combined
