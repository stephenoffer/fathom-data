"""Identities for the things AI systems are built out of.

A model is a dataset. So is a feature view, a vector index, a prompt template, an
eval suite, and an agent run. That is not a metaphor for the sake of reuse — it is
the observation this whole package rests on. Each of them:

- is produced from named inputs, which is an edge
- is produced in slices that can be rebuilt independently, which is a partition spec
- has a version history whose differences someone will need to explain, which is a
  profile
- can contain a person's data, which is a label, and therefore an erasure obligation

Once they carry `DatasetId`s, the planner, the profiler, the policy engine, and the
eraser already work on them. There is no second graph for ML, which matters because
a second graph is a graph that disagrees with the first one.

Namespaces follow the OpenLineage shape — `scheme://instance` — with schemes
reserved for each asset kind, so `model://registry.internal` sits beside
`s3://lake` in the same graph and nothing downstream has to special-case it.

Partitioning is the part worth reading twice. A model partitioned by `version` is
a model whose retraining can be planned like a table's rebuild. A vector index
partitioned by `shard` and `dt` is an index whose re-embedding cost scales with what
changed rather than with corpus size — and re-embedding is metered per token, so
that difference is a line item rather than an optimization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ..core.grains import Grain
from ..core.types import DatasetId, PartitionField, PartitionSpec

__all__ = [
    "AssetKind",
    "AssetRef",
    "adapter",
    "describe",
    "instance_of",
    "kinds_present",
    "parse_ref",
    "agent",
    "checkpoint",
    "corpus",
    "dataset_kind",
    "embedding_space",
    "eval_set",
    "feature_view",
    "is_ai_asset",
    "is_model",
    "is_vector_index",
    "kind_of",
    "model",
    "media_corpus",
    "safety_suite",
    "deployment",
    "annotation_set",
    "preference_set",
    "mixture",
    "tokenizer",
    "sweep",
    "run",
    "prompt",
    "spec_for",
    "tool",
    "vector_index",
]


class AssetKind(StrEnum):
    """What an identity denotes. `TABLE` is everything that predates this module."""

    TABLE = "table"
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    ADAPTER = "adapter"  # LoRA and friends: a delta over a base model
    FEATURE_VIEW = "feature"
    EMBEDDING_SPACE = "embedding"
    VECTOR_INDEX = "index"
    PROMPT = "prompt"
    EVAL_SET = "eval"
    CORPUS = "corpus"
    AGENT = "agent"
    TOOL = "tool"
    # Added as the package grew past "what was trained" into "how, and what happened
    # next". Each denotes something with its own inputs, versions, and obligations.
    RUN = "run"  # one training or evaluation execution
    SWEEP = "sweep"  # a set of runs varying hyperparameters
    TOKENIZER = "tokenizer"  # a vocabulary; changing it is a schema change for text
    MIXTURE = "mixture"  # corpus sampling weights
    PREFERENCE_SET = "preference"  # ranked pairs for DPO/RLHF
    ANNOTATION = "annotation"  # human labels, with annotator provenance
    DEPLOYMENT = "deployment"  # a served endpoint, downstream of a model
    SAFETY_SUITE = "safety"  # red-team findings turned into regressions
    MEDIA = "media"  # image, audio, and video corpora


# Scheme reserved per kind. Kept explicit rather than derived from the enum value so
# renaming an enum member cannot silently repartition an existing graph.
_SCHEMES: dict[AssetKind, str] = {
    AssetKind.MODEL: "model",
    AssetKind.CHECKPOINT: "checkpoint",
    AssetKind.ADAPTER: "adapter",
    AssetKind.FEATURE_VIEW: "feature",
    AssetKind.EMBEDDING_SPACE: "embedding",
    AssetKind.VECTOR_INDEX: "index",
    AssetKind.PROMPT: "prompt",
    AssetKind.EVAL_SET: "eval",
    AssetKind.CORPUS: "corpus",
    AssetKind.AGENT: "agent",
    AssetKind.TOOL: "tool",
    AssetKind.RUN: "run",
    AssetKind.SWEEP: "sweep",
    AssetKind.TOKENIZER: "tokenizer",
    AssetKind.MIXTURE: "mixture",
    AssetKind.PREFERENCE_SET: "preference",
    AssetKind.ANNOTATION: "annotation",
    AssetKind.DEPLOYMENT: "deployment",
    AssetKind.SAFETY_SUITE: "safety",
    AssetKind.MEDIA: "media",
}

_BY_SCHEME = {scheme: kind for kind, scheme in _SCHEMES.items()}

_NAMESPACE = re.compile(r"^(?P<scheme>[a-z]+)://(?P<instance>.*)$")


def _identity(kind: AssetKind, name: str, instance: str) -> DatasetId:
    scheme = _SCHEMES[kind]
    cleaned = name.strip().strip("/")
    if not cleaned:
        raise ValueError(f"a {kind.value} needs a name")
    return DatasetId(namespace=f"{scheme}://{instance.strip().lower()}", name=cleaned)


# -- constructors --------------------------------------------------------------


def model(name: str, *, registry: str = "local") -> DatasetId:
    """A trained model in a registry. ``model("fraud.scorer")``."""
    return _identity(AssetKind.MODEL, name, registry)


def checkpoint(name: str, *, registry: str = "local") -> DatasetId:
    """One saved state of a training run, distinct from the model it eventually becomes."""
    return _identity(AssetKind.CHECKPOINT, name, registry)


def run(name: str, *, tracker: str = "local") -> DatasetId:
    """One training or evaluation execution. ``run("pretrain/2026-07-14/a3f")``.

    Distinct from the checkpoint it writes and the model it becomes: a run has
    hyperparameters, a status, and a comparison to the run before it, none of which
    belong to the artefact it produced.
    """
    return _identity(AssetKind.RUN, name, tracker)


def sweep(name: str, *, tracker: str = "local") -> DatasetId:
    """A set of runs varying hyperparameters, and the thing a trial belongs to."""
    return _identity(AssetKind.SWEEP, name, tracker)


def tokenizer(name: str, *, registry: str = "local") -> DatasetId:
    """A vocabulary. Changing one is a schema change for every text asset downstream."""
    return _identity(AssetKind.TOKENIZER, name, registry)


def mixture(name: str, *, registry: str = "local") -> DatasetId:
    """Corpus sampling weights — the highest-leverage pretraining decision there is."""
    return _identity(AssetKind.MIXTURE, name, registry)


def preference_set(name: str, *, store: str = "local") -> DatasetId:
    """Ranked pairs for DPO or RLHF, downstream of the annotations that produced them."""
    return _identity(AssetKind.PREFERENCE_SET, name, store)


def annotation_set(name: str, *, store: str = "local") -> DatasetId:
    """Human labels, carrying annotator provenance and therefore consent obligations."""
    return _identity(AssetKind.ANNOTATION, name, store)


def deployment(name: str, *, environment: str = "prod") -> DatasetId:
    """A served endpoint. Downstream of a model, and where users actually meet it."""
    return _identity(AssetKind.DEPLOYMENT, name, environment)


def safety_suite(name: str, *, registry: str = "local") -> DatasetId:
    """Red-team findings turned into a regression suite rather than a closed ticket."""
    return _identity(AssetKind.SAFETY_SUITE, name, registry)


def media_corpus(name: str, *, store: str = "local") -> DatasetId:
    """An image, audio, or video corpus. Partitioned, drifting, and full of people."""
    return _identity(AssetKind.MEDIA, name, store)


def adapter(name: str, *, registry: str = "local") -> DatasetId:
    """A fine-tuning delta over a base model — LoRA weights, a prefix, a soft prompt.

    Kept separate from `model` because its lineage is genuinely different: it has two
    parents, the base model and the tuning data, and only the second is yours.
    """
    return _identity(AssetKind.ADAPTER, name, registry)


def feature_view(name: str, *, store: str = "local") -> DatasetId:
    """A named group of features materialized on a schedule."""
    return _identity(AssetKind.FEATURE_VIEW, name, store)


def embedding_space(name: str, *, provider: str = "local") -> DatasetId:
    """An embedding model and its output space, together.

    They are one identity because a vector is meaningless without the space it was
    produced in: re-embedding with a new model version invalidates every vector,
    and treating the space as its own node is what makes that an edge to follow.
    """
    return _identity(AssetKind.EMBEDDING_SPACE, name, provider)


def vector_index(name: str, *, store: str = "local") -> DatasetId:
    """A searchable index of embeddings — pgvector, Pinecone, LanceDB, FAISS on disk."""
    return _identity(AssetKind.VECTOR_INDEX, name, store)


def prompt(name: str, *, repo: str = "local") -> DatasetId:
    """A versioned prompt or template.

    Prompts belong in lineage for the same reason SQL does: they decide what the
    output is made of, and changing one changes downstream results with no schema
    change to notice.
    """
    return _identity(AssetKind.PROMPT, name, repo)


def eval_set(name: str, *, suite: str = "local") -> DatasetId:
    """A held-out set used to grade a model."""
    return _identity(AssetKind.EVAL_SET, name, suite)


def corpus(name: str, *, store: str = "local") -> DatasetId:
    """A document collection feeding retrieval or pre-training."""
    return _identity(AssetKind.CORPUS, name, store)


def agent(name: str, *, runtime: str = "local") -> DatasetId:
    """An autonomous program that reads and writes datasets on its own initiative.

    Worth a node of its own precisely because nobody reviewed its access pattern in
    a pull request the way they would a dbt model's.
    """
    return _identity(AssetKind.AGENT, name, runtime)


def tool(name: str, *, runtime: str = "local") -> DatasetId:
    """A callable an agent can invoke, and therefore a place data can leave through."""
    return _identity(AssetKind.TOOL, name, runtime)


# -- classification ------------------------------------------------------------


def kind_of(ds: DatasetId) -> AssetKind:
    """What kind of asset an identity denotes. Anything unrecognized is a table."""
    match = _NAMESPACE.match(ds.namespace)
    scheme = match.group("scheme") if match else ds.namespace
    return _BY_SCHEME.get(scheme.lower(), AssetKind.TABLE)


def dataset_kind(ds: DatasetId) -> str:
    """`kind_of` as a plain string, for JSON output and log lines."""
    return kind_of(ds).value


def is_ai_asset(ds: DatasetId) -> bool:
    """True for anything this module defines, false for ordinary tables and files."""
    return kind_of(ds) is not AssetKind.TABLE


def is_model(ds: DatasetId) -> bool:
    """True for models, checkpoints, and adapters — anything with weights."""
    return kind_of(ds) in {AssetKind.MODEL, AssetKind.CHECKPOINT, AssetKind.ADAPTER}


def is_vector_index(ds: DatasetId) -> bool:
    """True for vector indexes and the embedding spaces behind them."""
    return kind_of(ds) in {AssetKind.VECTOR_INDEX, AssetKind.EMBEDDING_SPACE}


def instance_of(ds: DatasetId) -> str:
    """The registry, store, or runtime an asset lives in."""
    match = _NAMESPACE.match(ds.namespace)
    return match.group("instance") if match else ""


# -- partitioning --------------------------------------------------------------

# One spec per kind, chosen so that "what has to be rebuilt" is answerable.
_SPECS: dict[AssetKind, PartitionSpec] = {
    # A model version is a value, not a time: v3 does not roll up into a month.
    AssetKind.MODEL: PartitionSpec.of(PartitionField.value("version")),
    AssetKind.CHECKPOINT: PartitionSpec.of(
        PartitionField.value("version"), PartitionField.value("step")
    ),
    AssetKind.ADAPTER: PartitionSpec.of(PartitionField.value("version")),
    # Feature views are materialized per entity slice per day, like any table.
    AssetKind.FEATURE_VIEW: PartitionSpec.of(
        PartitionField.time("dt", Grain.DAY), PartitionField.value("entity")
    ),
    # An embedding space is versioned by the model that defines it.
    AssetKind.EMBEDDING_SPACE: PartitionSpec.of(PartitionField.value("model_version")),
    # Shard is what makes partial re-embedding possible; dt is what makes it cheap.
    AssetKind.VECTOR_INDEX: PartitionSpec.of(
        PartitionField.value("shard"), PartitionField.time("dt", Grain.DAY)
    ),
    AssetKind.PROMPT: PartitionSpec.of(PartitionField.value("version")),
    AssetKind.EVAL_SET: PartitionSpec.of(PartitionField.value("version")),
    AssetKind.CORPUS: PartitionSpec.of(
        PartitionField.time("dt", Grain.DAY), PartitionField.value("source")
    ),
    AssetKind.AGENT: PartitionSpec.of(PartitionField.time("dt", Grain.HOUR)),
    AssetKind.TOOL: PartitionSpec.of(PartitionField.time("dt", Grain.HOUR)),
}


def spec_for(kind: AssetKind | DatasetId) -> PartitionSpec:
    """The conventional partition spec for an asset kind.

    A default, not a rule. Declaring a different one on the graph overrides it, and
    should whenever the real materialization differs — a spec that lies costs more
    than no spec at all, because the planner trusts it.
    """
    resolved = kind if isinstance(kind, AssetKind) else kind_of(kind)
    return _SPECS.get(resolved, PartitionSpec())


@dataclass(frozen=True)
class AssetRef:
    """An asset plus the slice of it being referred to."""

    dataset: DatasetId
    version: str = ""

    @property
    def kind(self) -> AssetKind:
        """Which kind of AI asset this identity denotes."""
        return kind_of(self.dataset)

    def __str__(self) -> str:
        return f"{self.dataset}@{self.version}" if self.version else str(self.dataset)


def parse_ref(text: str) -> AssetRef:
    """Parse ``model://registry/fraud.scorer@v3`` into an identity and a version."""
    body, _, version = text.partition("@")
    match = _NAMESPACE.match(body)
    if match is None:
        raise ValueError(f"{text!r} is not an asset reference; expected scheme://instance/name")
    instance, _, name = match.group("instance").partition("/")
    if not name:
        raise ValueError(f"{text!r} names no asset")
    namespace = f"{match.group('scheme')}://{instance}"
    return AssetRef(dataset=DatasetId(namespace=namespace, name=name), version=version)


def describe(ds: DatasetId) -> str:
    """A one-line human description of what an identity is.

    Used wherever an identity is shown to someone who did not build the graph, which
    in practice is most readers of an audit report.
    """
    kind = kind_of(ds)
    where = instance_of(ds)
    nouns = {
        AssetKind.TABLE: "dataset",
        AssetKind.MODEL: "model",
        AssetKind.CHECKPOINT: "training checkpoint",
        AssetKind.ADAPTER: "fine-tuning adapter",
        AssetKind.FEATURE_VIEW: "feature view",
        AssetKind.EMBEDDING_SPACE: "embedding space",
        AssetKind.VECTOR_INDEX: "vector index",
        AssetKind.PROMPT: "prompt",
        AssetKind.EVAL_SET: "eval set",
        AssetKind.CORPUS: "corpus",
        AssetKind.AGENT: "agent",
        AssetKind.TOOL: "tool",
    }
    noun = nouns[kind]
    return f"{ds.name} — {noun}" + (f" in {where}" if where else "")


def kinds_present(datasets: list[DatasetId]) -> dict[str, int]:
    """Count assets by kind. The inventory line of any lineage report."""
    counts: dict[str, int] = {}
    for ds in datasets:
        key = kind_of(ds).value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
