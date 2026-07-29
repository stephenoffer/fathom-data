"""Lineage for the things AI systems are made of.

The premise is one line long: a model, a feature view, a vector index, a prompt, and
an eval set are all datasets, so the graph, the planner, the profiler, the policy
engine, and the eraser already work on them. There is no second graph for ML, which
matters because a second graph is a graph that disagrees with the first one.

    assets       identities and partition specs for every AI asset kind
    training     training runs as edges: bills of material, retraining plans
    features     feature views, target leakage, training/serving skew
    vectors      embeddings and indexes, and re-embedding only what changed
    rag          what entered a context window, and what that means for policy
    prompts      prompts as versioned datasets whose variables carry data in
    evals        eval sets, and whether their scores can be believed
    agents       what an autonomous program actually read, wrote, and could leak
    attribution  turning a drift alert into a ranked diagnosis
    unlearning   erasure that reaches the model, and honesty about where it stops
    train/       the run, the checkpoint, and everything downstream of a base model
    quality/     contamination between a training corpus and an eval set
    serve/       endpoints, traffic splits, and the rollback that has to work

This package sits above `govern` and below nothing. A new asset kind is a member of
`AssetKind`, a constructor and a spec in `assets`, and a module here — no changes
anywhere else in the library.
"""

from . import (
    agents,
    assets,
    attribution,
    evals,
    features,
    prompts,
    quality,
    rag,
    serve,
    train,
    training,
    unlearning,
    vectors,
)
from .assets import (
    AssetKind,
    AssetRef,
    adapter,
    agent,
    checkpoint,
    corpus,
    embedding_space,
    eval_set,
    feature_view,
    is_ai_asset,
    is_model,
    is_vector_index,
    kind_of,
    model,
    prompt,
    spec_for,
    tool,
    vector_index,
)

__all__ = [
    "AssetKind",
    "AssetRef",
    "adapter",
    "agent",
    "agents",
    "assets",
    "quality",
    "serve",
    "train",
    "attribution",
    "checkpoint",
    "corpus",
    "embedding_space",
    "eval_set",
    "evals",
    "feature_view",
    "features",
    "is_ai_asset",
    "is_model",
    "is_vector_index",
    "kind_of",
    "model",
    "prompt",
    "prompts",
    "rag",
    "spec_for",
    "tool",
    "training",
    "unlearning",
    "vector_index",
    "vectors",
]
