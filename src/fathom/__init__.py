"""fathom — lineage, partition-scoped invalidation, profiling, and policy propagation.

Two durable artifacts, four verbs:

    dependency graph  +  profile history
        plan   what must be rebuilt, and only that
        check  what drifted, and what upstream caused it
        label  what a column means, and what policy applies to it
        erase  where a subject's data physically lives, and how to destroy it

The same four verbs cover models, feature views, vector indexes, prompts, and eval
sets, because those are datasets too — see `fathom.ai`. Obligations that travel with
the data rather than with the code live in `fathom.govern`.

**Layout.** Packages follow the lifecycle, so a new capability has one obvious home
and the tree stays shallow at any single level:

    core/        the IR: identity, grains, the partition lattice, codecs, errors
    ingest/      how the graph is learned: SQL, native events, dbt, OpenLineage
    graph/       the graph, traversal, selection, diffing, coverage, history, plans
    observe/     profiles, expectations, drift, completeness, shadow mode, freshness
    govern/      labels, erasure, licences, consent, contracts, compliance records
    ai/          models, features, vectors, RAG context, prompts, evals, agents
    adapters/    everything that talks to another system, by surface
    store/       persistence for the two durable artifacts
    report/      rendering out: Mermaid, DOT, Markdown, OpenLineage, DataHub
    cli/         the command line, project config, and `fathom.yml`

**Namespace.** The tree is deep; the import surface is not. The names below are
re-exported here, and the module aliases mean `from fathom import query, cost` keeps
working regardless of how deep the file that defines them sits:

    query selectors diff metrics history        over the graph
    cost schedule                               over a plan
    profile quality seasonal shadow freshness   over the data
    completeness usage                          over what should be there, and who reads it
    policy erasure licenses consent contracts   over the obligations
    reidentification                            over what the columns jointly reveal
    render emit                                 out to people and other tools
    ai govern adapters ingest store             the packages themselves
"""

from . import adapters, ai, govern, ingest, store
from .ai import assets
from .core.errors import AdapterUnavailable, ConfigError, FathomError, StorageAccessError
from .core.grains import Grain
from .core.ids import AliasRegistry, normalize, normalize_path, normalize_table
from .core.partitions import (
    UNBOUNDED,
    PartitionMapping,
    Passthrough,
    TimeWindow,
    Unbounded,
    apply,
    compose,
    join,
    leq,
)
from .core.paths import PathTemplate, key_from_path
from .core.types import (
    ANY,
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    ColumnRef,
    DatasetId,
    ErasureMode,
    KeyPredicate,
    LineageSource,
    PartitionField,
    PartitionSpec,
    Pushdown,
    covered_by,
    subsumes,
)
from .govern import contracts, erasure, licenses, policy, reidentification, replicas
from .govern.erasure import (
    ErasurePlan,
    ErasureProof,
    ErasureRequest,
    apply_erasure,
    plan_erasure,
)
from .govern.policy import Label, SinkPolicy, Violation, enforce, infer, propagate
from .graph import diff, history, metrics, query, selectors, sinks
from .graph.model import Edge, Graph, InvalidationPlan
from .graph.plan import billing, cost, lifetime, schedule
from .ingest.events import (
    IngestResult,
    graph_from_lineage,
    graph_from_queries,
    ingest_engine,
)
from .observe import (
    completeness,
    freshness,
    joins,
    profile,
    quality,
    seasonal,
    shadow,
    usage,
)
from .observe.profile import ColumnProfile, Finding, Profile, Severity, drift, profile_parquet
from .observe.shadow import ShadowObservation, ShadowReport, ShadowResult
from .report import emit, render
from .store.sqlite import Store

# The single source of truth for the version. `pyproject.toml` declares
# `dynamic = ["version"]` and reads it from here, so the wheel and `fathom --version`
# cannot disagree — they did, and the symptom was a reported version that matched no
# published artifact.
__version__ = "0.1.0"

__all__ = [
    "ANY",
    "UNBOUNDED",
    "UNPARTITIONED",
    "AdapterUnavailable",
    "AliasRegistry",
    "Capabilities",
    "ChangeSource",
    "ColumnProfile",
    "ColumnRef",
    "ConfigError",
    "DatasetId",
    "Edge",
    "ErasureMode",
    "ErasurePlan",
    "ErasureProof",
    "ErasureRequest",
    "FathomError",
    "Finding",
    "Grain",
    "Graph",
    "IngestResult",
    "InvalidationPlan",
    "KeyPredicate",
    "Label",
    "LineageSource",
    "PartitionField",
    "PartitionMapping",
    "PartitionSpec",
    "Passthrough",
    "PathTemplate",
    "Profile",
    "Pushdown",
    "Severity",
    "ShadowObservation",
    "ShadowReport",
    "ShadowResult",
    "SinkPolicy",
    "StorageAccessError",
    "Store",
    "TimeWindow",
    "Unbounded",
    "Violation",
    "__version__",
    "adapters",
    "ai",
    "apply",
    "apply_erasure",
    "assets",
    "billing",
    "completeness",
    "contracts",
    "compose",
    "consent",
    "cost",
    "covered_by",
    "diff",
    "drift",
    "emit",
    "enforce",
    "erasure",
    "freshness",
    "govern",
    "history",
    "graph_from_lineage",
    "graph_from_queries",
    "infer",
    "ingest",
    "ingest_engine",
    "join",
    "joins",
    "key_from_path",
    "leq",
    "licenses",
    "lifetime",
    "metrics",
    "normalize",
    "normalize_path",
    "normalize_table",
    "plan_erasure",
    "policy",
    "profile",
    "profile_parquet",
    "propagate",
    "quality",
    "query",
    "reidentification",
    "render",
    "replicas",
    "schedule",
    "seasonal",
    "selectors",
    "shadow",
    "sinks",
    "store",
    "subsumes",
    "usage",
]


def __getattr__(name: str) -> object:
    """Defer the heavier packages so `import fathom` stays cheap.

    `consent` pulls in the whole governance stack and `cli` pulls in click; neither
    belongs in the cost of importing the library to call `normalize`.
    """
    if name == "consent":
        from .govern import consent

        return consent
    if name == "cli":
        from . import cli

        return cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
