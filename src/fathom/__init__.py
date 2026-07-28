"""fathom — lineage, partition-scoped invalidation, profiling, and policy propagation.

Two durable artifacts, four verbs:

    dependency graph  +  profile history
        plan   what must be rebuilt, and only that
        check  what drifted, and what upstream caused it
        label  what a column means, and what policy applies to it
        erase  where a subject's data physically lives, and how to destroy it

Everything is built on adapters over three surfaces — engines (execution plans and
query logs), catalogs (table and partition metadata), and storage (objects, events,
inventories) — so the same graph spans a warehouse, a lakehouse, and a raw bucket.
"""

from .erasure import ErasurePlan, ErasureProof, ErasureRequest, apply_erasure, plan_erasure
from .grains import Grain
from .graph import Edge, Graph, InvalidationPlan
from .ids import AliasRegistry, normalize, normalize_path, normalize_table
from .ingest import IngestResult, graph_from_lineage, graph_from_queries, ingest_engine
from .partitions import (
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
from .paths import PathTemplate, key_from_path
from .policy import Label, SinkPolicy, Violation, enforce, infer, propagate
from .profile import ColumnProfile, Finding, Profile, Severity, drift, profile_parquet
from .shadow import ShadowReport, ShadowResult
from .store import ShadowObservation, Store
from .types import (
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

__version__ = "0.1.0"

__all__ = [
    "ANY",
    "UNBOUNDED",
    "UNPARTITIONED",
    "AliasRegistry",
    "Capabilities",
    "ChangeSource",
    "ColumnProfile",
    "ColumnRef",
    "DatasetId",
    "Edge",
    "ErasureMode",
    "ErasurePlan",
    "ErasureProof",
    "ErasureRequest",
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
    "Store",
    "TimeWindow",
    "Unbounded",
    "Violation",
    "__version__",
    "apply",
    "apply_erasure",
    "compose",
    "covered_by",
    "drift",
    "enforce",
    "graph_from_lineage",
    "graph_from_queries",
    "infer",
    "ingest_engine",
    "join",
    "key_from_path",
    "leq",
    "normalize",
    "normalize_path",
    "normalize_table",
    "plan_erasure",
    "profile_parquet",
    "propagate",
    "subsumes",
]
