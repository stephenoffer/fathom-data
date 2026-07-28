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

from .grains import Grain
from .graph import Edge, Graph, InvalidationPlan
from .ids import AliasRegistry, normalize, normalize_path, normalize_table
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
)

__version__ = "0.0.1"

__all__ = [
    "ANY",
    "UNBOUNDED",
    "UNPARTITIONED",
    "AliasRegistry",
    "Capabilities",
    "ChangeSource",
    "ColumnRef",
    "DatasetId",
    "Edge",
    "ErasureMode",
    "Grain",
    "Graph",
    "InvalidationPlan",
    "KeyPredicate",
    "LineageSource",
    "PartitionField",
    "PartitionMapping",
    "PartitionSpec",
    "Passthrough",
    "Pushdown",
    "TimeWindow",
    "Unbounded",
    "__version__",
    "apply",
    "compose",
    "join",
    "leq",
    "normalize",
    "normalize_path",
    "normalize_table",
]
