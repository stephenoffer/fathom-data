"""Everything that talks to another system, arranged by surface.

Three surfaces, because "what depends on what" and "what changed" come from
different places and neither assumes SQL:

    engines/    execution plans and query logs — Snowflake, Databricks, BigQuery, DuckDB
    catalogs/   table and partition metadata — Delta, Iceberg
    storage/    objects, prefixes, and etags — S3, GCS, ADLS, R2, MinIO, local

    base        the three protocols, the capability record, and the registry
    fs          one filesystem abstraction under all of storage
    predicates  partition keys rendered as SQL, per dialect
    sql_runner  a DB-API seam so engines are testable without a warehouse

Adapters declare capabilities rather than implement everything. One reporting
`LIST_DIFF` and `Pushdown.NONE` still works — it is slower and coarser, and the
planner degrades instead of failing. That is what makes the long tail reachable:
a new system starts as a `DeclaredCatalog` and earns precision later.
"""

from __future__ import annotations

from .base import (
    CatalogAdapter,
    ChangeSet,
    DeclaredCatalog,
    EngineAdapter,
    LineageEvent,
    ObjectMeta,
    QueryEvent,
    StorageAdapter,
    Token,
    get_adapter,
    register,
    registered,
)
from .catalogs import DeltaCatalog, IcebergCatalog
from .engines import BigQueryAdapter, DatabricksAdapter, DuckDBEngine, SnowflakeAdapter
from .fs import FileInfo, FileSystem, FsspecFileSystem, filesystem_for
from .predicates import literal, render_predicate
from .sql_runner import DBAPIRunner, QueryError, QueryRunner, RecordedRunner
from .storage import LocalStorage, ObjectStorage

__all__ = [
    "BigQueryAdapter",
    "CatalogAdapter",
    "ChangeSet",
    "DBAPIRunner",
    "DatabricksAdapter",
    "DeclaredCatalog",
    "DeltaCatalog",
    "DuckDBEngine",
    "EngineAdapter",
    "FileInfo",
    "FileSystem",
    "FsspecFileSystem",
    "IcebergCatalog",
    "LineageEvent",
    "LocalStorage",
    "ObjectMeta",
    "ObjectStorage",
    "QueryError",
    "QueryEvent",
    "QueryRunner",
    "RecordedRunner",
    "SnowflakeAdapter",
    "StorageAdapter",
    "Token",
    "filesystem_for",
    "get_adapter",
    "literal",
    "register",
    "registered",
    "render_predicate",
]
