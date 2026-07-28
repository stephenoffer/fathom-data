"""Adapters over the three surfaces: engines, catalogs, and storage.

Every adapter is imported here so `fathom adapters` lists them all. The Iceberg
adapter defers its `pyiceberg` import to the methods that need it, so importing it
without the extra installed costs nothing and fails only when actually used.
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
from .bigquery import BigQueryAdapter
from .databricks import DatabricksAdapter
from .delta import DeltaCatalog
from .duckdb_engine import DuckDBEngine
from .iceberg import IcebergCatalog
from .predicates import literal, render_predicate
from .snowflake import SnowflakeAdapter
from .sql_runner import DBAPIRunner, QueryError, QueryRunner, RecordedRunner
from .storage import LocalStorage, ObjectStorage

__all__ = [
    "BigQueryAdapter",
    "CatalogAdapter",
    "DBAPIRunner",
    "DatabricksAdapter",
    "ChangeSet",
    "DeclaredCatalog",
    "DeltaCatalog",
    "DuckDBEngine",
    "EngineAdapter",
    "IcebergCatalog",
    "LineageEvent",
    "LocalStorage",
    "ObjectStorage",
    "ObjectMeta",
    "QueryError",
    "QueryRunner",
    "RecordedRunner",
    "SnowflakeAdapter",
    "QueryEvent",
    "StorageAdapter",
    "Token",
    "get_adapter",
    "register",
    "registered",
    "literal",
    "render_predicate",
]
