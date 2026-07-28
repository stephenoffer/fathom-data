"""Adapters over the three surfaces: engines, catalogs, and storage.

Iceberg is imported lazily because it needs the `iceberg` extra; everything else
works from a bare install.
"""

from __future__ import annotations

from typing import Any

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
from .delta import DeltaCatalog
from .duckdb_engine import DuckDBEngine, render_predicate
from .local import LocalStorage

__all__ = [
    "CatalogAdapter",
    "ChangeSet",
    "DeclaredCatalog",
    "DeltaCatalog",
    "DuckDBEngine",
    "EngineAdapter",
    "IcebergCatalog",
    "LineageEvent",
    "LocalStorage",
    "ObjectMeta",
    "QueryEvent",
    "StorageAdapter",
    "Token",
    "get_adapter",
    "register",
    "registered",
    "render_predicate",
]


def __getattr__(name: str) -> Any:
    """Defer the Iceberg import so a bare install does not fail on `pyiceberg`."""
    if name == "IcebergCatalog":
        from .iceberg import IcebergCatalog

        return IcebergCatalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
