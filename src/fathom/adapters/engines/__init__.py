"""Engines: where lineage comes from, and where rebuilds are issued.

    duckdb      renders and executes partition-scoped rebuilds locally
    snowflake   native column lineage from ACCESS_HISTORY
    databricks  Unity Catalog lineage tables
    bigquery    INFORMATION_SCHEMA jobs and partition metadata

Adding one is a module here plus a line in `adapters/__init__`. The protocol in
`adapters.base` is deliberately three methods wide, and an engine that implements
only `fetch_queries` still works — the graph is coarser, not absent.
"""

from .bigquery import BigQueryAdapter
from .databricks import DatabricksAdapter
from .duckdb import DuckDBEngine
from .snowflake import SnowflakeAdapter

__all__ = ["BigQueryAdapter", "DatabricksAdapter", "DuckDBEngine", "SnowflakeAdapter"]
