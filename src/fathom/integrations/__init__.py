"""Integrations with tools that already know part of the answer.

Preferring these over re-deriving everything ourselves is the whole strategy for
breadth. dbt knows the dependency graph; OpenLineage producers already run inside
Spark, Flink, Trino, Airflow, and Dagster. Consuming what they emit reaches far
more of the ecosystem than writing a listener per engine ever would.
"""

from .dbt import DbtManifest, ingest_dbt, load_manifest, parse_manifest
from .openlineage import (
    OpenLineageRun,
    ingest_openlineage,
    load_events,
    parse_event,
    read_events,
)

__all__ = [
    "DbtManifest",
    "OpenLineageRun",
    "ingest_dbt",
    "ingest_openlineage",
    "load_events",
    "load_manifest",
    "parse_event",
    "parse_manifest",
    "read_events",
]
