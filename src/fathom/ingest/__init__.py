"""How the graph is learned.

Ordered by how much you should trust it. Native lineage a platform maintains beats
anything derived; a dbt manifest beats parsing the SQL dbt compiled; parsing the
query log beats a human declaring edges by hand. Every source lands in the same
`Graph`, so a warehouse with native lineage and a bucket with none coexist.

    events        native lineage and query-log events into a graph
    sql           lineage from SQL text, via sqlglot, for every dialect
    dbt           a dbt manifest, including its partition config
    openlineage   events any OpenLineage producer already emits

A new source is a new module here that returns an `IngestResult`. Nothing else in
the codebase needs to know it exists.
"""

from .dbt import DbtManifest, ingest_dbt, load_manifest, parse_manifest
from .events import IngestResult, graph_from_lineage, graph_from_queries, ingest_engine
from .openlineage import (
    OpenLineageRun,
    ingest_openlineage,
    load_events,
    parse_event,
    read_events,
)
from .sql import Extraction, extract

__all__ = [
    "DbtManifest",
    "Extraction",
    "IngestResult",
    "OpenLineageRun",
    "extract",
    "graph_from_lineage",
    "graph_from_queries",
    "ingest_dbt",
    "ingest_engine",
    "ingest_openlineage",
    "load_events",
    "load_manifest",
    "parse_event",
    "parse_manifest",
    "read_events",
]
