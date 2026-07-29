"""Building a graph from what adapters report.

Native lineage first, query-log parsing second, declarations last. When a platform
maintains lineage for us it is both cheaper and more accurate than re-deriving it
from SQL, so native events take precedence and parsing only fills the gaps.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..adapters.base import LineageEvent, QueryEvent
from ..core.partitions import PartitionMapping
from ..core.types import UNPARTITIONED, DatasetId, PartitionSpec
from ..graph.model import Edge, Graph
from .sql import extract

__all__ = ["IngestResult", "graph_from_lineage", "graph_from_queries", "ingest_engine"]


@dataclass
class IngestResult:
    """A graph plus everything we could not fully resolve while building it."""

    graph: Graph = field(default_factory=Graph)
    notes: list[str] = field(default_factory=list)
    unparsed: int = 0
    statements: int = 0

    @property
    def edges(self) -> int:
        """How many edges this ingest produced."""
        return len(self.graph.edges)

    def summary(self) -> str:
        """The result as text: edges, statements, and anything unparsed."""
        parts = [f"{self.edges} edge(s) from {self.statements} statement(s)"]
        if self.unparsed:
            parts.append(f"{self.unparsed} unparseable")
        if self.notes:
            parts.append(f"{len(self.notes)} note(s)")
        return ", ".join(parts)


def _seed(graph: Graph, specs: Mapping[DatasetId, PartitionSpec]) -> None:
    for ds, spec in specs.items():
        graph.add_dataset(ds, spec)


def graph_from_queries(
    queries: Iterable[QueryEvent],
    *,
    dialect: str,
    system: str | None = None,
    instance: str | None = None,
    specs: Mapping[DatasetId, PartitionSpec] | None = None,
    graph: Graph | None = None,
    evidence_label: str | None = None,
) -> IngestResult:
    """Parse statements into edges.

    Statements that fail to parse are counted, not raised. One malformed entry in a
    query log must not abort a whole ingest run.

    `evidence_label` fixes the evidence string for every edge produced. Pass it when
    the events carry a per-execution id, as a warehouse query log does. Evidence is
    an edge's identity in the store, so folding a `job_id` into it makes the same
    dependency a brand-new edge on every run: an hourly model accumulates 8,760 rows
    a year for one dependency, `fan_in` counts executions instead of inputs, and
    `edge_between` joins thousands of parallel edges — which widens. A filename or a
    dbt `unique_id` is stable and can safely stay in the evidence.
    """
    specs = dict(specs or {})
    result = IngestResult(graph=graph or Graph())
    _seed(result.graph, specs)

    for query in queries:
        result.statements += 1
        for extraction in extract(
            query.sql,
            dialect=query.dialect or dialect,
            system=system or dialect,
            instance=instance,
            default_database=query.default_database,
            default_schema=query.default_schema,
            specs=specs,
        ):
            if extraction.target is None:
                if any("unparseable" in n for n in extraction.notes):
                    result.unparsed += 1
                result.notes.extend(extraction.notes)
                continue
            for note in extraction.notes:
                result.notes.append(f"{extraction.target}: {note}")
            for src in extraction.sources:
                result.graph.add_edge(
                    Edge(
                        src=src,
                        dst=extraction.target,
                        mapping=extraction.mappings.get(
                            src,
                            PartitionMapping.unknown(specs.get(extraction.target, UNPARTITIONED)),
                        ),
                        columns=extraction.column_edges.get(src, ()),
                        evidence=evidence_label
                        or (f"sql:{query.query_id}" if query.query_id else "sql"),
                    )
                )
    return result


def graph_from_lineage(
    events: Iterable[LineageEvent],
    *,
    specs: Mapping[DatasetId, PartitionSpec] | None = None,
    graph: Graph | None = None,
) -> IngestResult:
    """Ingest lineage a platform reported directly.

    Native lineage names datasets and columns but never partition mappings, so every
    edge starts unbounded. A catalog adapter that knows both specs can tighten it
    afterwards with `PartitionMapping.rollup`.
    """
    specs = dict(specs or {})
    result = IngestResult(graph=graph or Graph())
    _seed(result.graph, specs)

    for event in events:
        result.statements += 1
        src_spec = specs.get(event.src, UNPARTITIONED)
        dst_spec = specs.get(event.dst, UNPARTITIONED)
        mapping = (
            PartitionMapping.rollup(src_spec, dst_spec)
            if src_spec.fields and dst_spec.fields
            else PartitionMapping.unknown(dst_spec)
        )
        result.graph.add_edge(
            Edge(
                src=event.src,
                dst=event.dst,
                mapping=mapping,
                columns=event.columns,
                evidence=event.evidence,
            )
        )
    return result


def ingest_engine(
    engine: object,
    *,
    specs: Mapping[DatasetId, PartitionSpec] | None = None,
    since: str | None = None,
    graph: Graph | None = None,
) -> IngestResult:
    """Pull whatever an engine can offer, best source first."""
    dialect = getattr(engine, "dialect", "")
    native = list(getattr(engine, "fetch_lineage", lambda _s: ())(since))
    if native:
        return graph_from_lineage(native, specs=specs, graph=graph)
    queries = list(getattr(engine, "fetch_queries", lambda _s: ())(since))
    return graph_from_queries(queries, dialect=dialect, specs=specs, graph=graph)
