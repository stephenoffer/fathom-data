"""Publishing lineage outward, in formats other tools already read.

`fathom.integrations` consumes what other systems emit. This is the other direction:
turning the graph, plans, and profiles into events those systems accept, so fathom
sits inside an existing metadata estate instead of asking to replace it.

That asymmetry is the adoption strategy. A team already running Marquez, DataHub, or
OpenMetadata is not going to swap it out for partition-scoped invalidation; they will
happily accept richer events into what they have. Every format here is a pure
function to a dict — no clients, no credentials, no network. Posting is the caller's
problem, and keeping it that way means this module is testable and safe to run
anywhere.

The OpenLineage output carries a `fathom_partitions` facet with the partition
mapping on it. Consumers that do not understand it ignore it, as the spec requires,
and consumers that do get information no other producer emits.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from ..core.types import DatasetId, KeyPredicate
from ..graph import sinks
from ..graph.model import Edge, Graph, InvalidationPlan
from ..observe.profile import Profile

__all__ = [
    "sink_facet",
    "PRODUCER",
    "dataset_facets",
    "openlineage_complete",
    "openlineage_events",
    "openlineage_start",
    "EmitSummary",
    "partition_facet",
    "partition_payload",
    "plan_events",
    "summarize",
    "to_marquez_namespaces",
    "run_id_for",
    "to_datahub",
    "to_json_lines",
    "to_openmetadata",
    "to_atlas",
]

PRODUCER = "https://github.com/stephenoffer/fathom-data"
SCHEMA_URL = "https://openlineage.io/spec/1-0-5/OpenLineage.json"


def run_id_for(name: str) -> str:
    """A deterministic run id from a job name.

    Deterministic rather than random so re-emitting the same graph is idempotent in
    the consumer. A catalog that gains a new run every time a sync job runs is a
    catalog nobody can read.
    """
    return str(uuid5(NAMESPACE_URL, f"{PRODUCER}/{name}"))


def _dataset(ds: DatasetId) -> dict[str, Any]:
    return {"namespace": ds.namespace, "name": ds.name}


def partition_facet(edge: Edge) -> dict[str, Any]:
    """The partition mapping on an edge, as a custom OpenLineage facet.

    The one thing this project knows that other producers do not: which output
    partitions an input partition dirties. Emitted so a downstream consumer can use it
    even if only to display it.
    """
    return {
        "_producer": PRODUCER,
        "_schemaURL": f"{PRODUCER}#partitionMapping",
        "mapping": {name: str(fm) for name, fm in edge.mapping.fields},
        "unbounded": edge.mapping.is_unbounded,
        "evidence": edge.evidence,
    }


def sink_facet(ds: DatasetId) -> dict[str, Any]:
    """A facet marking a dataset as a published artefact rather than a table.

    Emitted so the distinction survives export. A catalog that receives a filing and a
    staging table as two identical dataset nodes has lost the one property that makes
    the filing worth tracking, and no consumer of the export can recover it from the
    name.
    """
    kind = sinks.kind_of(ds)
    if kind is None:
        return {}
    return {
        "fathom_publishedArtefact": {
            "_producer": PRODUCER,
            "_schemaURL": f"{PRODUCER}#publishedArtefact",
            "kind": kind.value,
            "regulatory": kind in sinks.REGULATORY,
            "terminal": True,
        }
    }


def dataset_facets(graph: Graph, ds: DatasetId) -> dict[str, Any]:
    """Standard facets for one dataset: its partition spec, and whether it is a sink."""
    spec = graph.spec(ds)
    if not spec.fields:
        return sink_facet(ds)
    return {
        **sink_facet(ds),
        "fathom_partitionSpec": {
            "_producer": PRODUCER,
            "_schemaURL": f"{PRODUCER}#partitionSpec",
            "fields": [
                {"name": f.name, "kind": f.kind, "grain": f.grain.label if f.grain else None}
                for f in spec.fields
            ],
        },
    }


def _event(
    event_type: str,
    job_name: str,
    *,
    inputs: Sequence[dict[str, Any]],
    outputs: Sequence[dict[str, Any]],
    occurred: datetime | None = None,
    namespace: str = "fathom",
) -> dict[str, Any]:
    return {
        "eventType": event_type,
        "eventTime": (occurred or datetime.now(UTC)).isoformat(),
        "producer": PRODUCER,
        "schemaURL": SCHEMA_URL,
        "run": {"runId": run_id_for(job_name)},
        "job": {"namespace": namespace, "name": job_name},
        "inputs": list(inputs),
        "outputs": list(outputs),
    }


def openlineage_start(
    graph: Graph, target: DatasetId, *, namespace: str = "fathom", occurred: datetime | None = None
) -> dict[str, Any]:
    """A START event for the job that builds one dataset."""
    inputs = [
        {**_dataset(edge.src), "facets": {**dataset_facets(graph, edge.src)}}
        for edge in graph.in_edges(target)
    ]
    outputs = [
        {
            **_dataset(target),
            "facets": dataset_facets(graph, target),
            "outputFacets": {},
        }
    ]
    return _event(
        "START",
        str(target),
        inputs=inputs,
        outputs=outputs,
        occurred=occurred,
        namespace=namespace,
    )


def openlineage_complete(
    graph: Graph,
    target: DatasetId,
    *,
    namespace: str = "fathom",
    occurred: datetime | None = None,
    profile: Profile | None = None,
) -> dict[str, Any]:
    """A COMPLETE event, carrying column lineage and partition facets.

    Column lineage uses the standard `columnLineage` facet, so consumers that already
    render it need no changes. The partition facet rides alongside it.
    """
    column_fields: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = []
    for edge in graph.in_edges(target):
        inputs.append(
            {
                **_dataset(edge.src),
                "facets": {
                    **dataset_facets(graph, edge.src),
                    "fathom_partitionMapping": partition_facet(edge),
                },
            }
        )
        for source_column, target_column in edge.columns:
            entry = column_fields.setdefault(target_column, {"inputFields": []})
            entry["inputFields"].append({**_dataset(edge.src), "field": source_column})

    output_facets: dict[str, Any] = dict(dataset_facets(graph, target))
    if column_fields:
        output_facets["columnLineage"] = {
            "_producer": PRODUCER,
            "_schemaURL": f"{SCHEMA_URL}#/definitions/ColumnLineageDatasetFacet",
            "fields": column_fields,
        }
    if profile is not None:
        output_facets["dataQualityMetrics"] = {
            "_producer": PRODUCER,
            "_schemaURL": f"{SCHEMA_URL}#/definitions/DataQualityMetricsInputDatasetFacet",
            "rowCount": profile.row_count,
            "fileCount": profile.file_count,
            "columnMetrics": {
                column.name: {
                    "nullCount": column.null_count,
                    "distinctCount": column.distinct_estimate,
                    "min": None if column.min is None else str(column.min),
                    "max": None if column.max is None else str(column.max),
                }
                for column in profile.columns
            },
        }

    outputs = [{**_dataset(target), "facets": output_facets}]
    return _event(
        "COMPLETE",
        str(target),
        inputs=inputs,
        outputs=outputs,
        occurred=occurred,
        namespace=namespace,
    )


def openlineage_events(
    graph: Graph,
    *,
    datasets: Iterable[DatasetId] | None = None,
    namespace: str = "fathom",
    profiles: Mapping[DatasetId, Profile] | None = None,
) -> list[dict[str, Any]]:
    """START and COMPLETE events for every dataset that has inputs.

    Sources emit nothing: a dataset nothing produces has no job to report, and
    inventing one pollutes the consumer's job list with entries that never run.
    """
    targets = list(datasets) if datasets is not None else graph.datasets
    known = dict(profiles or {})
    out: list[dict[str, Any]] = []
    for ds in targets:
        if not graph.in_edges(ds):
            continue
        out.append(openlineage_start(graph, ds, namespace=namespace))
        out.append(openlineage_complete(graph, ds, namespace=namespace, profile=known.get(ds)))
    return out


def plan_events(
    graph: Graph, plan: InvalidationPlan, *, namespace: str = "fathom"
) -> list[dict[str, Any]]:
    """One event per dataset in a plan, carrying the partitions to be rebuilt.

    Turns a plan into something an orchestrator's metadata layer can display next to
    the runs it triggers, rather than a list of strings in a log.
    """
    out: list[dict[str, Any]] = []
    for ds in plan.order:
        event = openlineage_start(graph, ds, namespace=namespace)
        for output in event["outputs"]:
            output.setdefault("facets", {})["fathom_plan"] = {
                "_producer": PRODUCER,
                "_schemaURL": f"{PRODUCER}#plan",
                "partitions": sorted(str(k) for k in plan.partitions(ds)),
                "widened": ds in plan.widened,
                "reasons": plan.reasons.get(ds, []),
            }
        out.append(event)
    return out


def to_json_lines(events: Sequence[Mapping[str, Any]]) -> str:
    """Newline-delimited JSON, the transport every OpenLineage consumer accepts."""
    return "\n".join(json.dumps(event, sort_keys=True, separators=(",", ":")) for event in events)


# -- other catalogs ------------------------------------------------------------


def _urn(ds: DatasetId, platform: str) -> str:
    scheme = ds.namespace.split("://", 1)[0]
    return f"urn:li:dataset:(urn:li:dataPlatform:{platform or scheme},{ds.name},PROD)"


def to_datahub(graph: Graph, *, platform: str = "") -> list[dict[str, Any]]:
    """DataHub `UpstreamLineage` aspects, one per dataset with inputs.

    Shaped for the metadata change proposal format DataHub's REST sink accepts.
    """
    out: list[dict[str, Any]] = []
    for ds in graph.datasets:
        edges = graph.in_edges(ds)
        if not edges:
            continue
        out.append(
            {
                "entityType": "dataset",
                "entityUrn": _urn(ds, platform),
                "aspectName": "upstreamLineage",
                "aspect": {
                    "upstreams": [
                        {
                            "dataset": _urn(edge.src, platform),
                            "type": "TRANSFORMED",
                            "auditStamp": {"actor": "urn:li:corpuser:fathom", "time": 0},
                        }
                        for edge in edges
                    ],
                    "fineGrainedLineages": [
                        {
                            "upstreamType": "FIELD_SET",
                            "downstreamType": "FIELD",
                            "upstreams": [f"{_urn(edge.src, platform)},{source}"],
                            "downstreams": [f"{_urn(ds, platform)},{target}"],
                        }
                        for edge in edges
                        for source, target in edge.columns
                    ],
                },
            }
        )
    return out


def to_openmetadata(graph: Graph) -> list[dict[str, Any]]:
    """OpenMetadata `AddLineage` requests, one per edge."""
    return [
        {
            "edge": {
                "fromEntity": {"id": str(edge.src), "type": "table"},
                "toEntity": {"id": str(edge.dst), "type": "table"},
                "lineageDetails": {
                    "sqlQuery": "",
                    "columnsLineage": [
                        {"fromColumns": [source], "toColumn": target}
                        for source, target in edge.columns
                    ],
                    "description": f"{edge.mapping} [{edge.evidence}]",
                },
            }
        }
        for edge in graph.edges
    ]


def to_atlas(graph: Graph) -> dict[str, Any]:
    """An Apache Atlas entity bulk payload for the graph.

    Datasets become `DataSet` entities and edges become `Process` entities, which is
    Atlas's model: lineage is a node, not an edge.
    """
    entities: list[dict[str, Any]] = []
    for ds in graph.datasets:
        entities.append(
            {
                "typeName": "DataSet",
                "attributes": {
                    "qualifiedName": str(ds),
                    "name": ds.name,
                    "description": f"namespace {ds.namespace}",
                },
            }
        )
    for index, edge in enumerate(graph.edges):
        entities.append(
            {
                "typeName": "Process",
                "attributes": {
                    "qualifiedName": f"{edge.src}->{edge.dst}#{index}",
                    "name": f"{edge.src.name} to {edge.dst.name}",
                    "inputs": [
                        {
                            "typeName": "DataSet",
                            "uniqueAttributes": {"qualifiedName": str(edge.src)},
                        }
                    ],
                    "outputs": [
                        {
                            "typeName": "DataSet",
                            "uniqueAttributes": {"qualifiedName": str(edge.dst)},
                        }
                    ],
                    "description": f"{edge.mapping} [{edge.evidence}]",
                },
            }
        )
    return {"entities": entities}


def to_marquez_namespaces(graph: Graph) -> list[dict[str, str]]:
    """Namespace registration payloads, which Marquez wants before any event lands."""
    return [
        {"name": namespace, "ownerName": "fathom"}
        for namespace in sorted({ds.namespace for ds in graph.datasets})
    ]


@dataclass(frozen=True)
class EmitSummary:
    """What an emit run produced, for logging without dumping the payloads."""

    events: int = 0
    datasets: int = 0
    edges: int = 0
    with_column_lineage: int = 0

    def __str__(self) -> str:
        return (
            f"emitted {self.events} event(s) covering {self.datasets} dataset(s) and "
            f"{self.edges} edge(s), {self.with_column_lineage} with column lineage"
        )


def summarize(graph: Graph, events: Sequence[Mapping[str, Any]]) -> EmitSummary:
    """Describe an emit run without printing every payload."""
    return EmitSummary(
        events=len(events),
        datasets=len(graph.datasets),
        edges=len(graph.edges),
        with_column_lineage=sum(1 for edge in graph.edges if edge.columns),
    )


def partition_payload(keys: Iterable[KeyPredicate]) -> list[dict[str, str]]:
    """Partition predicates as plain dicts, for embedding in any of the above."""
    return [{name: str(value) for name, value in key.bindings} for key in keys]
