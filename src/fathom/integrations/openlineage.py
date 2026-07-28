"""OpenLineage event ingest.

The highest-leverage integration in the project, because the emitters already exist.
Spark, Flink, Trino, Airflow, dbt, and Dagster all ship OpenLineage producers, so
consuming the format means supporting all of them without writing a listener for
each. It is also why identities follow the OpenLineage naming convention (ADR 3) —
a `RunEvent` maps onto our graph with no translation layer.

What events give us and what they do not:

- **Dataset edges** — always, from `inputs` × `outputs`.
- **Column edges** — from the `columnLineage` facet, which Spark and dbt emit and
  most others do not. Without it the edge is dataset-level, which is still enough
  for invalidation and erasure scoping.
- **Aliases** — from the `symlinks` facet, which is exactly the "external Hive table
  pointing at an S3 prefix" case that ADR 3 said we would need declarations for.
  Producers that emit it save the user a manual alias.
- **Partition mappings** — never. OpenLineage has no partition facet, so mappings
  come from declared specs or stay `UNBOUNDED`.

Runs emit several events (START, RUNNING, COMPLETE). Taking them all would multiply
edges and, worse, record lineage for runs that failed. We keep one event per run:
the terminal one, preferring COMPLETE.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..fs import FileSystem, filesystem_for
from ..graph import Edge, Graph
from ..ids import AliasRegistry
from ..ingest import IngestResult
from ..partitions import PartitionMapping
from ..types import UNPARTITIONED, DatasetId, PartitionSpec

__all__ = [
    "OpenLineageRun",
    "ingest_openlineage",
    "load_events",
    "parse_event",
    "read_events",
]

# Terminal event types, best first. A run that only ever reported RUNNING still
# tells us what it read and wrote, so it is usable as a last resort.
_TERMINAL = ("COMPLETE", "RUNNING", "START")
_FAILED = {"FAIL", "ABORT"}


@dataclass
class OpenLineageRun:
    """One run's worth of lineage, after collapsing its event stream."""

    run_id: str
    job: str = ""
    event_type: str = ""
    event_time: datetime | None = None
    inputs: list[DatasetId] = field(default_factory=list)
    outputs: list[DatasetId] = field(default_factory=list)
    column_edges: dict[tuple[DatasetId, DatasetId], list[tuple[str, str]]] = field(
        default_factory=dict
    )
    symlinks: list[tuple[DatasetId, DatasetId]] = field(default_factory=list)
    producer: str = ""

    @property
    def failed(self) -> bool:
        return self.event_type.upper() in _FAILED


def _dataset(blob: Mapping[str, Any]) -> DatasetId | None:
    namespace = str(blob.get("namespace") or "").strip()
    name = str(blob.get("name") or "").strip()
    if not namespace or not name:
        return None
    # OpenLineage namespaces are already canonical, but producers vary on trailing
    # slashes and on whether the bucket carries the scheme.
    return DatasetId(namespace=namespace.rstrip("/"), name=name.strip("/"))


def _parse_time(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_event(blob: Mapping[str, Any]) -> OpenLineageRun | None:
    """Turn one RunEvent into our shape, or None if it carries no lineage."""
    run = blob.get("run") or {}
    job = blob.get("job") or {}
    run_id = str(run.get("runId") or "")
    if not run_id:
        return None

    parsed = OpenLineageRun(
        run_id=run_id,
        job=f"{job.get('namespace', '')}/{job.get('name', '')}".strip("/"),
        event_type=str(blob.get("eventType") or ""),
        event_time=_parse_time(blob.get("eventTime")),
        producer=str(blob.get("producer") or ""),
    )

    for entry in blob.get("inputs") or []:
        ds = _dataset(entry)
        if ds is not None and ds not in parsed.inputs:
            parsed.inputs.append(ds)

    for entry in blob.get("outputs") or []:
        output = _dataset(entry)
        if output is None:
            continue
        if output not in parsed.outputs:
            parsed.outputs.append(output)

        facets = entry.get("facets") or {}

        # Symlinks name the same bytes under another identity — precisely the alias
        # problem ADR 3 leaves to manual declaration when producers stay silent.
        for link in (facets.get("symlinks") or {}).get("identifiers") or []:
            alias = _dataset(link)
            if alias is not None and alias != output:
                parsed.symlinks.append((alias, output))

        for target, detail in ((facets.get("columnLineage") or {}).get("fields") or {}).items():
            for source in detail.get("inputFields") or []:
                upstream = _dataset(source)
                field_name = source.get("field")
                if upstream is None or not field_name:
                    continue
                parsed.column_edges.setdefault((upstream, output), []).append(
                    (str(field_name), str(target))
                )

    if not parsed.inputs or not parsed.outputs:
        return None
    return parsed


def read_events(raw: str) -> Iterator[Mapping[str, Any]]:
    """Parse a JSON array, a single object, or newline-delimited JSON.

    Producers disagree about which of the three they write, and a file that fails to
    parse should not be diagnosed by the user as "fathom does not support my tool".
    """
    text = raw.strip()
    if not text:
        return
    if text.startswith("["):
        for entry in json.loads(text):
            if isinstance(entry, dict):
                yield entry
        return
    if text.startswith("{") and "\n" not in text.strip().rstrip("}"):
        blob = json.loads(text)
        if isinstance(blob, dict):
            yield blob
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            blob = json.loads(stripped)
        except json.JSONDecodeError:
            continue  # a partially written line in a live event log
        if isinstance(blob, dict):
            yield blob


def load_events(
    uri: str, *, fs: FileSystem | None = None, **storage_options: Any
) -> list[dict[str, Any]]:
    """Read events from a file, a directory of files, or object storage."""
    filesystem = fs or filesystem_for(uri, **storage_options)
    targets = (
        [i.path for i in filesystem.ls(uri) if i.path.endswith((".json", ".jsonl", ".ndjson"))]
        if filesystem.is_dir(uri)
        else [uri]
    )
    out: list[dict[str, Any]] = []
    for path in sorted(targets):
        out.extend(dict(e) for e in read_events(filesystem.read_text(path)))
    return out


def _collapse(events: Iterable[Mapping[str, Any]]) -> list[OpenLineageRun]:
    """One run per runId, keeping the most authoritative event.

    Runs emit START, RUNNING, and COMPLETE for the same work. Ingesting all three
    triples the edges and, worse, records lineage for runs that later failed.
    """
    best: dict[str, OpenLineageRun] = {}
    for blob in events:
        parsed = parse_event(blob)
        if parsed is None:
            continue
        existing = best.get(parsed.run_id)
        if existing is None:
            best[parsed.run_id] = parsed
            continue
        current_rank = (
            _TERMINAL.index(parsed.event_type.upper())
            if parsed.event_type.upper() in _TERMINAL
            else len(_TERMINAL)
        )
        existing_rank = (
            _TERMINAL.index(existing.event_type.upper())
            if existing.event_type.upper() in _TERMINAL
            else len(_TERMINAL)
        )
        if current_rank < existing_rank:
            best[parsed.run_id] = parsed
    return list(best.values())


def ingest_openlineage(
    events: Iterable[Mapping[str, Any]],
    *,
    specs: Mapping[DatasetId, PartitionSpec] | None = None,
    graph: Graph | None = None,
    aliases: AliasRegistry | None = None,
    include_failed: bool = False,
) -> IngestResult:
    """Build a graph from OpenLineage RunEvents.

    Failed runs are skipped by default: a job that died halfway wrote something, but
    what it wrote is not a dependency anyone should plan against.
    """
    specs = dict(specs or {})
    result = IngestResult(graph=graph or Graph())
    for ds, spec in specs.items():
        result.graph.add_dataset(ds, spec)

    runs = _collapse(events)
    skipped_failures = 0

    for run in runs:
        result.statements += 1
        if run.failed and not include_failed:
            skipped_failures += 1
            continue

        if aliases is not None:
            for alias, canonical in run.symlinks:
                aliases.alias(alias, canonical)

        for output in run.outputs:
            dst_spec = specs.get(output, UNPARTITIONED)
            for source in run.inputs:
                if source == output:
                    continue  # self-referencing incremental model
                src_spec = specs.get(source, UNPARTITIONED)
                columns = tuple(run.column_edges.get((source, output), ()))
                mapping = (
                    PartitionMapping.rollup(src_spec, dst_spec)
                    if src_spec.fields and dst_spec.fields
                    else PartitionMapping.unknown(dst_spec)
                )
                result.graph.add_edge(
                    Edge(
                        src=source,
                        dst=output,
                        mapping=mapping,
                        columns=columns,
                        evidence=f"openlineage:{run.job}" if run.job else "openlineage",
                    )
                )
                if not columns:
                    result.notes.append(
                        f"{output}: no columnLineage facet from {run.producer or 'producer'}; "
                        "edge is dataset-level"
                    )

    if skipped_failures:
        result.notes.append(f"skipped {skipped_failures} failed run(s)")
    return result
