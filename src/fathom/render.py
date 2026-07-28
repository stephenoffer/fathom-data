"""Rendering the artifacts into things people and other tools read.

A lineage graph nobody can see is a lineage graph nobody trusts. This module turns
the graph, plans, profiles, and proofs into Mermaid, Graphviz, JSON, and Markdown —
formats that paste into a pull request, a runbook, or a wiki without a server.

Two rules the whole module follows:

- **Every renderer is a pure function returning a string.** Nothing writes files
  except the explicit `write_*` helpers, so output is trivially testable and safe to
  embed in a report an agent is composing.
- **Identifiers are escaped, never trusted.** Dataset names contain dots, slashes,
  quotes, and occasionally a stray bracket from a quoted identifier; a renderer that
  interpolates them raw produces a diagram that silently fails to parse.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .codec import (
    dataset_from_json,
    dataset_to_json,
    key_to_json,
    mapping_from_json,
    mapping_to_json,
    spec_from_json,
    spec_to_json,
)
from .erasure import ErasurePlan, ErasureProof
from .graph import Edge, Graph, InvalidationPlan
from .policy import LabelSet, Violation
from .profile import Finding, Profile
from .shadow import ShadowReport
from .types import DatasetId, KeyPredicate

__all__ = [
    "erasure_plan_to_markdown",
    "findings_to_markdown",
    "graph_from_json",
    "graph_to_cytoscape",
    "graph_to_d2",
    "graph_to_dot",
    "graph_to_json",
    "graph_to_markdown",
    "graph_to_mermaid",
    "graph_to_plantuml",
    "labels_to_markdown",
    "partition_table",
    "plan_to_json",
    "plan_to_markdown",
    "plan_to_mermaid",
    "plan_to_text",
    "profile_to_json",
    "profile_to_markdown",
    "proof_to_markdown",
    "shadow_to_markdown",
    "tree",
    "violations_to_markdown",
    "write_graph",
    "write_json",
    "write_text",
]

_UNSAFE = re.compile(r"[^0-9A-Za-z_]")


def _node_id(ds: DatasetId) -> str:
    """A diagram-safe identifier. Stable across runs so diffs stay readable."""
    return "n_" + _UNSAFE.sub("_", str(ds))


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _label(ds: DatasetId, *, short: bool) -> str:
    return ds.name.rsplit("/", 1)[-1] if short else str(ds)


# -- graph ---------------------------------------------------------------------


def graph_to_mermaid(
    graph: Graph,
    *,
    highlight: Iterable[DatasetId] = (),
    show_mappings: bool = True,
    short_names: bool = False,
    direction: str = "LR",
) -> str:
    """A Mermaid flowchart. Renders inline in GitHub, GitLab, and Notion.

    `highlight` styles a subset — pass a plan's dirty datasets and the diagram
    becomes an explanation of that plan rather than a picture of the warehouse.
    """
    marked = set(highlight)
    lines = [f"flowchart {direction}"]
    for ds in graph.datasets:
        shape = f'{_node_id(ds)}["{_label(ds, short=short_names)}"]'
        lines.append(f"    {shape}")
    for edge in graph.edges:
        text = f"|{edge.mapping}|" if show_mappings and edge.mapping.fields else ""
        lines.append(f"    {_node_id(edge.src)} -->{text} {_node_id(edge.dst)}")
    if marked:
        lines.append("    classDef dirty fill:#fde68a,stroke:#b45309,color:#111;")
        for ds in sorted(marked, key=str):
            if ds in set(graph.datasets):
                lines.append(f"    class {_node_id(ds)} dirty;")
    return "\n".join(lines)


def graph_to_dot(graph: Graph, *, short_names: bool = False, rankdir: str = "LR") -> str:
    """Graphviz DOT, for when the graph is too large for Mermaid to lay out."""
    lines = ["digraph lineage {", f"  rankdir={rankdir};", "  node [shape=box, fontsize=10];"]
    for ds in graph.datasets:
        lines.append(f"  {_node_id(ds)} [label={_quote(_label(ds, short=short_names))}];")
    for edge in graph.edges:
        attrs = f" [label={_quote(str(edge.mapping))}]" if edge.mapping.fields else ""
        lines.append(f"  {_node_id(edge.src)} -> {_node_id(edge.dst)}{attrs};")
    lines.append("}")
    return "\n".join(lines)


def graph_to_d2(graph: Graph, *, short_names: bool = False) -> str:
    """D2 source, which lays out large graphs better than DOT does."""
    lines: list[str] = []
    for ds in graph.datasets:
        lines.append(f"{_node_id(ds)}: {_quote(_label(ds, short=short_names))}")
    for edge in graph.edges:
        suffix = f": {_quote(str(edge.mapping))}" if edge.mapping.fields else ""
        lines.append(f"{_node_id(edge.src)} -> {_node_id(edge.dst)}{suffix}")
    return "\n".join(lines)


def graph_to_plantuml(graph: Graph, *, short_names: bool = False) -> str:
    """PlantUML component source, for teams whose docs pipeline already renders it."""
    lines = ["@startuml", "left to right direction"]
    for ds in graph.datasets:
        lines.append(f"component {_quote(_label(ds, short=short_names))} as {_node_id(ds)}")
    for edge in graph.edges:
        lines.append(f"{_node_id(edge.src)} --> {_node_id(edge.dst)}")
    lines.append("@enduml")
    return "\n".join(lines)


def graph_to_cytoscape(graph: Graph) -> dict[str, Any]:
    """Cytoscape.js elements, the format most web lineage viewers already accept."""
    nodes = [
        {"data": {"id": _node_id(ds), "label": str(ds), "namespace": ds.namespace}}
        for ds in graph.datasets
    ]
    edges = [
        {
            "data": {
                "id": f"{_node_id(e.src)}__{_node_id(e.dst)}__{_UNSAFE.sub('_', e.evidence)}",
                "source": _node_id(e.src),
                "target": _node_id(e.dst),
                "label": str(e.mapping),
                "evidence": e.evidence,
            }
        }
        for e in graph.edges
    ]
    return {"elements": {"nodes": nodes, "edges": edges}}


def graph_to_json(graph: Graph, *, indent: int | None = 2) -> str:
    """Round-trippable JSON. Partition values keep their types, so keys still compare equal."""
    body = {
        "version": 1,
        "datasets": [
            {
                "id": json.loads(dataset_to_json(ds)),
                "spec": json.loads(spec_to_json(graph.spec(ds))),
            }
            for ds in graph.datasets
        ],
        "edges": [
            {
                "src": json.loads(dataset_to_json(e.src)),
                "dst": json.loads(dataset_to_json(e.dst)),
                "mapping": json.loads(mapping_to_json(e.mapping)),
                "columns": [list(pair) for pair in e.columns],
                "evidence": e.evidence,
            }
            for e in graph.edges
        ],
    }
    return json.dumps(body, indent=indent, sort_keys=True)


def graph_from_json(raw: str) -> Graph:
    """Rebuild a graph written by `graph_to_json`."""
    blob = json.loads(raw)
    graph = Graph()
    for entry in blob.get("datasets", ()):
        graph.add_dataset(
            dataset_from_json(json.dumps(entry["id"])),
            spec_from_json(json.dumps(entry["spec"])),
        )
    for entry in blob.get("edges", ()):
        graph.add_edge(
            Edge(
                src=dataset_from_json(json.dumps(entry["src"])),
                dst=dataset_from_json(json.dumps(entry["dst"])),
                mapping=mapping_from_json(json.dumps(entry["mapping"])),
                columns=tuple((str(a), str(b)) for a, b in entry.get("columns", ())),
                evidence=entry.get("evidence", "declared"),
            )
        )
    return graph


def graph_to_markdown(graph: Graph, *, limit: int = 200) -> str:
    """A table of edges, for a docs page or a PR body."""
    lines = [
        f"### Lineage — {len(graph.datasets)} dataset(s), {len(graph.edges)} edge(s)",
        "",
        "| Source | Target | Mapping | Evidence |",
        "|---|---|---|---|",
    ]
    for edge in sorted(graph.edges, key=lambda e: (str(e.src), str(e.dst)))[:limit]:
        lines.append(f"| `{edge.src}` | `{edge.dst}` | `{edge.mapping}` | {edge.evidence} |")
    if len(graph.edges) > limit:
        lines.append(f"| … | | | _{len(graph.edges) - limit} more_ |")
    return "\n".join(lines)


def tree(graph: Graph, root: DatasetId, *, downstream: bool = True, max_depth: int = 8) -> str:
    """An indented text tree from one dataset. What a terminal wants.

    Repeated subtrees are marked rather than expanded, so a diamond does not print
    twice and a cycle terminates.
    """
    from .query import children, parents

    lines: list[str] = []
    seen: set[DatasetId] = set()

    def walk(node: DatasetId, prefix: str, depth: int, last: bool) -> None:
        connector = "" if depth == 0 else ("└── " if last else "├── ")
        marker = " (↩)" if node in seen and depth > 0 else ""
        lines.append(f"{prefix}{connector}{node}{marker}")
        if node in seen or depth >= max_depth:
            return
        seen.add(node)
        kids = children(graph, node) if downstream else parents(graph, node)
        child_prefix = prefix + ("" if depth == 0 else ("    " if last else "│   "))
        for index, kid in enumerate(kids):
            walk(kid, child_prefix, depth + 1, index == len(kids) - 1)

    walk(root, "", 0, True)
    return "\n".join(lines)


# -- plans ---------------------------------------------------------------------


def plan_to_text(plan: InvalidationPlan) -> str:
    """One line per dataset and partition. The default CLI rendering."""
    lines: list[str] = []
    for ds in plan.order:
        for key in sorted(plan.dirty.get(ds, frozenset()), key=str):
            lines.append(f"{ds}\t{key}")
    return "\n".join(lines)


def plan_to_markdown(plan: InvalidationPlan, *, limit: int = 8) -> str:
    """A rebuild plan as a table, with the widening reasons kept visible.

    The `widened` column is the one that matters in review: it says where the
    planner gave up on precision, which is where a missing partition spec usually is.
    """
    if plan.is_empty:
        return "**Rebuild plan:** nothing to rebuild."
    lines = [
        f"**Rebuild plan** — {len(plan.dirty)} dataset(s), in order",
        "",
        "| # | Dataset | Partitions | Widened | Why |",
        "|---|---|---|---|---|",
    ]
    for index, ds in enumerate(plan.order, start=1):
        keys = sorted(str(k) for k in plan.dirty.get(ds, frozenset()))
        shown = ", ".join(f"`{k}`" for k in keys[:limit])
        if len(keys) > limit:
            shown += f", _+{len(keys) - limit} more_"
        widened = "yes" if ds in plan.widened else ""
        why = "; ".join(plan.reasons.get(ds, [])[:2])
        lines.append(f"| {index} | `{ds}` | {shown} | {widened} | {why} |")
    return "\n".join(lines)


def plan_to_mermaid(graph: Graph, plan: InvalidationPlan, **kwargs: Any) -> str:
    """The graph with the plan's dirty datasets highlighted."""
    return graph_to_mermaid(graph, highlight=plan.dirty.keys(), **kwargs)


def plan_to_json(plan: InvalidationPlan, *, indent: int | None = 2) -> str:
    """Machine-readable plan, for an orchestrator to consume as a task list."""
    body = {
        "order": [str(ds) for ds in plan.order],
        "datasets": {
            str(ds): {
                "partitions": [json.loads(key_to_json(k)) for k in sorted(keys, key=str)],
                "rendered": sorted(str(k) for k in keys),
                "widened": ds in plan.widened,
                "cyclic": ds in plan.cyclic,
                "reasons": plan.reasons.get(ds, []),
            }
            for ds, keys in sorted(plan.dirty.items(), key=lambda kv: str(kv[0]))
        },
    }
    return json.dumps(body, indent=indent, sort_keys=True)


# -- profiles and findings -----------------------------------------------------


def profile_to_markdown(profile: Profile, *, limit: int = 50) -> str:
    """One profile as a column table."""
    lines = [
        f"**Profile** `{profile.dataset}` `{profile.partition}` — "
        f"{profile.row_count:,} rows across {profile.file_count} file(s), "
        f"source `{profile.source}`",
        "",
        "| Column | Type | Nulls | Min | Max | Bytes |",
        "|---|---|---|---|---|---|",
    ]
    for column in profile.columns[:limit]:
        rate = "—" if column.null_rate is None else f"{column.null_rate:.1%}"
        lines.append(
            f"| `{column.name}` | {column.dtype} | {rate} | "
            f"{column.min if column.min is not None else '—'} | "
            f"{column.max if column.max is not None else '—'} | "
            f"{column.byte_size or 0:,} |"
        )
    return "\n".join(lines)


def profile_to_json(profile: Profile, *, indent: int | None = 2) -> str:
    """A profile as JSON, suitable for storing outside the SQLite store."""
    return json.dumps(
        {
            "dataset": str(profile.dataset),
            "partition": str(profile.partition),
            "row_count": profile.row_count,
            "file_count": profile.file_count,
            "source": profile.source,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "row_count": c.row_count,
                    "null_count": c.null_count,
                    "null_rate": c.null_rate,
                    "min": None if c.min is None else str(c.min),
                    "max": None if c.max is None else str(c.max),
                    "distinct_estimate": c.distinct_estimate,
                    "byte_size": c.byte_size,
                }
                for c in profile.columns
            ],
        },
        indent=indent,
        sort_keys=True,
    )


def findings_to_markdown(findings: Sequence[Finding], *, title: str = "Drift") -> str:
    """Drift findings as a severity-ordered table."""
    if not findings:
        return f"**{title}:** none detected."
    order = {"error": 0, "warn": 1, "info": 2}
    lines = [
        f"**{title}** — {len(findings)} finding(s)",
        "",
        "| Severity | Column | What | Before → After |",
        "|---|---|---|---|",
    ]
    for f in sorted(findings, key=lambda f: order.get(f.severity.value, 9)):
        column = f"`{f.column}`" if f.column else "—"
        lines.append(
            f"| {f.severity.value} | {column} | {f.detail} | "
            f"{'—' if f.before is None else f.before} → {'—' if f.after is None else f.after} |"
        )
    return "\n".join(lines)


# -- policy --------------------------------------------------------------------


def labels_to_markdown(labels: LabelSet, *, min_confidence: float = 0.0) -> str:
    """Inferred and propagated labels, grouped by dataset."""
    lines = [
        "| Dataset | Column | Label | Confidence | Origin | Confirmed |",
        "|---|---|---|---|---|---|",
    ]
    rows = 0
    for ref in sorted(labels, key=lambda r: (str(r.dataset), r.column)):
        for label in sorted(labels[ref]):
            if label.confidence < min_confidence and not label.confirmed:
                continue
            rows += 1
            lines.append(
                f"| `{ref.dataset}` | `{ref.column}` | {label.name} | "
                f"{label.confidence:.0%} | {label.origin} | {'yes' if label.confirmed else ''} |"
            )
    if rows == 0:
        return "**Labels:** none above the confidence floor."
    return "\n".join([f"**Labels** — {rows} claim(s)", "", *lines])


def violations_to_markdown(violations: Sequence[Violation]) -> str:
    """Policy violations, phrased so the fix is obvious from the row."""
    if not violations:
        return "**Policy:** no violations."
    lines = [
        f"**Policy** — {len(violations)} violation(s)",
        "",
        "| Dataset | Column | Rule | Label | Confidence | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for v in violations:
        column = "_(unknown)_" if v.is_unattributed else f"`{v.column}`"
        lines.append(
            f"| `{v.dataset}` | {column} | {v.rule} | {v.label} | {v.confidence:.0%} | {v.reason} |"
        )
    return "\n".join(lines)


# -- erasure -------------------------------------------------------------------


def erasure_plan_to_markdown(plan: ErasurePlan) -> str:
    """An erasure plan, with incompleteness stated first rather than buried."""
    header = (
        f"**Erasure plan** — subject `{plan.request.subject_digest()[:12]}…` "
        f"on `{plan.request.key_column}`, {len(plan.targets)} dataset(s)"
    )
    lines = [header, ""]
    if not plan.is_complete:
        lines.append("> **INCOMPLETE.** Some copies cannot be destroyed by this tool. ")
        lines.append("> Do not report the request as fulfilled.")
        lines.append("")
    lines.extend(["| Dataset | Mode | Partitions | Widened | Blocked |", "|---|---|---|---|---|"])
    for target in plan.targets:
        keys = ", ".join(f"`{k}`" for k in sorted(str(k) for k in target.partitions)[:4]) or "whole"
        lines.append(
            f"| `{target.dataset}` | {target.mode.value} | {keys} | "
            f"{'yes' if target.widened else ''} | {target.blocked or ''} |"
        )
    return "\n".join(lines)


def proof_to_markdown(proof: ErasureProof) -> str:
    """The auditor-facing rendering of a proof artifact."""
    lines = [
        "**Erasure proof**",
        "",
        f"- subject digest: `{proof.subject_digest}`",
        f"- reference: `{proof.reference or '—'}`",
        f"- generated: {proof.generated.isoformat()}",
        f"- executed: {proof.executed}",
        f"- complete: {proof.complete}",
        f"- digest: `{proof.digest}`",
        "",
        "| Dataset | Status | Mode | Rows deleted |",
        "|---|---|---|---|",
    ]
    for entry in proof.entries:
        lines.append(
            f"| `{entry.get('dataset')}` | {entry.get('status')} | "
            f"{entry.get('mode')} | {entry.get('rows_deleted', '—')} |"
        )
    return "\n".join(lines)


# -- shadow --------------------------------------------------------------------


def shadow_to_markdown(report: ShadowReport) -> str:
    """Shadow results, leading with soundness because nothing else matters first."""
    verdict = "SOUND" if report.is_sound else f"UNSOUND — {report.missed_total} missed"
    lines = [
        f"**Shadow run** — {verdict}, {report.savings:.0%} of partitions skipped",
        "",
        "| Dataset | Sound | Planned | Actual | Total | Savings | Precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        lines.append(
            f"| `{r.dataset}` | {'yes' if r.is_sound else '**NO**'} | {len(r.planned)} | "
            f"{len(r.actual)} | {r.total} | {r.savings:.0%} | {r.precision:.0%} |"
        )
    return "\n".join(lines)


# -- writing -------------------------------------------------------------------


def write_text(path: str | Path, content: str) -> Path:
    """Write a rendering to disk, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def write_json(path: str | Path, body: Mapping[str, Any] | Sequence[Any]) -> Path:
    """Write JSON with stable key ordering, so committed artifacts diff cleanly."""
    return write_text(path, json.dumps(body, indent=2, sort_keys=True))


def write_graph(path: str | Path, graph: Graph, *, format: str = "json") -> Path:
    """Write a graph in any supported format, chosen by name."""
    if format == "json":
        return write_text(path, graph_to_json(graph))
    if format == "mermaid":
        return write_text(path, graph_to_mermaid(graph))
    if format == "dot":
        return write_text(path, graph_to_dot(graph))
    if format == "d2":
        return write_text(path, graph_to_d2(graph))
    if format == "plantuml":
        return write_text(path, graph_to_plantuml(graph))
    if format == "markdown":
        return write_text(path, graph_to_markdown(graph))
    if format == "cytoscape":
        return write_json(path, graph_to_cytoscape(graph))
    raise ValueError(
        f"unknown format {format!r}; try one of "
        "['cytoscape', 'd2', 'dot', 'json', 'markdown', 'mermaid', 'plantuml']"
    )


def partition_table(keys: Iterable[KeyPredicate]) -> str:
    """Render a set of partition predicates as a sorted bullet list."""
    items = sorted(str(k) for k in keys)
    return "\n".join(f"- `{k}`" for k in items) if items else "_(none)_"
