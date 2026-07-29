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

from ..core.codec import (
    dataset_from_json,
    dataset_to_json,
    key_to_json,
    mapping_from_json,
    mapping_to_json,
    spec_from_json,
    spec_to_json,
)
from ..core.types import DatasetId, KeyPredicate
from ..core.util import markdown as md
from ..core.util import text
from ..govern.contracts import ContractReport
from ..govern.erasure import ErasurePlan, ErasureProof
from ..govern.policy import LabelSet, Violation
from ..govern.reidentification import RiskReport
from ..graph.history import History
from ..graph.model import Edge, Graph, InvalidationPlan
from ..graph.plan.lifetime import LifetimeCost, ValueFinding
from ..graph.sinks import RestatementImpact
from ..graph.sinks import describe as describe_sink
from ..observe.completeness import CompletenessReport
from ..observe.profile import Finding, Profile
from ..observe.seasonal import SeasonalBaseline
from ..observe.shadow import ShadowReport
from ..observe.usage import RetirementCandidate, UsageStats

__all__ = [
    "completeness_to_markdown",
    "contract_report_to_markdown",
    "erasure_plan_to_markdown",
    "history_to_markdown",
    "lifetime_to_markdown",
    "restatement_to_markdown",
    "retirement_to_markdown",
    "risk_to_markdown",
    "seasonal_to_markdown",
    "usage_to_markdown",
    "value_to_markdown",
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
    rows = [
        [md.code(edge.src), md.code(edge.dst), md.code(edge.mapping), edge.evidence]
        for edge in sorted(graph.edges, key=lambda e: (str(e.src), str(e.dst)))
    ]
    return "\n".join(
        [
            f"### Lineage — {len(graph.datasets)} dataset(s), {len(graph.edges)} edge(s)",
            "",
            md.table(["Source", "Target", "Mapping", "Evidence"], rows, limit=limit),
        ]
    )


def tree(graph: Graph, root: DatasetId, *, downstream: bool = True, max_depth: int = 8) -> str:
    """An indented text tree from one dataset. What a terminal wants.

    Repeated subtrees are marked rather than expanded, so a diamond does not print
    twice and a cycle terminates.
    """
    from ..graph.query import children, parents

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
    """One line per dataset and partition. The default CLI rendering.

    An empty plan renders as a statement rather than as an empty string. "Nothing to
    rebuild" and "the renderer produced nothing" look identical on a terminal, and
    they are very different answers to give someone who just asked what a change
    would cost.
    """
    lines: list[str] = []
    for ds in plan.order:
        for key in sorted(plan.dirty.get(ds, frozenset()), key=str):
            lines.append(f"{ds}\t{key}")
    return "\n".join(lines) if lines else "nothing to rebuild"


def plan_to_markdown(plan: InvalidationPlan, *, limit: int = 8) -> str:
    """A rebuild plan as a table, with the widening reasons kept visible.

    The `widened` column is the one that matters in review: it says where the
    planner gave up on precision, which is where a missing partition spec usually is.
    """
    if plan.is_empty:
        return "**Rebuild plan:** nothing to rebuild."
    rows = []
    for index, ds in enumerate(plan.order, start=1):
        keys = sorted(str(k) for k in plan.dirty.get(ds, frozenset()))
        shown = ", ".join(f"`{k}`" for k in keys[:limit])
        if len(keys) > limit:
            shown += f", _+{len(keys) - limit} more_"
        rows.append(
            [
                index,
                md.code(ds),
                shown,
                "yes" if ds in plan.widened else "",
                "; ".join(plan.reasons.get(ds, [])[:2]),
            ]
        )
    return "\n".join(
        [
            f"**Rebuild plan** — {len(plan.dirty)} dataset(s), in order",
            "",
            md.table(["#", "Dataset", "Partitions", "Widened", "Why"], rows),
        ]
    )


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
    rows = [
        [
            md.code(column.name),
            column.dtype,
            None if column.null_rate is None else f"{column.null_rate:.1%}",
            column.min,
            column.max,
            f"{column.byte_size or 0:,}",
        ]
        for column in profile.columns
    ]
    return "\n".join(
        [
            f"**Profile** `{profile.dataset}` `{profile.partition}` — "
            f"{profile.row_count:,} rows across {profile.file_count} file(s), "
            f"source `{profile.source}`",
            "",
            md.table(["Column", "Type", "Nulls", "Min", "Max", "Bytes"], rows, limit=limit),
        ]
    )


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
    rows = [
        [
            f.severity.value,
            md.code(f.column),
            f.detail,
            f"{md.cell(f.before)} → {md.cell(f.after)}",
        ]
        for f in sorted(findings, key=lambda f: order.get(f.severity.value, 9))
    ]
    return "\n".join(
        [
            f"**{title}** — {len(findings)} finding(s)",
            "",
            md.table(["Severity", "Column", "What", "Before → After"], rows),
        ]
    )


# -- policy --------------------------------------------------------------------


def labels_to_markdown(labels: LabelSet, *, min_confidence: float = 0.0) -> str:
    """Inferred and propagated labels, grouped by dataset."""
    rows = [
        [
            md.code(ref.dataset),
            md.code(ref.column),
            label.name,
            f"{label.confidence:.0%}",
            label.origin,
            "yes" if label.confirmed else "",
        ]
        for ref in sorted(labels, key=lambda r: (str(r.dataset), r.column))
        for label in sorted(labels[ref])
        if label.confirmed or label.confidence >= min_confidence
    ]
    if not rows:
        return "**Labels:** none above the confidence floor."
    return "\n".join(
        [
            f"**Labels** — {len(rows)} claim(s)",
            "",
            md.table(["Dataset", "Column", "Label", "Confidence", "Origin", "Confirmed"], rows),
        ]
    )


def violations_to_markdown(violations: Sequence[Violation]) -> str:
    """Policy violations, phrased so the fix is obvious from the row."""
    if not violations:
        return "**Policy:** no violations."
    rows = [
        [
            md.code(v.dataset),
            "_(unknown)_" if v.is_unattributed else md.code(v.column),
            v.rule,
            v.label,
            f"{v.confidence:.0%}",
            v.reason,
        ]
        for v in violations
    ]
    return "\n".join(
        [
            f"**Policy** — {len(violations)} violation(s)",
            "",
            md.table(["Dataset", "Column", "Rule", "Label", "Confidence", "Reason"], rows),
        ]
    )


# -- erasure -------------------------------------------------------------------


def erasure_plan_to_markdown(plan: ErasurePlan) -> str:
    """An erasure plan, with incompleteness stated first rather than buried."""
    # Deliberately no subject digest: rendering one here would have to be unsalted,
    # and an unsalted digest of a low-entropy identifier is reversible. The salted
    # digest lives in the proof artifact.
    reference = f" [`{plan.request.reference}`]" if plan.request.reference else ""
    header = (
        f"**Erasure plan**{reference} — on `{plan.request.key_column}`, "
        f"{len(plan.targets)} dataset(s)"
    )
    lines = [header, ""]
    if not plan.is_complete:
        lines.append("> **INCOMPLETE.** Some copies cannot be destroyed by this tool. ")
        lines.append("> Do not report the request as fulfilled.")
        lines.append("")
    rows = [
        [
            md.code(target.dataset),
            target.mode.value,
            text.join_truncated((f"`{k}`" for k in (str(k) for k in target.partitions)), 4)
            or "whole",
            "yes" if target.widened else "",
            target.blocked or "",
        ]
        for target in plan.targets
    ]
    lines.append(md.table(["Dataset", "Mode", "Partitions", "Widened", "Blocked"], rows))
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
        md.table(
            ["Dataset", "Status", "Mode", "Rows deleted"],
            [
                [
                    md.code(entry.get("dataset")),
                    entry.get("status"),
                    entry.get("mode"),
                    entry.get("rows_deleted"),
                ]
                for entry in proof.entries
            ],
        ),
    ]
    return "\n".join(lines)


# -- shadow --------------------------------------------------------------------


def shadow_to_markdown(report: ShadowReport) -> str:
    """Shadow results, leading with soundness because nothing else matters first."""
    verdict = "SOUND" if report.is_sound else f"UNSOUND — {report.missed_total} missed"
    rows = [
        [
            md.code(r.dataset),
            "yes" if r.is_sound else "**NO**",
            len(r.planned),
            len(r.actual),
            r.total,
            f"{r.savings:.0%}",
            f"{r.precision:.0%}",
        ]
        for r in report.results
    ]
    return "\n".join(
        [
            f"**Shadow run** — {verdict}, {report.savings:.0%} of partitions skipped",
            "",
            md.table(
                ["Dataset", "Sound", "Planned", "Actual", "Total", "Savings", "Precision"],
                rows,
            ),
        ]
    )


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
    return md.bullets(md.code(k) for k in sorted(str(k) for k in keys))


# -- the newer artifacts -------------------------------------------------------
#
# Each of these renders a report whose *caveat* is as important as its contents, so
# every one carries the qualifying sentence its module carries. A Markdown table that
# drops the caveat turns "no reads observed in 90 days" into "unused", which is the
# one reading the underlying module refuses.


def completeness_to_markdown(result: CompletenessReport) -> str:
    """Missing partitions as runs, with the inferred domains stated."""
    if result.is_complete:
        return (
            f"**Completeness** — {md.code(result.dataset)}: complete, "
            f"{result.present} partition(s)."
        )
    rows = [
        [md.cell(run), run.count, ", ".join(f"{k}={v}" for k, v in run.within) or md.ABSENT]
        for run in result.runs
    ]
    lines = [
        f"**Completeness** — {md.code(result.dataset)}: {len(result.absent)} of "
        f"{result.expected} partition(s) missing ({result.ratio:.0%} present)",
        "",
        md.table(["Run", "Buckets", "Slice"], rows, limit=20),
    ]
    if result.assumed_domains:
        inferred = ", ".join(f"`{k}`" for k in sorted(result.assumed_domains))
        lines += [
            "",
            md.note(
                f"Domains for {inferred} were inferred from observed data. A value that "
                "has never appeared cannot be reported missing."
            ),
        ]
    return "\n".join(lines)


def usage_to_markdown(stats: Mapping[DatasetId, UsageStats], *, limit: int = 20) -> str:
    """Observed reads per dataset, busiest first."""
    ranked = sorted(stats.values(), key=lambda s: (-s.reads, str(s.dataset)))
    rows = [
        [
            md.code(s.dataset),
            s.reads,
            len(s.principals),
            len(s.human_principals),
            s.last_read.date().isoformat() if s.last_read else md.ABSENT,
        ]
        for s in ranked
    ]
    return "\n".join(
        [
            f"**Usage** — {len(stats)} dataset(s) with observed reads",
            "",
            md.table(["Dataset", "Reads", "Principals", "People", "Last read"], rows, limit=limit),
        ]
    )


def retirement_to_markdown(candidates: Sequence[RetirementCandidate]) -> str:
    """Retirement candidates, with the caveat that makes them a review list."""
    if not candidates:
        return "**Retirement candidates** — none."
    rows = [
        [
            md.code(c.dataset),
            c.reason,
            c.descendants_checked,
            c.last_read.date().isoformat() if c.last_read else md.ABSENT,
        ]
        for c in candidates
    ]
    return "\n".join(
        [
            f"**Retirement candidates** — {len(candidates)} dataset(s) to review",
            "",
            md.table(["Dataset", "Why", "Descendants checked", "Last read"], rows, limit=30),
            "",
            md.note(
                "No reads observed is not the same as no reads. Query logs have retention "
                "limits, and a table read once a year for a filing looks identical here to "
                "a dead one. This is a review list, not a delete list."
            ),
        ]
    )


def risk_to_markdown(result: RiskReport) -> str:
    """Re-identification findings, and what a clear result does not mean."""
    rows = [[f.kind, ", ".join(f.columns), f.severity.value, f.detail] for f in result.findings]
    head = f"**Re-identification** — {md.code(result.dataset)}: " + (
        "no risk proven"
        if result.is_clear
        else f"{len(result.findings)} risk(s) proven at k={result.k_threshold}"
    )
    lines = [head, "", md.table(["Kind", "Columns", "Severity", "Detail"], rows)]
    if result.unmeasurable:
        lines += [
            "",
            md.note(
                "Not measurable, no distinct count profiled: "
                + ", ".join(f"`{c}`" for c in sorted(result.unmeasurable))
            ),
        ]
    lines += [
        "",
        md.note(
            "A clear result means no risk was proven, not that the data is safe. The "
            "minimum group size needs a scan this does not do."
        ),
    ]
    return "\n".join(lines)


def contract_report_to_markdown(result: ContractReport) -> str:
    """Contract breaches, with who each one is owed to."""
    rows = [
        [b.kind, b.severity.value, b.detail, ", ".join(b.consumers) or md.ABSENT]
        for b in result.breaches
    ]
    lines = [
        f"**Contract** — {md.code(result.contract.dataset)} by {result.contract.producer}: "
        + ("met" if result.is_met else f"{len(result.breaches)} breach(es)"),
        "",
        md.table(["Kind", "Severity", "Detail", "Owed to"], rows),
    ]
    if result.unchecked:
        lines += ["", md.note("Not checked: " + "; ".join(result.unchecked))]
    return "\n".join(lines)


def lifetime_to_markdown(totals: Mapping[DatasetId, LifetimeCost], *, limit: int = 20) -> str:
    """What each dataset has cost across its recorded runs."""
    ranked = sorted(totals.values(), key=lambda t: -t.spend)
    rows = [
        [
            md.code(t.dataset),
            f"{t.spend:,.2f}",
            t.runs,
            t.partitions,
            t.span.days if t.span else md.ABSENT,
        ]
        for t in ranked
    ]
    return "\n".join(
        [
            f"**Lifetime cost** — {len(totals)} measured dataset(s)",
            "",
            md.table(["Dataset", "Spend", "Runs", "Partitions", "Days"], rows, limit=limit),
        ]
    )


def value_to_markdown(findings: Sequence[ValueFinding], *, limit: int = 20) -> str:
    """Cost against usage, review list first."""
    rows = [
        [
            md.code(f.dataset),
            f.verdict.value,
            f"{f.spend:,.2f}" if f.spend is not None else md.ABSENT,
            f.reads,
        ]
        for f in findings
    ]
    actionable = [f for f in findings if f.is_actionable]
    return "\n".join(
        [
            f"**Value** — {len(actionable)} dataset(s) unread and above the threshold",
            "",
            md.table(["Dataset", "Verdict", "Spend", "Reads"], rows, limit=limit),
            "",
            md.note(
                "Cost is measured; usage is observed. A table read once a year for a "
                "filing looks identical here to a dead one."
            ),
        ]
    )


def history_to_markdown(history: History, *, limit: int = 20) -> str:
    """The graph's revisions, newest first, with the unsafe ones marked."""
    rows = [
        [
            r.digest,
            r.at.date().isoformat(),
            md.cell(r.author),
            md.cell(r.note),
            "" if r.is_initial or r.is_safe else "**unsafe**",
        ]
        for r in reversed(list(history.revisions)[-limit:])
    ]
    return "\n".join(
        [
            f"**Graph history** — {len(history)} revision(s)",
            "",
            md.table(["Digest", "When", "Author", "Note", "Safety"], rows),
        ]
    )


def restatement_to_markdown(impact: RestatementImpact) -> str:
    """What has already been published downstream of a dataset."""
    if not impact.is_published:
        return (
            f"**Restatement impact** — {md.code(impact.dataset)}: nothing published "
            f"downstream; {len(impact.tables)} table(s) to rebuild."
        )
    rows = [[describe_sink(s), "yes" if s in impact.regulatory else ""] for s in impact.sinks]
    return "\n".join(
        [
            f"**Restatement impact** — {md.code(impact.dataset)}: "
            f"{len(impact.sinks)} published artefact(s), {len(impact.tables)} table(s)",
            "",
            md.table(["Artefact", "Regulatory"], rows, limit=40),
            "",
            md.note(
                "Downstream is not the same as material. That judgement is not the graph's to make."
            ),
        ]
    )


def seasonal_to_markdown(baseline: SeasonalBaseline, *, limit: int = 40) -> str:
    """Learned bands per cycle bucket, and the buckets deliberately left unmodelled.

    The unmodelled list is the part that must survive rendering. Without it, a bucket
    that is never checked is indistinguishable from one that always passes.
    """
    if not baseline.is_usable:
        return f"**Seasonal baseline** — {md.code(baseline.dataset)}: {baseline.summary()}"

    rows = [
        [
            band.label or band.bucket,
            md.code(band.column) if band.column else md.ABSENT,
            band.metric,
            f"{band.low:g}",
            f"{band.high:g}",
            band.observations,
        ]
        for band in sorted(
            baseline.bands.values(), key=lambda b: (b.bucket, b.column or "", b.metric)
        )
    ]
    lines = [
        f"**Seasonal baseline** — {md.code(baseline.dataset)} by {baseline.cycle.value}: "
        f"{len(baseline.bands)} band(s) across {len(baseline.modelled_buckets)} bucket(s)",
        "",
        md.table(["Bucket", "Column", "Metric", "Low", "High", "Seen"], rows, limit=limit),
    ]
    if baseline.unmodelled:
        skipped = ", ".join(
            f"{label} ({baseline.unmodelled[b]})"
            for b, label in zip(sorted(baseline.unmodelled), _labels(baseline), strict=True)
        )
        lines += [
            "",
            md.note(
                f"Not modelled, too few observations: {skipped}. These buckets are **not "
                "checked** — silence there means unmodelled, not passing."
            ),
        ]
    return "\n".join(lines)


def _labels(baseline: SeasonalBaseline) -> list[str]:
    from ..observe.seasonal import unmodelled_buckets

    return unmodelled_buckets(baseline)
