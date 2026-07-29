"""Provenance for what autonomous programs actually did.

An agent is a process that decides at runtime which data to read and which tools to
call. Nobody reviewed those decisions in a pull request, because they were not
written down anywhere before they happened. That is the whole difficulty: a dbt
model's access pattern is a diff, and an agent's is a log — if one was kept.

Recording tool calls as graph edges makes four questions answerable that are
otherwise guesswork:

- **What did it read?** Including transitively: an agent that queried one gold table
  read everything upstream of it, and that closure is where the personal data lives.
- **What did it write?** And therefore what downstream of those writes is now built
  on output nobody reviewed.
- **Where could data have left?** A tool that posts to an external endpoint is an
  egress point. Combined with labels, `exfiltration_paths` names every route by
  which something sensitive could have gone out.
- **Was this normal?** `first_time_access` compares one run against history, which
  is the cheapest useful anomaly signal and needs no model of its own.

`least_privilege_gap` runs the comparison in the other direction: grants the agent
holds and has never used. Those are the ones to revoke, and the argument for
revoking them is the evidence rather than a policy preference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..core.types import DatasetId
from ..govern.policy import LabelSet, labels_over
from ..graph.model import Graph, link
from ..graph.query import ancestors, descendants
from .assets import AssetKind, spec_for

__all__ = [
    "AgentRun",
    "RiskReport",
    "ToolCall",
    "access_frequency",
    "blast_radius",
    "unreviewed_writes",
    "datasets_read",
    "datasets_written",
    "egress_points",
    "exfiltration_paths",
    "first_time_access",
    "labels_touched",
    "least_privilege_gap",
    "reach",
    "record_agent_run",
    "risk_report",
    "tools_used",
]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation, and what it touched.

    `egress` marks a tool that can move data outside the boundary — an HTTP client,
    an email sender, a webhook. It is declared rather than inferred, because whether
    a tool is an exit depends on how it is configured, and guessing in either
    direction is worse than asking.
    """

    tool: DatasetId
    reads: tuple[DatasetId, ...] = ()
    writes: tuple[DatasetId, ...] = ()
    egress: bool = False
    occurred: datetime = field(default_factory=lambda: datetime.now(UTC))
    detail: str = ""

    def __str__(self) -> str:
        arrow = f"{len(self.reads)}r/{len(self.writes)}w"
        return f"{self.tool} [{arrow}]{' EGRESS' if self.egress else ''}"


@dataclass
class AgentRun:
    """One agent invocation and every tool call it made."""

    agent: DatasetId
    run_id: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    started: datetime = field(default_factory=lambda: datetime.now(UTC))

    def call(
        self,
        tool: DatasetId,
        *,
        reads: Iterable[DatasetId] = (),
        writes: Iterable[DatasetId] = (),
        egress: bool = False,
        detail: str = "",
    ) -> ToolCall:
        """Record one tool call."""
        item = ToolCall(
            tool=tool,
            reads=tuple(reads),
            writes=tuple(writes),
            egress=egress,
            detail=detail,
        )
        self.calls.append(item)
        return item

    def summary(self) -> str:
        """The run as text: what it touched and what it was allowed to touch."""
        return (
            f"{self.agent} run {self.run_id or '(unnamed)'}: {len(self.calls)} call(s), "
            f"{len(datasets_read(self))} read, {len(datasets_written(self))} written"
        )


def datasets_read(run: AgentRun) -> list[DatasetId]:
    """Datasets this run read directly."""
    out: set[DatasetId] = set()
    for call in run.calls:
        out.update(call.reads)
    return sorted(out, key=str)


def datasets_written(run: AgentRun) -> list[DatasetId]:
    """Datasets this run wrote."""
    out: set[DatasetId] = set()
    for call in run.calls:
        out.update(call.writes)
    return sorted(out, key=str)


def tools_used(run: AgentRun) -> list[DatasetId]:
    """Distinct tools invoked."""
    return sorted({call.tool for call in run.calls}, key=str)


def egress_points(run: AgentRun) -> list[ToolCall]:
    """Calls that could have moved data outside the boundary."""
    return [call for call in run.calls if call.egress]


def record_agent_run(graph: Graph, run: AgentRun) -> Graph:
    """Write an agent run into the graph as read and write edges.

    Every edge is unbounded. An agent's transformation of its inputs is opaque by
    construction, so claiming any partition relationship would be inventing one, and
    the planner would then trust it.
    """
    agent_spec = spec_for(AssetKind.AGENT)
    graph.add_dataset(run.agent, agent_spec)
    evidence = f"agent:{run.run_id}" if run.run_id else "agent"

    for call in run.calls:
        link(
            graph,
            call.tool,
            run.agent,
            evidence=f"{evidence}:tool",
            src_spec=spec_for(AssetKind.TOOL),
            dst_spec=agent_spec,
        )
        for source in call.reads:
            link(graph, source, run.agent, evidence=f"{evidence}:read", dst_spec=agent_spec)
        for target in call.writes:
            link(graph, run.agent, target, evidence=f"{evidence}:write")
    return graph


def reach(graph: Graph, run: AgentRun) -> list[DatasetId]:
    """Everything this run could have seen, including transitively.

    An agent that read one aggregate read everything behind it, in the sense that
    matters for a disclosure question.
    """
    out: set[DatasetId] = set()
    for source in datasets_read(run):
        out.add(source)
        out.update(ancestors(graph, source))
    return sorted(out, key=str)


def labels_touched(graph: Graph, run: AgentRun, labels: LabelSet) -> dict[str, list[DatasetId]]:
    """Labels carried by anything within this run's reach, grouped by label name."""
    return {
        name: sorted({ref.dataset for ref in refs}, key=str)
        for name, refs in labels_over(labels, reach(graph, run)).items()
    }


def exfiltration_paths(
    graph: Graph, run: AgentRun, labels: LabelSet, *, sensitive: Iterable[str] = ("pii",)
) -> list[str]:
    """Routes by which labelled data could have left through an egress tool.

    Says *could have*, deliberately. It reports that sensitive data was in reach and
    an egress tool was called in the same run — not that one flowed into the other,
    which nothing outside the agent's own process can establish.
    """
    exits = egress_points(run)
    if not exits:
        return []

    watched = set(sensitive)
    present = {
        name: datasets
        for name, datasets in labels_touched(graph, run, labels).items()
        if name in watched
    }
    if not present:
        return []

    out: list[str] = []
    for name, datasets in sorted(present.items()):
        for call in exits:
            out.append(
                f"{run.agent} read {len(datasets)} dataset(s) labelled `{name}` and called "
                f"the egress tool {call.tool} in the same run"
            )
    return out


def first_time_access(run: AgentRun, history: Sequence[AgentRun]) -> list[DatasetId]:
    """Datasets this run read that no prior run of the same agent ever did.

    The cheapest useful anomaly signal available, and it needs no model: an agent
    that has read the same twelve tables for a month and suddenly reads a thirteenth
    is worth a look.
    """
    seen: set[DatasetId] = set()
    for prior in history:
        if prior.run_id == run.run_id:
            continue
        seen.update(datasets_read(prior))
    return sorted(set(datasets_read(run)) - seen, key=str)


def least_privilege_gap(
    granted: Iterable[DatasetId], history: Sequence[AgentRun]
) -> list[DatasetId]:
    """Grants the agent holds and has never exercised.

    The revocation list, backed by evidence rather than by an argument about what an
    agent probably needs.
    """
    used: set[DatasetId] = set()
    for run in history:
        used.update(datasets_read(run))
        used.update(datasets_written(run))
    return sorted(set(granted) - used, key=str)


def blast_radius(graph: Graph, agent: DatasetId) -> list[DatasetId]:
    """Everything downstream of what this agent writes.

    An agent writing one table is an agent whose mistakes reach everything built on
    that table, which is usually more than the team that deployed it assumes.
    """
    return descendants(graph, agent)


def unreviewed_writes(graph: Graph, agent: DatasetId) -> list[DatasetId]:
    """Datasets that an agent writes and that other pipelines then consume.

    The interesting subset of an agent's writes: output nobody reviewed feeding
    something somebody depends on.
    """
    written = {e.dst for e in graph.out_edges(agent)}
    return sorted((ds for ds in written if graph.out_edges(ds)), key=str)


@dataclass
class RiskReport:
    """Everything worth knowing about one agent run, in one object."""

    run_id: str = ""
    agent: DatasetId | None = None
    read: list[DatasetId] = field(default_factory=list)
    written: list[DatasetId] = field(default_factory=list)
    reachable: int = 0
    sensitive_labels: dict[str, list[DatasetId]] = field(default_factory=dict)
    egress_calls: int = 0
    exfiltration: list[str] = field(default_factory=list)
    novel_access: list[DatasetId] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when nothing the agent touched breached its declared scope."""
        return not self.exfiltration and not self.novel_access

    def summary(self) -> str:
        """The run as text: what it touched and what it was allowed to touch."""
        lines = [
            f"agent run {self.run_id or '(unnamed)'}: {len(self.read)} read, "
            f"{len(self.written)} written, {self.reachable} dataset(s) in reach"
        ]
        for name, datasets in sorted(self.sensitive_labels.items()):
            lines.append(f"  label `{name}` present on {len(datasets)} reachable dataset(s)")
        for note in self.exfiltration:
            lines.append(f"  [risk] {note}")
        for ds in self.novel_access:
            lines.append(f"  [new] first access to {ds}")
        if self.is_clean:
            lines.append("  nothing anomalous")
        return "\n".join(lines)


def risk_report(
    graph: Graph,
    run: AgentRun,
    *,
    labels: LabelSet | None = None,
    history: Sequence[AgentRun] = (),
    sensitive: Iterable[str] = ("pii",),
) -> RiskReport:
    """Assemble the full picture of one agent run."""
    label_set = labels or {}
    touched = labels_touched(graph, run, label_set)
    watched = set(sensitive)
    return RiskReport(
        run_id=run.run_id,
        agent=run.agent,
        read=datasets_read(run),
        written=datasets_written(run),
        reachable=len(reach(graph, run)),
        sensitive_labels={k: v for k, v in touched.items() if k in watched},
        egress_calls=len(egress_points(run)),
        exfiltration=exfiltration_paths(graph, run, label_set, sensitive=sensitive),
        novel_access=first_time_access(run, history),
    )


def access_frequency(history: Sequence[AgentRun]) -> Mapping[str, int]:
    """How often each dataset was read across a run history.

    Feeds two decisions: which datasets to profile first, and which grants are load
    bearing rather than historical.
    """
    counts: dict[str, int] = {}
    for run in history:
        for ds in datasets_read(run):
            counts[str(ds)] = counts.get(str(ds), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
