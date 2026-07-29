"""The `Project` facade — what makes this one tool rather than a library of parts.

Everything the CLI does goes through here, so the Python API and the command line
cannot drift apart. A `Project` binds a config file to a store and resolves adapters
on demand.

Live connections are injected, never configured. `fathom.yml` declares *that* a
dataset lives in Snowflake and how it is partitioned; the caller supplies the
connection with `register_runner`. That keeps credentials out of a file people want
to commit, and it is why the warehouse adapters take a runner rather than a DSN.

The verbs compose in one direction:

    detect  ->  plan  ->  (apply)      what changed, and what that invalidates
    profile ->  check                  what the data looks like, and what moved
    label   ->  enforce                what columns mean, and what policy allows
    locate  ->  erase                  where a subject is, and how to destroy it
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..adapters.base import ChangeSet
from ..core.errors import ConfigError
from ..core.ids import is_path_dataset
from ..core.types import (
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    DatasetId,
    ErasureMode,
    KeyPredicate,
    LineageSource,
)
from ..govern.erasure import ErasurePlan, ErasureRequest, plan_erasure
from ..govern.policy import LabelSet, PolicyReport, SinkPolicy, enforce, infer, propagate
from ..graph import sinks
from ..graph.diff import GraphDiff
from ..graph.history import History, Revision, graph_digest, record
from ..graph.model import Graph, InvalidationPlan
from ..ingest.events import IngestResult, graph_from_lineage, graph_from_queries
from ..observe.profile import Finding, Profile, Severity, drift
from ..store.sqlite import Store
from .config import ProjectConfig, load_config

__all__ = ["Project"]

# Where the lineage cursor lives. Not a real dataset, just a stable key in the store.
_LINEAGE_MARKER = DatasetId(namespace="fathom", name="_lineage_cursor")


@dataclass
class Project:
    """A config file, a store, and the adapters they imply."""

    config: ProjectConfig
    store: Store
    runners: dict[str, Any] = field(default_factory=dict)
    _adapters: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- lifecycle -------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None, *, store: Store | None = None) -> Project:
        """Open a project from a config file, creating its store if needed."""
        config = load_config(path)
        if store is None:
            config.store.parent.mkdir(parents=True, exist_ok=True)
            store = Store(config.store)
        return cls(config=config, store=store)

    def close(self) -> None:
        """Close the store."""
        self.store.close()

    def __enter__(self) -> Project:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def register_runner(self, name: str, runner: Any) -> None:
        """Supply a live connection for a warehouse adapter.

        Configs declare shape; callers supply credentials. Nothing here ever reads a
        password from the config file.
        """
        self.runners[name] = runner
        self._adapters.pop(name, None)

    # -- adapters --------------------------------------------------------------

    def _sniff(self, dataset: DatasetId) -> str:
        """Pick an adapter for a path dataset by looking at what is actually there."""
        from ..adapters import DeltaCatalog

        options = self.config.options_for(dataset.namespace.split("://")[0])
        if DeltaCatalog(storage_options=options).is_delta_table(dataset):
            return "delta"
        try:
            from ..adapters.catalogs.iceberg import IcebergCatalog

            if IcebergCatalog(storage_options=options).is_iceberg_table(dataset):
                return "iceberg"
        except ImportError:  # pragma: no cover - only without the extra
            pass
        return "storage"

    def adapter(self, name: str) -> Any:
        """Build (and cache) a configured adapter by registry name."""
        if name in self._adapters:
            return self._adapters[name]

        from ..adapters import get_adapter

        options = dict(self.config.adapters.get(name, {}))
        if name in self.runners:
            options["runner"] = self.runners[name]
        if name in {"delta", "iceberg", "storage"} and "storage_options" not in options:
            merged: dict[str, Any] = {}
            for per_protocol in self.config.storage_options.values():
                merged.update(per_protocol)
            if merged:
                options["storage_options"] = merged

        try:
            built = get_adapter(name)(**options)
        except KeyError as exc:
            raise ConfigError(f"unknown adapter {name!r}") from exc
        except TypeError as exc:
            raise ConfigError(f"adapter {name!r} rejected its options: {exc}") from exc

        self._adapters[name] = built
        return built

    def adapter_for(self, dataset: DatasetId) -> tuple[Any, str]:
        """The adapter that owns a dataset, declared or sniffed."""
        entry = self.config.dataset(dataset)
        name = entry.adapter if entry and entry.adapter else ""
        if not name:
            name = self._sniff(dataset) if is_path_dataset(dataset) else self.config.system
        adapter = self.adapter(name)

        # Push declared partitioning down so the adapter does not have to infer it.
        if entry is not None and entry.spec.fields and hasattr(adapter, "declare"):
            kwargs: dict[str, Any] = {}
            if entry.template is not None and name == "storage":
                adapter.declare(dataset, entry.spec, entry.template)
            else:
                if entry.watermark and name == "snowflake":
                    kwargs["watermark"] = entry.watermark
                if entry.location and name == "databricks":
                    kwargs["location"] = entry.location
                adapter.declare(dataset, entry.spec, **kwargs)
        return adapter, name

    def is_configured(self, dataset: DatasetId) -> bool:
        """True when an adapter was chosen deliberately rather than by fallback.

        This gates erasure. A bare table name with no `adapter:` falls back to the
        project's default system, which is a guess — and an erasure plan that acts
        on a guess is how a request gets reported fulfilled while the data survives
        somewhere nobody configured.
        """
        entry = self.config.dataset(dataset)
        if entry is not None and entry.adapter:
            return True
        return is_path_dataset(dataset)

    def capabilities(self, *, graph: Graph | None = None) -> dict[DatasetId, Capabilities]:
        """What can be done to each declared dataset, for erasure planning.

        Pass `graph` to reuse one already loaded; otherwise this reads the whole
        graph back out of the store.
        """
        out: dict[DatasetId, Capabilities] = {}
        for dataset in (graph if graph is not None else self.graph()).datasets:
            if not self.is_configured(dataset):
                continue
            try:
                adapter, _ = self.adapter_for(dataset)
            except (ConfigError, KeyError):
                continue
            caps = getattr(adapter, "capabilities", None)
            if caps is None:
                continue
            # An adapter that needs a live connection cannot erase without one.
            if hasattr(adapter, "runner") and adapter.runner is None:
                continue
            out[dataset] = caps
        return out

    # -- graph -----------------------------------------------------------------

    def graph(self) -> Graph:
        """The stored graph, with declared specs and publications applied on top.

        Publications are re-applied on every load rather than persisted as ordinary
        edges, so removing one from `fathom.yml` removes it from the graph. A declared
        artefact that outlived its declaration would keep appearing in restatement
        notices, and the whole value of those notices is that they are current.
        """
        graph = self.store.load_graph()
        for dataset, spec in self.config.specs.items():
            graph.add_dataset(dataset, spec)
        for publication in self.config.publications:
            sink = sinks.of_kind(publication.kind, publication.name, publication.instance)
            if graph.out_edges(sink):
                continue  # already applied by an earlier load of the same config
            sinks.record_publication(graph, sink, publication.inputs, evidence="declared:config")
        return graph

    def ingest(self, *, save: bool = True, author: str = "", note: str = "") -> IngestResult:
        """Build the graph from every configured lineage source.

        `author` and `note` are recorded against the resulting revision, so the
        history can answer who changed an edge and why rather than only when.

        Sources accumulate into one graph rather than competing: a dbt manifest and
        an OpenLineage stream describing the same pipeline reinforce each other,
        because edges are keyed by evidence.
        """
        from ..ingest import ingest_dbt, ingest_openlineage, load_events

        result = IngestResult(graph=Graph())
        # Evidence prefixes this run rebuilt from scratch. A file-backed source is
        # read in full every time, so whatever it no longer reports is a dependency
        # that genuinely went away and must leave the store with it. Adapter sources
        # resume from a token and report only deltas, so they are never listed here.
        regenerated: set[str] = set()
        specs = self.config.specs
        for dataset, spec in specs.items():
            result.graph.add_dataset(dataset, spec)

        for source in self.config.lineage:
            if source.kind == "sql":
                regenerated.add("sql:")
                from ..adapters.base import QueryEvent

                queries = []
                for pattern in source.paths:
                    for path in sorted(_expand_glob(pattern)):
                        queries.append(
                            QueryEvent(
                                sql=path.read_text(), dialect=source.dialect, query_id=path.name
                            )
                        )
                merged = graph_from_queries(
                    queries,
                    dialect=source.dialect,
                    system=self.config.system,
                    instance=self.config.instance,
                    specs=specs,
                    graph=result.graph,
                )
            elif source.kind == "dbt":
                regenerated.add("dbt:")
                if source.manifest is None:
                    raise ConfigError("a dbt lineage source needs a `manifest` path")
                merged = ingest_dbt(str(source.manifest), specs=specs, graph=result.graph)
            elif source.kind == "openlineage":
                regenerated.add("openlineage:")
                if not source.events:
                    raise ConfigError("an openlineage lineage source needs an `events` path")
                events = load_events(source.events)
                merged = ingest_openlineage(events, specs=specs, graph=result.graph)
            else:  # adapter
                adapter = self.adapter(source.adapter or self.config.system)
                token = self.store.get_token(_LINEAGE_MARKER, source.adapter or "adapter")
                native = list(adapter.fetch_lineage(token))
                if native:
                    merged = graph_from_lineage(native, specs=specs, graph=result.graph)
                    if hasattr(adapter, "lineage_token"):
                        advanced = adapter.lineage_token(native)
                        if advanced:
                            self.store.set_token(
                                _LINEAGE_MARKER, source.adapter or "adapter", advanced
                            )
                else:
                    # A warehouse query log identifies each *execution*, not each
                    # dependency. Keeping that id in the evidence would mint a fresh
                    # edge every time the model runs.
                    label = source.adapter or self.config.system
                    merged = graph_from_queries(
                        list(adapter.fetch_queries(token)),
                        dialect=getattr(adapter, "dialect", self.config.system),
                        system=self.config.system,
                        instance=self.config.instance,
                        specs=specs,
                        graph=result.graph,
                        evidence_label=f"{label}:query_log",
                    )

            result.statements += merged.statements
            result.unparsed += merged.unparsed
            result.notes.extend(merged.notes)

        # Models declared inline get parsed too, so a project with no lineage block
        # still produces a graph.
        inline = [m for m in self.config.models if self.config.sql_for(m)]
        if inline:
            from ..adapters.base import QueryEvent

            queries = [
                QueryEvent(
                    sql=self.config.sql_for(m) or "",
                    dialect=self.config.system,
                    query_id=m.raw_name,
                )
                for m in inline
            ]
            merged = graph_from_queries(
                queries,
                dialect=self.config.system,
                system=self.config.system,
                instance=self.config.instance,
                specs=specs,
                graph=result.graph,
            )
            result.statements += merged.statements
            result.notes.extend(merged.notes)

        if inline:
            regenerated.add("sql:")

        if save:
            previous = self.store.load_graph()
            self.store.save_graph(result.graph, replace_evidence=sorted(regenerated))
            self._record_revision(previous, result.graph, author=author, note=note)
        return result

    def _record_revision(
        self, previous: Graph, current: Graph, *, author: str = "", note: str = ""
    ) -> None:
        """Append this ingest to the graph's revision history, if it changed anything.

        Recorded here rather than left to the caller because a history nobody
        remembers to write is a history that is empty on the day it is needed. An
        unchanged graph appends nothing, so a nightly ingest that found no new lineage
        does not fill the log with noise.

        The chain is rebuilt from the store on every call rather than held in memory,
        so two processes ingesting concurrently cannot produce a revision whose parent
        is a graph neither of them saw.
        """
        log = History()
        for stored in self.store.revisions():
            log.revisions.append(
                Revision(
                    digest=stored["digest"],
                    at=stored["at"],
                    author=stored["author"],
                    note=stored["note"],
                    diff=GraphDiff(),
                    parent=stored["parent"],
                    datasets=stored["datasets"],
                    edges=stored["edges"],
                )
            )

        head = log.head
        if head is not None and head.digest == graph_digest(current):
            return
        try:
            revision = record(
                log,
                current,
                author=author,
                note=note,
                previous=previous if head is not None else None,
            )
        except ValueError:
            # The head describes a graph this process never held — another writer got
            # there first. Skipping is right: appending anyway would attribute their
            # change to this run, which is worse than a gap in the log.
            return
        self.store.record_revision(revision)

    # -- detect and plan -------------------------------------------------------

    def detect(self, *, save: bool = True) -> dict[DatasetId, ChangeSet]:
        """Ask every source adapter what changed since its stored token."""
        out: dict[DatasetId, ChangeSet] = {}
        for entry in self.config.sources:
            adapter, name = self.adapter_for(entry.dataset)
            token = self.store.get_token(entry.dataset, name)
            changes = adapter.changed(entry.dataset, token)
            out[entry.dataset] = changes
            if save and changes.token:
                self.store.set_token(entry.dataset, name, changes.token)
        return out

    def plan(
        self,
        seeds: Mapping[DatasetId, Iterable[KeyPredicate]] | None = None,
        *,
        detect: bool = False,
    ) -> InvalidationPlan:
        """What must be rebuilt. With `detect`, discover the seeds first."""
        if seeds is None:
            if not detect:
                raise ValueError("pass seeds explicitly, or plan(detect=True)")
            seeds = {ds: c.partitions for ds, c in self.detect().items() if c.partitions}
        return self.graph().invalidate(seeds)

    # -- profile and check -----------------------------------------------------

    def profile(self, dataset: DatasetId, *, partition: KeyPredicate | None = None) -> Profile:
        """Profile one dataset through whichever adapter owns it."""
        adapter, _ = self.adapter_for(dataset)
        if not hasattr(adapter, "profile"):
            raise ConfigError(
                f"the adapter for {dataset} cannot profile; only storage-backed "
                "datasets support footer profiling today"
            )
        got: Profile = adapter.profile(dataset, partition=partition)
        return got

    def check(
        self, *, save: bool = True, rebaseline_on_error: bool = False
    ) -> dict[DatasetId, list[Finding]]:
        """Profile every path-backed dataset and diff against its last profile.

        A profile that just failed its own check does not become the next baseline.
        Adopting it means the following run compares the broken shape against itself
        and reports no drift, so a dropped column or a changed type alerts exactly
        once and is then silently normalised — the failure mode that makes people
        stop trusting a drift monitor. Pass `rebaseline_on_error=True` to accept the
        new shape deliberately.
        """
        out: dict[DatasetId, list[Finding]] = {}
        for entry in self.config.datasets:
            if not is_path_dataset(entry.dataset):
                continue
            try:
                current = self.profile(entry.dataset)
            except (ConfigError, FileNotFoundError):
                continue
            previous = self.store.latest_profile(entry.dataset)
            findings = drift(previous, current) if previous else []
            out[entry.dataset] = findings
            broken = any(f.severity is Severity.ERROR for f in findings)
            if save and (rebaseline_on_error or not broken):
                self.store.save_profile(current)
        return out

    # -- label and enforce -----------------------------------------------------

    def labels(self, *, save: bool = True) -> LabelSet:
        """Infer labels from stored profiles and propagate them along the graph."""
        seeds: LabelSet = {}
        for dataset in self.store.datasets():
            latest = self.store.latest_profile(dataset)
            if latest is not None:
                seeds.update(infer(latest))

        graph = self.graph()
        labels = propagate(graph, seeds) if graph.edges else seeds

        if save:
            for ref, values in labels.items():
                for entry in values:
                    self.store.set_label(
                        ref.dataset,
                        ref.column,
                        entry.name,
                        confidence=entry.confidence,
                        origin=entry.origin,
                        confirmed=entry.confirmed,
                    )
        return labels

    def enforce(self, labels: LabelSet | None = None) -> PolicyReport:
        """Check configured sink policies against current labels."""
        policies = [
            SinkPolicy(dataset=p.dataset, forbid=p.forbid, require=p.require, reason=p.reason)
            for p in self.config.policies
        ]
        return enforce(labels if labels is not None else self.labels(save=False), policies)

    # -- erase -----------------------------------------------------------------

    def locate(self, request: ErasureRequest) -> ErasurePlan:
        """Where a subject's data reached, and whether it can be destroyed."""
        # One load. `graph()` reads and rebuilds the whole graph from SQLite on every
        # call, and locating a subject used to do it four times over — once here,
        # once inside `capabilities`, and twice more below — producing four separate
        # Graph objects for one question.
        graph = self.graph()
        capabilities = self.capabilities(graph=graph)
        files: dict[DatasetId, Sequence[str]] = {}
        for dataset in graph.datasets:
            # An unconfigured dataset gets no erasure capability, so the plan
            # reports it blocked rather than assuming it can be deleted from.
            capabilities.setdefault(
                dataset,
                Capabilities(
                    lineage=LineageSource.DECLARED,
                    change=ChangeSource.WATERMARK,
                    erasure=ErasureMode.NONE,
                ),
            )
            if not is_path_dataset(dataset):
                continue
            try:
                adapter, _ = self.adapter_for(dataset)
            except ConfigError:
                continue
            if hasattr(adapter, "files_for"):
                files[dataset] = adapter.files_for(dataset)
        return plan_erasure(graph, request, capabilities=capabilities, files=files)

    # -- diagnostics -----------------------------------------------------------

    def doctor(self) -> list[str]:
        """Configuration problems that would silently degrade a plan.

        Every item here is something that produces a *working* but worse result, so
        none of them raise. They are exactly the things a user would otherwise
        discover as "why is it rebuilding everything".
        """
        problems: list[str] = []
        graph = self.graph()

        declared = set(self.config.specs)
        for dataset in graph.datasets:
            if graph.spec(dataset) is UNPARTITIONED or not graph.spec(dataset).fields:
                if dataset in declared:
                    continue
                problems.append(
                    f"{dataset}: no partition spec, so any change rebuilds the whole dataset"
                )

        for edge in graph.edges:
            if edge.mapping.is_unbounded and edge.mapping.fields:
                problems.append(
                    f"{edge.src} -> {edge.dst}: unbounded mapping ({edge.evidence}); "
                    "every source change rebuilds the whole target"
                )
            if not edge.columns:
                problems.append(
                    f"{edge.src} -> {edge.dst}: no column lineage, so drift cannot be "
                    "attributed and labels propagate unattributed"
                )

        # Only datasets something can actually scan; a bare warehouse table with no
        # adapter is not "unscanned", it is not scannable, and saying otherwise is noise.
        for entry in self.config.sources:
            if not self.is_configured(entry.dataset):
                continue
            if not self.store.get_token(entry.dataset, entry.adapter or "storage"):
                problems.append(f"{entry.dataset}: never scanned; run `fathom detect`")

        if not graph.edges:
            problems.append("no lineage in the store; run `fathom ingest`")

        problems.extend(self._declaration_problems())
        return problems

    def _declaration_problems(self) -> list[str]:
        """Ways a declared publication or contract is quietly not doing its job.

        Everything declared in `fathom.yml` rather than discovered can point at
        nothing and still parse. A contract on a dataset that is never profiled looks
        met on every run, and a publication whose input left the graph names the wrong
        blast radius in a restatement notice — both fail by being silently vacuous,
        which is the failure `doctor` exists to surface.

        Membership is tested against the *ingested* graph plus declared specs, not
        against `self.graph()`. Applying a publication registers its endpoints, so a
        mistyped input becomes a real node the moment the graph is built — and
        checking after that would only ever confirm the typo it created.
        """
        problems: list[str] = []
        known = set(self.store.load_graph().datasets) | set(self.config.specs)
        known.update(entry.dataset for entry in self.config.datasets)

        for publication in self.config.publications:
            missing = [str(i) for i in publication.inputs if i not in known]
            if missing:
                problems.append(
                    f"publication {publication.name!r}: input(s) {', '.join(sorted(missing))} "
                    "are not in the graph, so a restatement notice would miss what "
                    "actually feeds it"
                )

        for contract in self.config.contracts:
            if contract.dataset not in known:
                problems.append(
                    f"contract on {contract.dataset}: the dataset is not in the graph, "
                    "so nothing this contract promises is being checked"
                )
                continue
            if not contract.consumers:
                problems.append(
                    f"contract on {contract.dataset}: no consumers named, so a breach "
                    "is only a warning and escalates to nobody"
                )
            needs_profile = bool(contract.columns) or contract.max_staleness is not None
            if needs_profile and self.store.latest_profile(contract.dataset) is None:
                problems.append(
                    f"contract on {contract.dataset}: never profiled, so its column and "
                    "staleness promises report as unchecked on every run"
                )

        return problems


def _expand_glob(pattern: Path) -> list[Path]:
    """Expand a path that may contain glob characters."""
    text = str(pattern)
    positions = [text.find(c) for c in "*?[" if text.find(c) >= 0]
    if not positions:
        return [pattern] if pattern.is_file() else []
    head = Path(text[: min(positions)])
    root = head if head.is_dir() else head.parent
    try:
        relative = str(pattern.relative_to(root))
    except ValueError:
        return []
    return sorted(p for p in root.glob(relative) if p.is_file())
