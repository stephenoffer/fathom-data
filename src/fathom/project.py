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

from .adapters.base import ChangeSet
from .config import ProjectConfig, load_config
from .erasure import ErasurePlan, ErasureRequest, plan_erasure
from .errors import ConfigError
from .graph import Graph, InvalidationPlan
from .ids import is_path_dataset
from .ingest import IngestResult, graph_from_lineage, graph_from_queries
from .policy import LabelSet, PolicyReport, SinkPolicy, enforce, infer, propagate
from .profile import Finding, Profile, drift
from .store import Store
from .types import (
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    DatasetId,
    ErasureMode,
    KeyPredicate,
    LineageSource,
)

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
        config = load_config(path)
        if store is None:
            config.store.parent.mkdir(parents=True, exist_ok=True)
            store = Store(config.store)
        return cls(config=config, store=store)

    def close(self) -> None:
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
        from .adapters import DeltaCatalog

        options = self.config.options_for(dataset.namespace.split("://")[0])
        if DeltaCatalog(storage_options=options).is_delta_table(dataset):
            return "delta"
        try:
            from .adapters.iceberg import IcebergCatalog

            if IcebergCatalog(storage_options=options).is_iceberg_table(dataset):
                return "iceberg"
        except ImportError:  # pragma: no cover - only without the extra
            pass
        return "storage"

    def adapter(self, name: str) -> Any:
        """Build (and cache) a configured adapter by registry name."""
        if name in self._adapters:
            return self._adapters[name]

        from .adapters import get_adapter

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

    def capabilities(self) -> dict[DatasetId, Capabilities]:
        """What can be done to each declared dataset, for erasure planning."""
        out: dict[DatasetId, Capabilities] = {}
        for dataset in self.graph().datasets:
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
        """The stored graph, with declared specs applied on top."""
        graph = self.store.load_graph()
        for dataset, spec in self.config.specs.items():
            graph.add_dataset(dataset, spec)
        return graph

    def ingest(self, *, save: bool = True) -> IngestResult:
        """Build the graph from every configured lineage source.

        Sources accumulate into one graph rather than competing: a dbt manifest and
        an OpenLineage stream describing the same pipeline reinforce each other,
        because edges are keyed by evidence.
        """
        from .integrations import ingest_dbt, ingest_openlineage, load_events

        result = IngestResult(graph=Graph())
        specs = self.config.specs
        for dataset, spec in specs.items():
            result.graph.add_dataset(dataset, spec)

        for source in self.config.lineage:
            if source.kind == "sql":
                from .adapters.base import QueryEvent

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
                if source.manifest is None:
                    raise ConfigError("a dbt lineage source needs a `manifest` path")
                merged = ingest_dbt(str(source.manifest), specs=specs, graph=result.graph)
            elif source.kind == "openlineage":
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
                    merged = graph_from_queries(
                        list(adapter.fetch_queries(token)),
                        dialect=getattr(adapter, "dialect", self.config.system),
                        system=self.config.system,
                        instance=self.config.instance,
                        specs=specs,
                        graph=result.graph,
                    )

            result.statements += merged.statements
            result.unparsed += merged.unparsed
            result.notes.extend(merged.notes)

        # Models declared inline get parsed too, so a project with no lineage block
        # still produces a graph.
        inline = [m for m in self.config.models if self.config.sql_for(m)]
        if inline:
            from .adapters.base import QueryEvent

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

        if save:
            self.store.save_graph(result.graph)
        return result

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
        adapter, _ = self.adapter_for(dataset)
        if not hasattr(adapter, "profile"):
            raise ConfigError(
                f"the adapter for {dataset} cannot profile; only storage-backed "
                "datasets support footer profiling today"
            )
        got: Profile = adapter.profile(dataset, partition=partition)
        return got

    def check(self, *, save: bool = True) -> dict[DatasetId, list[Finding]]:
        """Profile every path-backed dataset and diff against its last profile."""
        out: dict[DatasetId, list[Finding]] = {}
        for entry in self.config.datasets:
            if not is_path_dataset(entry.dataset):
                continue
            try:
                current = self.profile(entry.dataset)
            except (ConfigError, FileNotFoundError):
                continue
            previous = self.store.latest_profile(entry.dataset)
            out[entry.dataset] = drift(previous, current) if previous else []
            if save:
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
        capabilities = self.capabilities()
        for dataset in self.graph().datasets:
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
        files: dict[DatasetId, Sequence[str]] = {}
        for dataset in self.graph().datasets:
            if not is_path_dataset(dataset):
                continue
            try:
                adapter, _ = self.adapter_for(dataset)
            except ConfigError:
                continue
            if hasattr(adapter, "files_for"):
                files[dataset] = adapter.files_for(dataset)
        return plan_erasure(self.graph(), request, capabilities=capabilities, files=files)

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
