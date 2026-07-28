"""Command line interface.

Read-only by default and deliberately so. `plan` prints what it would rebuild;
`erase` prints what it would destroy. Neither has an `--execute` flag, because
executing needs a live engine binding and that belongs in a pipeline, not in a
shell one-liner where a typo costs you a table.

Shadow mode comes before trust: run `plan` alongside your existing full rebuild,
compare, and only then wire the library into anything that writes.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from . import __version__
from .adapters import DeltaCatalog, LocalStorage, registered
from .erasure import ErasureRequest, apply_erasure, plan_erasure
from .grains import Grain
from .graph import Graph
from .ids import normalize, normalize_table
from .ingest import graph_from_queries
from .paths import PathTemplate
from .policy import SinkPolicy, enforce, infer, propagate
from .profile import Profile, drift, summarize
from .store import Store
from .types import (
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    DatasetId,
    ErasureMode,
    KeyPredicate,
    LineageSource,
    PartitionField,
    PartitionSpec,
)

SPEC_HELP = (
    "Partition spec as TABLE:FIELD[:GRAIN], repeatable. "
    "A grain (hour|day|month|year) makes it a time field; without one it is a value field."
)
DIRTY_HELP = "Changed source partitions as TABLE@FIELD=VALUE[,FIELD=VALUE], repeatable."
DEFAULT_STORE = ".fathom/fathom.db"


# -- shared parsing ------------------------------------------------------------


def _field(name: str, grain: str | None) -> PartitionField:
    return PartitionField.time(name, Grain.parse(grain)) if grain else PartitionField.value(name)


def _parse_specs(
    entries: tuple[str, ...], system: str, instance: str | None
) -> dict[DatasetId, PartitionSpec]:
    fields: dict[DatasetId, list[PartitionField]] = {}
    for entry in entries:
        parts = entry.split(":")
        if len(parts) not in (2, 3):
            raise click.BadParameter(f"{entry!r} is not TABLE:FIELD[:GRAIN]")
        ds = normalize_table(parts[0], system=system, instance=instance)
        fields.setdefault(ds, []).append(_field(parts[1], parts[2] if len(parts) == 3 else None))
    return {ds: PartitionSpec.of(*fs) for ds, fs in fields.items()}


def _parse_bindings(binding: str, spec: PartitionSpec) -> KeyPredicate:
    pairs: list[tuple[str, object]] = []
    for chunk in filter(None, binding.split(",")):
        key, _, raw = chunk.partition("=")
        field = spec.field(key)
        if field is not None and field.kind == "time":
            try:
                pairs.append((key, datetime.fromisoformat(raw)))
            except ValueError as exc:
                raise click.BadParameter(f"{raw!r} is not an ISO datetime") from exc
        else:
            pairs.append((key, raw))
    return KeyPredicate(bindings=tuple(pairs))


def _parse_dirty(
    entries: tuple[str, ...],
    system: str,
    instance: str | None,
    specs: dict[DatasetId, PartitionSpec],
) -> dict[DatasetId, list[KeyPredicate]]:
    seeds: dict[DatasetId, list[KeyPredicate]] = {}
    for entry in entries:
        table, _, binding = entry.partition("@")
        ds = normalize_table(table, system=system, instance=instance)
        seeds.setdefault(ds, []).append(_parse_bindings(binding, specs.get(ds, UNPARTITIONED)))
    return seeds


def _spec_from_flags(entries: tuple[str, ...]) -> PartitionSpec:
    """A bare FIELD[:GRAIN] spec, for commands addressing a single dataset."""
    fields = []
    for entry in entries:
        name, _, grain = entry.partition(":")
        fields.append(_field(name, grain or None))
    return PartitionSpec.of(*fields)


def _dataset_arg(value: str, system: str, instance: str | None) -> DatasetId:
    """Resolve a CLI argument naming a dataset, whether path or table.

    Paths get resolved first. On macOS `/tmp` is a symlink to `/private/tmp`, so a
    sink named one way silently fails to match a dataset profiled the other way.
    """
    candidate = Path(value)
    if value.startswith((".", "/")) or candidate.exists():
        return normalize(str(candidate.resolve()))
    return normalize(value, system=system, instance=instance)


def _open_store(ctx: click.Context) -> Store:
    path = Path(ctx.obj["store"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return Store(path)


def _require_graph(store: Store) -> Graph:
    graph = store.load_graph()
    if not graph.edges:
        raise click.ClickException("no lineage in the store; run `fathom ingest` first")
    return graph


# -- group ---------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="fathom")
@click.option(
    "--store",
    default=DEFAULT_STORE,
    show_default=True,
    envvar="FATHOM_STORE",
    help="Where the graph and profile history live.",
)
@click.pass_context
def main(ctx: click.Context, store: str) -> None:
    """Lineage, partition-scoped invalidation, profiling, and policy for data platforms."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = store


@main.command()
def adapters() -> None:
    """List registered adapters."""
    for name in registered():
        click.echo(name)


# -- graph ---------------------------------------------------------------------


@main.command()
@click.argument("sql_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--dialect", default="duckdb", show_default=True)
@click.option("--system", help="Identity system; defaults to the dialect.")
@click.option("--instance", help="Account or workspace, e.g. a Snowflake account locator.")
@click.option("--spec", "specs", multiple=True, help=SPEC_HELP)
@click.pass_context
def ingest(
    ctx: click.Context,
    sql_files: tuple[str, ...],
    dialect: str,
    system: str | None,
    instance: str | None,
    specs: tuple[str, ...],
) -> None:
    """Parse SQL into a dependency graph and persist it."""
    from .adapters.base import QueryEvent

    system = system or dialect
    parsed = _parse_specs(specs, system, instance)
    queries = [
        QueryEvent(sql=Path(p).read_text(), dialect=dialect, query_id=Path(p).name)
        for p in sql_files
    ]
    result = graph_from_queries(
        queries, dialect=dialect, system=system, instance=instance, specs=parsed
    )

    with _open_store(ctx) as store:
        store.save_graph(result.graph)

    click.echo(result.summary())
    for note in result.notes:
        click.echo(f"  ! {note}", err=True)
    if not result.edges:
        raise click.ClickException("no lineage extracted")


@main.command()
@click.pass_context
def lineage(ctx: click.Context) -> None:
    """Show the stored dependency graph."""
    with _open_store(ctx) as store:
        graph = _require_graph(store)
        for edge in graph.edges:
            click.echo(str(edge))
            for src_col, dst_col in edge.columns:
                click.echo(f"    {src_col} -> {dst_col}")


@main.command()
@click.option("--dirty", "dirties", multiple=True, required=True, help=DIRTY_HELP)
@click.option("--system", default="duckdb", show_default=True)
@click.option("--instance")
@click.pass_context
def plan(ctx: click.Context, dirties: tuple[str, ...], system: str, instance: str | None) -> None:
    """Show which partitions a set of source changes invalidates."""
    with _open_store(ctx) as store:
        graph = _require_graph(store)
        specs = {ds: graph.spec(ds) for ds in graph.datasets}
        seeds = _parse_dirty(dirties, system, instance, specs)

        # A seeded name the graph has never heard of is almost always a typo, and it
        # would otherwise produce a confident-looking plan containing only itself.
        unknown = [ds for ds in seeds if ds not in set(graph.datasets)]
        for ds in unknown:
            click.echo(f"  ! {ds} is not in the graph; check the table name", err=True)

        result = graph.invalidate(seeds)
        if result.is_empty:
            click.echo("nothing to rebuild")
            return

        click.echo(result.summary())
        if result.widened:
            click.echo("", err=True)
            click.echo("widened to whole dataset (no provable partition bound):", err=True)
            for ds in sorted(result.widened, key=str):
                for reason in result.reasons[ds][:2]:
                    click.echo(f"  {ds}: {reason}", err=True)
        if result.cyclic:
            cycles = ", ".join(str(d) for d in sorted(result.cyclic, key=str))
            click.echo(f"\ncycles detected in: {cycles}", err=True)


# -- change detection ----------------------------------------------------------


def _pick_adapter(dataset: DatasetId, catalog: str) -> tuple[Any, str]:
    """Resolve which adapter reads this dataset, sniffing the layout when asked to."""
    if catalog == "delta":
        return DeltaCatalog(), "delta"
    if catalog == "iceberg":
        from .adapters.iceberg import IcebergCatalog

        return IcebergCatalog(), "iceberg"
    if catalog == "local":
        return LocalStorage(), "local"

    delta = DeltaCatalog()
    if delta.is_delta_table(dataset):
        return delta, "delta"
    try:
        from .adapters.iceberg import IcebergCatalog

        iceberg = IcebergCatalog()
        if iceberg.is_iceberg_table(dataset):
            return iceberg, "iceberg"
    except ImportError:  # pragma: no cover - only without the extra
        pass
    return LocalStorage(), "local"


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--catalog",
    type=click.Choice(["auto", "delta", "iceberg", "local"]),
    default="auto",
    show_default=True,
)
@click.option("--spec", "specs", multiple=True, help="FIELD[:GRAIN], repeatable.")
@click.option("--template", help="Path template for non-Hive layouts.")
@click.option("--reset", is_flag=True, help="Ignore the stored token and report everything.")
@click.pass_context
def changed(
    ctx: click.Context,
    path: str,
    catalog: str,
    specs: tuple[str, ...],
    template: str | None,
    reset: bool,
) -> None:
    """Detect which partitions changed since the last run."""
    dataset = normalize(str(Path(path).resolve()))
    spec = _spec_from_flags(specs)
    adapter, resolved = _pick_adapter(dataset, catalog)

    if resolved == "local":
        adapter.declare(dataset, spec, PathTemplate(template) if template else None)
    elif spec.fields:
        adapter.declare(dataset, spec)

    with _open_store(ctx) as store:
        token = None if reset else store.get_token(dataset, resolved)
        result = adapter.changed(dataset, token)
        store.set_token(dataset, resolved, result.token)

    click.echo(f"{dataset}  [{resolved}]  token={result.token}")
    if not result.complete:
        click.echo(
            "  ! the source could not enumerate exhaustively; treat this as widened", err=True
        )
    if result.is_empty:
        click.echo("  no changes")
        return
    for key in sorted(result.partitions, key=str):
        click.echo(f"  {key}")


# -- profiling -----------------------------------------------------------------


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--spec", "specs", multiple=True, help="FIELD[:GRAIN], repeatable.")
@click.option("--template", help="Path template for non-Hive layouts.")
@click.option("--save/--no-save", default=False, help="Record the profile in the store.")
@click.option("--compare", type=click.Path(exists=True), help="Diff against a second path.")
@click.pass_context
def profile(
    ctx: click.Context,
    path: str,
    specs: tuple[str, ...],
    template: str | None,
    save: bool,
    compare: str | None,
) -> None:
    """Profile Parquet data from footers alone. Reads no data pages."""
    spec = _spec_from_flags(specs)
    adapter = LocalStorage()

    def snapshot(target: str) -> Profile:
        ds = normalize(str(Path(target).resolve()))
        adapter.declare(ds, spec, PathTemplate(template) if template else None)
        return adapter.profile(ds)

    got = snapshot(path)

    if save:
        with _open_store(ctx) as store:
            store.save_profile(got)

    if compare:
        findings = drift(got, snapshot(compare))
        click.echo(summarize(findings))
        sys.exit(1 if any(f.severity == "error" for f in findings) else 0)

    click.echo(f"{got.dataset}")
    click.echo(f"  {got.row_count:,} rows across {got.file_count} file(s), source={got.source}")
    for col in got.columns:
        rate = f"{col.null_rate:.1%}" if col.null_rate is not None else "unknown"
        rng = f"{col.min}..{col.max}" if col.min is not None else "no stats"
        click.echo(f"  {col.name:<24} {col.dtype:<12} nulls={rate:<8} {rng}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--spec", "specs", multiple=True, help="FIELD[:GRAIN], repeatable.")
@click.option("--template", help="Path template for non-Hive layouts.")
@click.pass_context
def check(ctx: click.Context, path: str, specs: tuple[str, ...], template: str | None) -> None:
    """Compare a dataset against its last recorded profile, and attribute any drift."""
    from .types import ColumnRef

    dataset = normalize(str(Path(path).resolve()))
    adapter = LocalStorage()
    adapter.declare(dataset, _spec_from_flags(specs), PathTemplate(template) if template else None)
    current = adapter.profile(dataset)

    with _open_store(ctx) as store:
        previous = store.latest_profile(dataset)
        if previous is None:
            store.save_profile(current)
            click.echo(f"{dataset}: baseline recorded ({current.row_count:,} rows)")
            return

        findings = drift(previous, current)
        store.save_profile(current)
        click.echo(summarize(findings))

        # Turn each finding into a diagnosis rather than an alert, where we can.
        graph = store.load_graph()
        if findings and graph.edges:
            click.echo("")
            for finding in findings:
                if not finding.column:
                    continue
                paths = graph.upstream_columns(ColumnRef(dataset, finding.column), max_depth=3)
                if paths:
                    trail = " <- ".join(str(step) for step in paths[0])
                    click.echo(f"  {finding.column} derives from: {trail}")

    sys.exit(1 if any(f.severity == "error" for f in findings) else 0)


# -- labels --------------------------------------------------------------------


@main.command()
@click.option("--sink", "sinks", multiple=True, help="TABLE:forbid=LABEL[,LABEL], repeatable.")
@click.option("--system", default="duckdb", show_default=True)
@click.option("--instance")
@click.option("--min-confidence", default=0.5, show_default=True, type=float)
@click.pass_context
def label(
    ctx: click.Context,
    sinks: tuple[str, ...],
    system: str,
    instance: str | None,
    min_confidence: float,
) -> None:
    """Infer column labels from stored profiles, propagate them, and check sink policies."""
    with _open_store(ctx) as store:
        graph = store.load_graph()
        seeds: dict[Any, Any] = {}
        for dataset in store.datasets():
            latest = store.latest_profile(dataset)
            if latest is not None:
                seeds.update(infer(latest))

        if not seeds:
            raise click.ClickException(
                "no labels inferred; run `fathom profile --save` on some datasets first"
            )

        labels = propagate(graph, seeds) if graph.edges else seeds

        for ref, values in sorted(
            labels.items(), key=lambda kv: (str(kv[0].dataset), kv[0].column)
        ):
            for entry in sorted(values):
                store.set_label(
                    ref.dataset,
                    ref.column,
                    entry.name,
                    confidence=entry.confidence,
                    origin=entry.origin,
                    confirmed=entry.confirmed,
                )
                mark = "*" if entry.confirmed else " "
                click.echo(
                    f"{mark} {ref.dataset} {ref.column:<24} {entry.name:<18} "
                    f"{entry.confidence:.0%}  {entry.origin}"
                )

        policies = []
        for entry in sinks:
            table, _, rule = entry.partition(":")
            _, _, forbidden = rule.partition("=")
            policies.append(
                SinkPolicy(
                    dataset=_dataset_arg(table, system, instance),
                    forbid=frozenset(filter(None, forbidden.split(","))),
                    reason="declared on the command line",
                )
            )

        if policies:
            report = enforce(labels, policies, min_confidence=min_confidence)
            click.echo("")
            click.echo(report.summary())
            if not report.ok:
                sys.exit(1)


# -- erasure -------------------------------------------------------------------


@main.command()
@click.option("--subject", required=True, help="The subject identifier value.")
@click.option("--key-column", required=True, help="Column holding the subject identifier.")
@click.option("--origin", required=True, help="Dataset where the subject's rows live.")
@click.option("--partition", "partitions", multiple=True, help="FIELD=VALUE[,FIELD=VALUE].")
@click.option("--worm", multiple=True, help="Dataset that cannot be deleted from.")
@click.option("--reference", default="", help="Your request ticket id, recorded in the proof.")
@click.option("--salt", default="", envvar="FATHOM_SALT", help="Secret salt for the subject hash.")
@click.option("--system", default="duckdb", show_default=True)
@click.option("--instance")
@click.option("--proof", type=click.Path(), help="Write the proof artifact here.")
@click.pass_context
def erase(
    ctx: click.Context,
    subject: str,
    key_column: str,
    origin: str,
    partitions: tuple[str, ...],
    worm: tuple[str, ...],
    reference: str,
    salt: str,
    system: str,
    instance: str | None,
    proof: str | None,
) -> None:
    """Plan an erasure: locate a subject's data everywhere it flowed.

    Always a dry run. Executing needs a live engine binding, which belongs in a
    pipeline rather than a shell command.
    """
    with _open_store(ctx) as store:
        graph = _require_graph(store)
        origin_ds = _dataset_arg(origin, system, instance)
        keys = frozenset(_parse_bindings(p, graph.spec(origin_ds)) for p in partitions)

        blocked = {_dataset_arg(w, system, instance) for w in worm}
        capabilities = {
            ds: Capabilities(
                lineage=LineageSource.DECLARED,
                change=ChangeSource.SNAPSHOT_DIFF,
                erasure=ErasureMode.NONE if ds in blocked else ErasureMode.REWRITE,
            )
            for ds in graph.datasets
        }

        request = ErasureRequest(
            subject=subject,
            key_column=key_column,
            origin=origin_ds,
            partitions=keys,
            reference=reference,
        )
        result = plan_erasure(graph, request, capabilities=capabilities)
        click.echo(result.summary())

        artifact = apply_erasure(result, {}, dry_run=True, salt=salt)
        if proof:
            Path(proof).write_text(artifact.to_json())
            click.echo(f"\nproof written to {proof} (digest {artifact.digest[:12]}…)")

        if not result.is_complete:
            sys.exit(1)


# -- shadow --------------------------------------------------------------------


@main.command()
@click.pass_context
def shadow(ctx: click.Context) -> None:
    """Report accumulated shadow results: how much was skipped, and what was missed."""
    with _open_store(ctx) as store:
        summary = store.shadow_summary()
        if not summary["runs"]:
            raise click.ClickException("no shadow observations recorded yet")

        click.echo(f"runs        {summary['runs']}")
        click.echo(f"partitions  {summary['planned']} planned of {summary['total']} total")
        click.echo(f"savings     {summary['savings']:.0%}")
        click.echo(f"missed      {summary['missed']}")
        if summary["missed"]:
            click.echo("")
            click.echo(
                "MISSED PARTITIONS ARE A SOUNDNESS FAILURE. The planner called them "
                "clean and a full rebuild proved otherwise. Do not enable apply mode.",
                err=True,
            )
            sys.exit(1)
        click.echo("\nno missed partitions across every run recorded here")


if __name__ == "__main__":  # pragma: no cover
    main()
