"""Command line interface.

Everything here is read-only. `plan` prints what it would rebuild and exits; there
is deliberately no `apply` yet. Shadow mode comes first — run alongside your existing
full rebuild, compare, and only then trust the planner to skip work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .adapters import LocalStorage, registered
from .grains import Grain
from .graph import Edge, Graph
from .ids import normalize, normalize_table
from .lineage import extract
from .partitions import PartitionMapping
from .paths import PathTemplate
from .profile import Profile, drift, summarize
from .types import DatasetId, KeyPredicate, PartitionField, PartitionSpec

SPEC_HELP = (
    "Partition spec as TABLE:FIELD[:GRAIN], repeatable. "
    "A grain (hour|day|month|year) makes it a time field; without one it is a value field."
)


def _parse_specs(
    entries: tuple[str, ...], system: str, instance: str | None
) -> dict[DatasetId, PartitionSpec]:
    """Turn repeated --spec flags into partition specs keyed by dataset."""
    fields: dict[DatasetId, list[PartitionField]] = {}
    for entry in entries:
        parts = entry.split(":")
        if len(parts) not in (2, 3):
            raise click.BadParameter(f"{entry!r} is not TABLE:FIELD[:GRAIN]")
        table, name = parts[0], parts[1]
        ds = normalize_table(table, system=system, instance=instance)
        field = (
            PartitionField.time(name, Grain.parse(parts[2]))
            if len(parts) == 3
            else PartitionField.value(name)
        )
        fields.setdefault(ds, []).append(field)
    return {ds: PartitionSpec.of(*fs) for ds, fs in fields.items()}


def _parse_dirty(
    entries: tuple[str, ...],
    system: str,
    instance: str | None,
    specs: dict[DatasetId, PartitionSpec],
) -> dict[DatasetId, list[KeyPredicate]]:
    """Turn repeated --dirty flags into seed partitions."""
    from datetime import datetime

    seeds: dict[DatasetId, list[KeyPredicate]] = {}
    for entry in entries:
        table, _, binding = entry.partition("@")
        ds = normalize_table(table, system=system, instance=instance)
        spec = specs.get(ds, PartitionSpec())
        pairs: list[tuple[str, object]] = []
        for chunk in filter(None, binding.split(",")):
            key, _, raw = chunk.partition("=")
            f = spec.field(key)
            if f is not None and f.kind == "time":
                assert f.grain is not None
                try:
                    pairs.append((key, datetime.fromisoformat(raw)))
                except ValueError as exc:
                    raise click.BadParameter(f"{raw!r} is not an ISO datetime") from exc
            else:
                pairs.append((key, raw))
        seeds.setdefault(ds, []).append(KeyPredicate(bindings=tuple(pairs)))
    return seeds


def _build_graph(
    sql_files: tuple[str, ...],
    dialect: str,
    system: str,
    instance: str | None,
    specs: dict[DatasetId, PartitionSpec],
) -> Graph:
    graph = Graph()
    for ds, spec in specs.items():
        graph.add_dataset(ds, spec)
    for path in sql_files:
        text = Path(path).read_text()
        for x in extract(text, dialect=dialect, system=system, instance=instance, specs=specs):
            if x.target is None:
                for note in x.notes:
                    click.echo(f"  ! {Path(path).name}: {note}", err=True)
                continue
            for src in x.sources:
                graph.add_edge(
                    Edge(
                        src=src,
                        dst=x.target,
                        mapping=x.mappings.get(
                            src, PartitionMapping.unknown(specs.get(x.target, PartitionSpec()))
                        ),
                        columns=x.column_edges.get(src, ()),
                        evidence=f"sql:{Path(path).name}",
                    )
                )
    return graph


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="fathom")
def main() -> None:
    """Lineage, partition-scoped invalidation, and profiling for data platforms."""


@main.command()
def adapters() -> None:
    """List registered adapters."""
    for name in registered():
        click.echo(name)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--spec", "specs", multiple=True, help="FIELD[:GRAIN], repeatable")
@click.option(
    "--template", help="Path template for non-Hive layouts, e.g. 'events/{yyyy}/{MM}/{dd}'"
)
@click.option("--compare", type=click.Path(exists=True), help="Second path to diff against")
def profile(path: str, specs: tuple[str, ...], template: str | None, compare: str | None) -> None:
    """Profile Parquet data from footers alone. Reads no data pages."""
    fields = []
    for entry in specs:
        name, _, grain = entry.partition(":")
        fields.append(
            PartitionField.time(name, Grain.parse(grain)) if grain else PartitionField.value(name)
        )
    spec = PartitionSpec.of(*fields)

    store = LocalStorage()

    def snapshot(target: str) -> Profile:
        ds = normalize(str(Path(target).resolve()))
        store.declare(ds, spec, PathTemplate(template) if template else None)
        return store.profile(ds)

    got = snapshot(path)

    if compare:
        findings = drift(got, snapshot(compare))
        click.echo(summarize(findings))
        sys.exit(1 if any(f.severity.value == "error" for f in findings) else 0)

    click.echo(f"{got.dataset}")
    click.echo(f"  {got.row_count:,} rows across {got.file_count} file(s), source={got.source}")
    for col in got.columns:
        rate = f"{col.null_rate:.1%}" if col.null_rate is not None else "unknown"
        rng = f"{col.min}..{col.max}" if col.min is not None else "no stats"
        click.echo(f"  {col.name:<24} {col.dtype:<12} nulls={rate:<8} {rng}")


@main.command()
@click.argument("sql_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--dialect", default="duckdb", show_default=True)
@click.option("--system", help="Identity system; defaults to the dialect")
@click.option("--instance", help="Account or workspace, e.g. a Snowflake account locator")
@click.option("--spec", "specs", multiple=True, help=SPEC_HELP)
def lineage(
    sql_files: tuple[str, ...],
    dialect: str,
    system: str | None,
    instance: str | None,
    specs: tuple[str, ...],
) -> None:
    """Extract lineage from SQL files."""
    system = system or dialect
    parsed = _parse_specs(specs, system, instance)
    graph = _build_graph(sql_files, dialect, system, instance, parsed)
    if not graph.edges:
        click.echo("no lineage extracted", err=True)
        sys.exit(1)
    for edge in graph.edges:
        click.echo(str(edge))
        for src_col, dst_col in edge.columns:
            click.echo(f"    {src_col} -> {dst_col}")


@main.command()
@click.argument("sql_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--dialect", default="duckdb", show_default=True)
@click.option("--system", help="Identity system; defaults to the dialect")
@click.option("--instance", help="Account or workspace")
@click.option("--spec", "specs", multiple=True, help=SPEC_HELP)
@click.option("--dirty", "dirties", multiple=True, required=True, help="TABLE@FIELD=VALUE,...")
def plan(
    sql_files: tuple[str, ...],
    dialect: str,
    system: str | None,
    instance: str | None,
    specs: tuple[str, ...],
    dirties: tuple[str, ...],
) -> None:
    """Show which partitions a set of source changes invalidates.

    Prints a plan and stops. Compare it against a full rebuild before trusting it.
    """
    system = system or dialect
    parsed = _parse_specs(specs, system, instance)
    graph = _build_graph(sql_files, dialect, system, instance, parsed)
    seeds = _parse_dirty(dirties, system, instance, parsed)

    result = graph.invalidate(seeds)
    if result.is_empty:
        click.echo("nothing to rebuild")
        return

    click.echo(result.summary())
    if result.widened:
        click.echo("")
        click.echo("widened to whole dataset (no provable partition bound):", err=True)
        for ds in sorted(result.widened, key=str):
            for reason in result.reasons[ds][:2]:
                click.echo(f"  {ds}: {reason}", err=True)
    if result.cyclic:
        click.echo("")
        click.echo(
            f"cycles detected in: {', '.join(str(d) for d in sorted(result.cyclic, key=str))}",
            err=True,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
