"""Command line interface.

Every command goes through `Project`, so the CLI and the Python API cannot drift
apart. Configuration lives in `fathom.yml`; flags exist for one-off overrides, not
as the primary way to describe a project — partition specs passed as flags on every
invocation are partition specs that drift.

Read-only by default and deliberately so. `plan` prints what it would rebuild;
`erase` prints what it would destroy. Neither has an `--execute` flag, because
executing needs a live engine binding and that belongs in a pipeline, not in a
shell one-liner where a typo costs you a table.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from . import __version__
from .adapters import registered
from .config import CONFIG_NAMES, load_config
from .erasure import ErasureRequest, apply_erasure
from .errors import ConfigError
from .profile import summarize
from .project import Project
from .store import Store
from .types import UNPARTITIONED, DatasetId, KeyPredicate

STARTER_CONFIG = """\
# fathom project configuration.
# Partition specs live here because they cannot be reliably inferred: Snowflake has
# no partitions to read, and Delta records column names but not grain.

version: 1
store: .fathom/fathom.db

system: duckdb          # default identity system for bare table names
# instance: xy12345     # Snowflake account or Databricks workspace

datasets:
  - name: raw.events
    partition:
      - {field: dt, grain: day}
      - {field: region}

  - name: gold.monthly
    model: models/gold_monthly.sql
    partition:
      - {field: dt, grain: month}
      - {field: region}

lineage:
  - type: sql
    paths: ["models/*.sql"]
    dialect: duckdb

# policies:
#   - dataset: ml.training_set
#     forbid: [pii]
#     reason: not cleared for personal data

# storage_options:
#   s3: {key: "${AWS_ACCESS_KEY_ID}", secret: "${AWS_SECRET_ACCESS_KEY}"}
"""


def _project(ctx: click.Context) -> Project:
    """Open the project named by --config, or found by searching upward."""
    try:
        config = load_config(ctx.obj.get("config"))
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    override = ctx.obj.get("store")
    store_path = Path(override) if override else config.store
    store_path.parent.mkdir(parents=True, exist_ok=True)
    return Project(config=config, store=Store(store_path))


def _parse_bindings(binding: str, spec: Any) -> KeyPredicate:
    pairs: list[tuple[str, object]] = []
    for chunk in filter(None, binding.split(",")):
        key, _, raw = chunk.partition("=")
        field = spec.field(key) if spec is not None else None
        if field is not None and field.kind == "time":
            try:
                pairs.append((key, datetime.fromisoformat(raw)))
            except ValueError as exc:
                raise click.BadParameter(f"{raw!r} is not an ISO datetime") from exc
        else:
            pairs.append((key, raw))
    return KeyPredicate(bindings=tuple(pairs))


def _seeds(project: Project, dirties: tuple[str, ...]) -> dict[DatasetId, list[KeyPredicate]]:
    graph = project.graph()
    known = set(graph.datasets)
    out: dict[DatasetId, list[KeyPredicate]] = {}
    for entry in dirties:
        name, _, binding = entry.partition("@")
        dataset = project.config.resolve(name)
        if dataset not in known:
            # Almost always a typo, and otherwise it produces a confident-looking
            # plan containing only the misspelled name.
            click.echo(f"  ! {dataset} is not in the graph; check the name", err=True)
        out.setdefault(dataset, []).append(
            _parse_bindings(binding, graph.spec(dataset) or UNPARTITIONED)
        )
    return out


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="fathom")
@click.option("--config", type=click.Path(), envvar="FATHOM_CONFIG", help="Path to fathom.yml.")
@click.option("--store", type=click.Path(), envvar="FATHOM_STORE", help="Override the store path.")
@click.pass_context
def main(ctx: click.Context, config: str | None, store: str | None) -> None:
    """Lineage, partition-scoped invalidation, profiling, and policy for data platforms."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["store"] = store


# -- setup ---------------------------------------------------------------------


@main.command()
@click.option("--force", is_flag=True, help="Overwrite an existing config.")
def init(force: bool) -> None:
    """Write a starter fathom.yml in the current directory."""
    target = Path(CONFIG_NAMES[0])
    if target.exists() and not force:
        raise click.ClickException(f"{target} already exists; pass --force to overwrite")
    target.write_text(STARTER_CONFIG)
    click.echo(f"wrote {target}")
    click.echo("Next: edit the datasets block, then run `fathom ingest`.")


@main.command()
def adapters() -> None:
    """List registered adapters."""
    for name in registered():
        click.echo(name)


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Report configuration problems that would silently degrade a plan."""
    with _project(ctx) as project:
        click.echo(f"config   {project.config.path}")
        click.echo(f"store    {project.store.path}")
        click.echo(f"system   {project.config.system}")
        click.echo(f"datasets {len(project.config.datasets)}, edges {len(project.graph().edges)}")

        problems = project.doctor()
        if not problems:
            click.echo("\nno problems found")
            return
        click.echo(f"\n{len(problems)} problem(s):")
        for problem in problems:
            click.echo(f"  {problem}")


# -- graph ---------------------------------------------------------------------


@main.command()
@click.pass_context
def ingest(ctx: click.Context) -> None:
    """Build the dependency graph from every configured lineage source."""
    with _project(ctx) as project:
        result = project.ingest()
        click.echo(result.summary())
        for note in result.notes[:20]:
            click.echo(f"  ! {note}", err=True)
        if len(result.notes) > 20:
            click.echo(f"  ! ... and {len(result.notes) - 20} more", err=True)
        if not result.edges:
            raise click.ClickException(
                "no lineage extracted; check the `lineage` block in your config"
            )


@main.command()
@click.pass_context
def lineage(ctx: click.Context) -> None:
    """Show the stored dependency graph."""
    with _project(ctx) as project:
        graph = project.graph()
        if not graph.edges:
            raise click.ClickException("no lineage in the store; run `fathom ingest` first")
        for edge in graph.edges:
            click.echo(str(edge))
            for src_col, dst_col in edge.columns:
                click.echo(f"    {src_col} -> {dst_col}")


@main.command()
@click.pass_context
def detect(ctx: click.Context) -> None:
    """Ask every configured source what changed since the last run."""
    with _project(ctx) as project:
        if not project.config.sources:
            raise click.ClickException("no sources configured; add them under `datasets`")
        for dataset, changes in project.detect().items():
            click.echo(f"{dataset}  token={changes.token or '-'}")
            if not changes.complete:
                click.echo(
                    "  ! the source could not enumerate exhaustively; treat as widened",
                    err=True,
                )
            if changes.is_empty:
                click.echo("  no changes")
            for key in sorted(changes.partitions, key=str):
                click.echo(f"  {key}")


@main.command()
@click.option("--dirty", "dirties", multiple=True, help="TABLE@FIELD=VALUE[,FIELD=VALUE].")
@click.option("--detect", "auto", is_flag=True, help="Discover the seeds by scanning sources.")
@click.pass_context
def plan(ctx: click.Context, dirties: tuple[str, ...], auto: bool) -> None:
    """Show which partitions a set of source changes invalidates."""
    if not dirties and not auto:
        raise click.UsageError("pass --dirty, or --detect to scan sources first")

    with _project(ctx) as project:
        result = project.plan(detect=True) if auto else project.plan(_seeds(project, dirties))
        if result.is_empty:
            click.echo("nothing to rebuild")
            return

        click.echo(result.summary())
        if result.widened:
            click.echo("\nwidened to whole dataset (no provable partition bound):", err=True)
            for ds in sorted(result.widened, key=str):
                for reason in result.reasons[ds][:2]:
                    click.echo(f"  {ds}: {reason}", err=True)
        if result.cyclic:
            cycles = ", ".join(str(d) for d in sorted(result.cyclic, key=str))
            click.echo(f"\ncycles detected in: {cycles}", err=True)


# -- profiling -----------------------------------------------------------------


@main.command()
@click.argument("dataset", required=False)
@click.pass_context
def profile(ctx: click.Context, dataset: str | None) -> None:
    """Profile datasets from Parquet footers alone. Reads no data pages."""
    with _project(ctx) as project:
        targets = (
            [project.config.resolve(dataset)]
            if dataset
            else [d.dataset for d in project.config.datasets]
        )
        shown = 0
        for target in targets:
            try:
                got = project.profile(target)
            except (ConfigError, FileNotFoundError, ValueError) as exc:
                if dataset:
                    raise click.ClickException(str(exc)) from exc
                continue
            shown += 1
            project.store.save_profile(got)
            click.echo(f"{got.dataset}")
            click.echo(f"  {got.row_count:,} rows across {got.file_count} file(s)")
            for col in got.columns:
                rate = f"{col.null_rate:.1%}" if col.null_rate is not None else "unknown"
                rng = f"{col.min}..{col.max}" if col.min is not None else "no stats"
                click.echo(f"  {col.name:<24} {col.dtype:<12} nulls={rate:<8} {rng}")
        if not shown:
            raise click.ClickException("no profilable datasets; only path-backed ones qualify")


@main.command()
@click.pass_context
def check(ctx: click.Context) -> None:
    """Compare each dataset against its last profile, and attribute any drift."""
    from .types import ColumnRef

    with _project(ctx) as project:
        results = project.check()
        if not results:
            raise click.ClickException("no path-backed datasets to check")

        graph = project.graph()
        failed = False
        for dataset, findings in results.items():
            if not findings:
                click.echo(f"{dataset}: no drift")
                continue
            click.echo(f"{dataset}:")
            click.echo("  " + summarize(findings).replace("\n", "\n  "))
            failed = failed or any(f.severity == "error" for f in findings)

            for finding in findings:
                if not finding.column or not graph.edges:
                    continue
                paths = graph.upstream_columns(ColumnRef(dataset, finding.column), max_depth=3)
                if paths:
                    trail = " <- ".join(str(step) for step in paths[0])
                    click.echo(f"    {finding.column} derives from: {trail}")

        if failed:
            sys.exit(1)


# -- labels --------------------------------------------------------------------


@main.command()
@click.pass_context
def label(ctx: click.Context) -> None:
    """Infer column labels, propagate them, and check configured sink policies."""
    with _project(ctx) as project:
        labels = project.labels()
        if not labels:
            raise click.ClickException(
                "no labels inferred; run `fathom profile` on some datasets first"
            )

        for ref, values in sorted(
            labels.items(), key=lambda kv: (str(kv[0].dataset), kv[0].column)
        ):
            for entry in sorted(values):
                mark = "*" if entry.confirmed else " "
                click.echo(
                    f"{mark} {ref.dataset} {ref.column:<24} {entry.name:<18} "
                    f"{entry.confidence:.0%}  {entry.origin}"
                )

        if project.config.policies:
            report = project.enforce(labels)
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
@click.option("--reference", default="", help="Your request ticket id, recorded in the proof.")
@click.option("--salt", default="", envvar="FATHOM_SALT", help="Secret salt for the subject hash.")
@click.option("--proof", type=click.Path(), help="Write the proof artifact here.")
@click.pass_context
def erase(
    ctx: click.Context,
    subject: str,
    key_column: str,
    origin: str,
    partitions: tuple[str, ...],
    reference: str,
    salt: str,
    proof: str | None,
) -> None:
    """Plan an erasure: locate a subject's data everywhere it flowed.

    Always a dry run. Executing needs a live engine binding, which belongs in a
    pipeline rather than a shell command.
    """
    with _project(ctx) as project:
        graph = project.graph()
        if not graph.edges:
            raise click.ClickException("no lineage in the store; run `fathom ingest` first")

        origin_ds = project.config.resolve(origin)
        keys = frozenset(_parse_bindings(p, graph.spec(origin_ds)) for p in partitions)
        request = ErasureRequest(
            subject=subject,
            key_column=key_column,
            origin=origin_ds,
            partitions=keys,
            reference=reference,
        )

        result = project.locate(request)
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
    with _project(ctx) as project:
        summary = project.store.shadow_summary()
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
