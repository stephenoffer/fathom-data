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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click

from .. import __version__
from ..adapters import registered
from ..core.errors import ConfigError
from ..core.types import UNPARTITIONED, DatasetId, KeyPredicate
from ..core.util.clock import as_utc, now
from ..govern import contracts as contracts_mod
from ..govern import reidentification as reid
from ..govern.erasure import ErasureRequest, apply_erasure
from ..graph import query, selectors, sinks
from ..graph.plan import lifetime, schedule
from ..graph.plan.cost import CostModel
from ..graph.selectors import SelectorError
from ..observe import completeness as completeness_mod
from ..observe import seasonal as seasonal_mod
from ..observe import usage as usage_mod
from ..observe.profile import Severity, summarize
from ..report import orchestrators, render
from ..store.sqlite import Store
from .config import CONFIG_NAMES, load_config
from .project import Project

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


def _narrow(graph: Any, selector: str | None) -> Any:
    """Restrict a graph to a selector expression, or return it unchanged."""
    if not selector:
        return graph

    try:
        chosen = selectors.resolve(graph, selector)
    except SelectorError as exc:
        raise click.ClickException(f"{exc}") from exc
    return query.subgraph(graph, chosen)


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
@click.option("--author", default="", help="Recorded against this ingest's revision.")
@click.option("--note", default="", help="Why the graph changed, for the history.")
@click.pass_context
def ingest(ctx: click.Context, author: str, note: str) -> None:
    """Build the dependency graph from every configured lineage source."""
    with _project(ctx) as project:
        result = project.ingest(author=author, note=note)
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
@click.option("--select", "selector", help="Restrict to a selector, e.g. '+gold.monthly+'.")
@click.pass_context
def lineage(ctx: click.Context, selector: str | None) -> None:
    """Show the stored dependency graph."""
    with _project(ctx) as project:
        graph = project.graph()
        if not graph.edges:
            raise click.ClickException("no lineage in the store; run `fathom ingest` first")
        graph = _narrow(graph, selector)
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
@click.option("--json", "as_json", is_flag=True, help="Emit the plan as JSON for a pipeline.")
@click.pass_context
def plan(ctx: click.Context, dirties: tuple[str, ...], auto: bool, as_json: bool) -> None:
    """Show which partitions a set of source changes invalidates."""
    if not dirties and not auto:
        raise click.UsageError("pass --dirty, or --detect to scan sources first")

    with _project(ctx) as project:
        result = project.plan(detect=True) if auto else project.plan(_seeds(project, dirties))
        if as_json:
            # The plan is the one output another program acts on, and screen-scraping
            # a summary written for humans is how an orchestrator ends up rebuilding
            # the wrong thing after a wording change.
            click.echo(render.plan_to_json(result))
            return
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
@click.option("--json", "as_json", is_flag=True, help="Emit findings as JSON for a pipeline.")
@click.pass_context
def check(ctx: click.Context, as_json: bool) -> None:
    """Compare each dataset against its last profile, and attribute any drift."""
    import json as _json

    from ..core.types import ColumnRef

    with _project(ctx) as project:
        results = project.check()
        if not results:
            raise click.ClickException("no path-backed datasets to check")

        if as_json:
            payload = {
                str(dataset): [
                    {
                        "column": f.column,
                        "kind": f.kind,
                        "severity": f.severity.value,
                        "detail": f.detail,
                    }
                    for f in findings
                ]
                for dataset, findings in sorted(results.items(), key=lambda kv: str(kv[0]))
            }
            click.echo(_json.dumps(payload, indent=2, sort_keys=True))
            if any(f.severity is Severity.ERROR for fs in results.values() for f in fs):
                sys.exit(1)
            return

        graph = project.graph()
        failed = False
        for dataset, findings in results.items():
            if not findings:
                click.echo(f"{dataset}: no drift")
                continue
            click.echo(f"{dataset}:")
            click.echo("  " + summarize(findings).replace("\n", "\n  "))
            failed = failed or any(f.severity is Severity.ERROR for f in findings)

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

        if proof:
            if not salt:
                # Refuse rather than emit a proof whose subject digest can be
                # reversed. Identifiers are low-entropy, so an unsalted SHA-256
                # identifies the subject as well as the raw value does — and the
                # proof is the artifact that gets handed to people who must not
                # learn who they were.
                raise click.ClickException(
                    "writing a proof needs a secret salt: set FATHOM_SALT (or pass "
                    "--salt) to a per-organization secret. Without it the subject "
                    "digest in the proof is reversible by dictionary attack."
                )
            artifact = apply_erasure(result, {}, dry_run=True, salt=salt)
            Path(proof).write_text(artifact.to_json())
            click.echo(f"\nproof written to {proof} (digest {artifact.digest[:12]}…)")

        if not result.is_complete:
            sys.exit(1)


# -- shadow --------------------------------------------------------------------


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Emit the totals as JSON for a pipeline.")
@click.pass_context
def shadow(ctx: click.Context, as_json: bool) -> None:
    """Report accumulated shadow results: how much was skipped, and what was missed."""
    with _project(ctx) as project:
        summary = project.store.shadow_summary()
        if not summary["runs"]:
            raise click.ClickException("no shadow observations recorded yet")

        if as_json:
            import json as _json

            click.echo(_json.dumps(dict(summary), indent=2, sort_keys=True))
            if summary["missed"]:
                sys.exit(1)
            return

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


# -- what should be there, who reads it, what it costs --------------------------


@main.command()
@click.option("--dataset", required=True, help="Dataset to check, as named in fathom.yml.")
@click.option("--since", required=True, help="ISO date the expected range starts at.")
@click.option("--until", required=True, help="ISO date the expected range ends at, inclusive.")
@click.pass_context
def completeness(ctx: click.Context, dataset: str, since: str, until: str) -> None:
    """Report partitions that should exist and do not.

    Reads what is present from recorded arrivals rather than from a listing, so it
    still answers after a partition has been deleted.
    """
    with _project(ctx) as project:
        ds = project.config.resolve(dataset)
        spec = project.graph().spec(ds)
        if not spec.fields:
            raise click.ClickException(
                f"{ds} has no partition spec; declare one in fathom.yml before "
                "asking which of its partitions are missing"
            )
        present = project.store.present_partitions(ds)
        if not present:
            raise click.ClickException(
                f"no arrivals recorded for {ds}; nothing has been observed landing, so "
                "every expected partition would be reported missing"
            )
        try:
            result = completeness_mod.report(
                ds,
                spec,
                present,
                start=datetime.fromisoformat(since),
                end=datetime.fromisoformat(until),
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(result.summary())
        if not result.is_complete:
            sys.exit(1)


@main.command()
@click.option("--days", default=90, show_default=True, help="Window of read history to read.")
@click.option("--retire", is_flag=True, help="Show retirement candidates instead of a ranking.")
@click.pass_context
def usage(ctx: click.Context, days: int, retire: bool) -> None:
    """Report who reads each dataset over a window of recorded reads."""
    with _project(ctx) as project:
        window = timedelta(days=days)
        stats = project.store.usage(window=window)
        if not stats:
            raise click.ClickException(
                f"no reads recorded in the last {days} day(s); nothing to report. "
                "Absence of read events is not evidence a dataset is unused."
            )
        if not retire:
            click.echo(render.usage_to_markdown(stats))
            return

        candidates = usage_mod.retirement_candidates(project.graph(), stats, window=window)
        click.echo(render.retirement_to_markdown(candidates))


@main.command()
@click.option("--days", default=90, show_default=True, help="Window of read history to read.")
@click.option(
    "--threshold",
    type=float,
    required=True,
    help="Spend above which an unread dataset is worth reviewing. No default: the "
    "right number is a fraction of a budget this tool cannot see.",
)
@click.option("--price-per-partition", default=0.0, help="Cost model: price per partition.")
@click.option("--price-per-tb", default=0.0, help="Cost model: price per TB scanned.")
@click.pass_context
def value(
    ctx: click.Context,
    days: int,
    threshold: float,
    price_per_partition: float,
    price_per_tb: float,
) -> None:
    """Set each dataset's lifetime cost against whether anyone reads it."""
    with _project(ctx) as project:
        model = CostModel(
            price_per_partition=price_per_partition, price_per_tb_scanned=price_per_tb
        )
        runs = project.store.runs()
        if not runs:
            raise click.ClickException(
                "no runs recorded, so no dataset has a lifetime cost. Record runs with "
                "`Store.record_run` from your orchestrator first."
            )
        totals = lifetime.accumulate(runs, model)
        window = timedelta(days=days)
        reads = {ds: s.reads for ds, s in project.store.usage(window=window).items()}
        click.echo(
            lifetime.summarize(lifetime.value(totals, reads, threshold=threshold, window=window))
        )


@main.command()
@click.option("--dataset", required=True, help="Dataset whose published artefacts to list.")
@click.option("--reason", default="", help="Why it is being restated, for the notice.")
@click.pass_context
def impact(ctx: click.Context, dataset: str, reason: str) -> None:
    """Name every published artefact downstream of a dataset.

    The question conventional lineage cannot answer, because it stops at the tables.
    """
    with _project(ctx) as project:
        ds = project.config.resolve(dataset)
        graph = project.graph()
        if ds not in set(graph.datasets):
            raise click.ClickException(f"{ds} is not in the stored graph; run `fathom ingest`")
        click.echo(sinks.notice_text(graph, ds, reason=reason))
        if sinks.has_regulatory_exposure(graph, ds):
            sys.exit(1)


@main.command()
@click.option("--min-k", default=5, show_default=True, help="Average group size floor.")
@click.pass_context
def risk(ctx: click.Context, min_k: int) -> None:
    """Report columns that identify nobody alone and everybody together.

    Proves risk and never proves safety: the bound needs the distinct count of the
    quasi-identifier *combination*, which is a scan this does not do.
    """
    with _project(ctx) as project:
        labels = project.labels(save=False)
        found = False
        for dataset in project.graph().datasets:
            profile = project.store.latest_profile(dataset)
            if profile is None:
                continue
            report = reid.assess(profile, labels, k_threshold=min_k)
            if report.is_clear and not report.unmeasurable:
                continue
            found = True
            click.echo(report.summary())
            click.echo("")
        if not found:
            raise click.ClickException(
                "no profiles in the store to assess; run `fathom profile` first"
            )


@main.command()
@click.pass_context
def contracts(ctx: click.Context) -> None:
    """Verify every contract declared in fathom.yml against what is currently true."""
    with _project(ctx) as project:
        declared = project.config.contracts
        if not declared:
            raise click.ClickException(
                "no contracts declared in fathom.yml; add a `contracts:` block"
            )
        breached = 0
        for entry in declared:
            contract = contracts_mod.Contract(
                dataset=entry.dataset,
                producer=entry.producer,
                consumers=entry.consumers,
                columns=entry.columns,
                max_staleness=entry.max_staleness,
                note=entry.note,
            )
            profile = project.store.latest_profile(entry.dataset)
            age = None
            if entry.max_staleness is not None:
                built = project.store.last_profiled(entry.dataset)
                age = (now() - as_utc(built)) if built else None
            report = contracts_mod.verify(contract, profile=profile, age=age)
            click.echo(report.summary())
            breached += len(report.errors)
        if breached:
            click.echo("")
            click.echo(f"{breached} contract error(s)", err=True)
            sys.exit(1)


@main.command()
@click.option("--limit", default=20, show_default=True, help="How many revisions to show.")
@click.option("--edge", default="", help="Restrict to one edge, as 'src->dst'.")
@click.option("--unsafe", is_flag=True, help="Only revisions that narrowed or removed an edge.")
@click.pass_context
def history(ctx: click.Context, limit: int, edge: str, unsafe: bool) -> None:
    """Show the graph's revision history — who changed an edge, and when.

    With `--edge`, answers the question an incident actually asks: six days of
    downstream data stopped being invalidated, so when did that window shrink.
    """
    with _project(ctx) as project:
        if edge:
            src_name, sep, dst_name = edge.partition("->")
            if not sep:
                raise click.BadParameter("--edge takes 'src->dst'")
            changes = project.store.edge_changes(
                project.config.resolve(src_name.strip()),
                project.config.resolve(dst_name.strip()),
                verb="narrowed" if unsafe else None,
            )
            if not changes:
                raise click.ClickException(f"no recorded revision touched {edge}")
            for change in changes:
                click.echo(
                    f"{change['digest']} {change['at'].date().isoformat()} "
                    f"{change['author'] or '(unknown)'}: {change['verb']}"
                    + (f" — {change['note']}" if change["note"] else "")
                )
            return

        revisions = project.store.unsafe_revisions() if unsafe else project.store.revisions()
        if not revisions:
            raise click.ClickException(
                "no revisions recorded; run `fathom ingest` to record the first"
            )
        for entry in reversed(revisions[-limit:]):
            mark = "" if entry["is_safe"] else "  UNSAFE"
            click.echo(
                f"{entry['digest']} {entry['at'].date().isoformat()} "
                f"{entry['author'] or '(unknown)'}: {entry['note'] or '(no note)'}"
                f"  [{entry['datasets']} datasets, {entry['edges']} edges]{mark}"
            )


@main.command()
@click.option(
    "--flavor",
    type=click.Choice(["airflow", "dagster", "prefect", "shell", "json"]),
    default="airflow",
    show_default=True,
)
@click.option("--dirty", "dirties", multiple=True, help="Same syntax as `fathom plan`.")
@click.option("--command", default="fathom rebuild", show_default=True)
@click.option("--out", type=click.Path(), default="", help="Write to a file instead of stdout.")
@click.pass_context
def dag(ctx: click.Context, flavor: str, dirties: tuple[str, ...], command: str, out: str) -> None:
    """Generate the DAG file your orchestrator already reads.

    Nothing here imports Airflow, Dagster, or Prefect: the output is a file you commit
    and read, not a runtime binding. Intervals, retries, and alerting are deliberately
    absent, because guessing at them would be wrong in a way that looks authoritative.
    """
    with _project(ctx) as project:
        graph = project.graph()
        plan = graph.invalidate(_seeds(project, dirties))
        if plan.is_empty:
            raise click.ClickException("nothing to rebuild; no DAG to generate")
        built = schedule.schedule(graph, plan)

        if flavor == "shell":
            body = schedule.to_shell(built, command=command)
        elif flavor == "json":
            body = schedule.to_dag(built)
        else:
            body = orchestrators.write_dag(built, flavor=flavor, command=command)

        if out:
            Path(out).write_text(body)
            click.echo(f"wrote {out} ({len(body.splitlines())} lines)")
        else:
            click.echo(body)


@main.command()
@click.option("--dataset", required=True, help="Dataset to learn a baseline for.")
@click.option(
    "--cycle",
    type=click.Choice(["day_of_week", "hour_of_day", "day_of_month", "month_of_year"]),
    default="day_of_week",
    show_default=True,
)
@click.option("--min-observations", default=4, show_default=True)
@click.pass_context
def seasonal(ctx: click.Context, dataset: str, cycle: str, min_observations: int) -> None:
    """Learn a baseline bucketed by a cycle, for data that knows Tuesday from Sunday.

    A flat bound across a weekly cycle is wide enough to admit Tuesday's floor and
    Sunday's ceiling, which is to say wide enough to catch nothing. Reports how much
    of the variation the cycle actually explains, so reaching for this over
    `fathom check` stays a decision rather than a default.
    """
    with _project(ctx) as project:
        ds = project.config.resolve(dataset)
        history = project.store.seasonal_observations(ds)
        if not history:
            raise click.ClickException(
                f"no partition-dated profiles for {ds}; run `fathom profile` over "
                "partitions before a cycle can be learned"
            )

        chosen = seasonal_mod.Cycle(cycle)
        baseline = seasonal_mod.learn_seasonal(
            history, cycle=chosen, min_observations=min_observations
        )
        click.echo(render.seasonal_to_markdown(baseline))

        score = seasonal_mod.strength(history, cycle=chosen)
        click.echo("")
        if score is None:
            click.echo(
                "seasonality: no answer — not enough spread across buckets to measure. "
                "That is not the same as no seasonality."
            )
        else:
            click.echo(f"seasonality: {score:.0%} of the variation is explained by {cycle}")
            if score < 0.2:
                click.echo(
                    "  Low. A flat bound from `fathom check` is the better tool here, "
                    "with less machinery behind it."
                )


if __name__ == "__main__":  # pragma: no cover
    main()
