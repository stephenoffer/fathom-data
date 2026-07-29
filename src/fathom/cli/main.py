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
from ..adapters import get_adapter, registered
from ..core.errors import ConfigError
from ..core.types import UNPARTITIONED, DatasetId, KeyPredicate
from ..core.util.clock import as_utc, now
from ..core.util.text import did_you_mean
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
from . import explain as explain_topics
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
    """Parse `FIELD=VALUE,FIELD=VALUE` against a spec, as a click parameter would.

    Delegates to `KeyPredicate.parse` so the CLI syntax and the Python one cannot
    drift, and re-raises as `BadParameter` so click prints usage rather than a
    traceback.
    """
    try:
        return KeyPredicate.parse(binding, spec)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


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
            suggestion = did_you_mean(str(dataset), [str(d) for d in known])
            click.echo(
                f"  ! {dataset} is not in the graph, so nothing will propagate from it"
                f"{suggestion or '. Run `fathom lineage` to see what is'}",
                err=True,
            )
        out.setdefault(dataset, []).append(
            _parse_bindings(binding, graph.spec(dataset) or UNPARTITIONED)
        )
    return out


class _Sections(click.Group):
    """A group that lists its commands by what they are for, not alphabetically.

    Twenty-odd commands in one flat alphabetical list tells a new reader nothing
    about where to start. Grouped by stage, the list is itself the walkthrough:
    set up, learn the graph, plan against it, then everything that reads it.
    """

    #: Section title -> commands, in the order a project actually uses them.
    SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Start here", ("init", "doctor", "explain", "adapters")),
        ("Build the graph", ("ingest", "lineage", "history")),
        ("Plan a rebuild", ("detect", "plan", "dag", "shadow")),
        ("Watch the data", ("profile", "check", "completeness", "seasonal")),
        ("Govern it", ("label", "erase", "risk", "contracts")),
        ("Justify it", ("usage", "value", "impact")),
    )

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        listed: set[str] = set()
        for title, names in self.SECTIONS:
            rows = []
            for name in names:
                command = self.get_command(ctx, name)
                if command is None or command.hidden:
                    continue
                listed.add(name)
                rows.append((name, command.get_short_help_str(limit=68)))
            if rows:
                with formatter.section(title):
                    formatter.write_dl(rows)

        # Anything added later still appears, so a new command cannot go missing
        # just because nobody remembered to put it in a section.
        rest = [
            (name, cmd.get_short_help_str(limit=68))
            for name in sorted(self.list_commands(ctx))
            if name not in listed and (cmd := self.get_command(ctx, name)) and not cmd.hidden
        ]
        if rest:
            with formatter.section("Other"):
                formatter.write_dl(rest)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Suggest the nearest command before giving up on an unknown one."""
        name = args[0] if args else ""
        if name and self.get_command(ctx, name) is None:
            suggestion = did_you_mean(name, self.list_commands(ctx))
            if suggestion:
                ctx.fail(f"no such command {name!r}{suggestion}")
        return super().resolve_command(ctx, args)


@click.group(cls=_Sections, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="fathom")
@click.option("--config", type=click.Path(), envvar="FATHOM_CONFIG", help="Path to fathom.yml.")
@click.option("--store", type=click.Path(), envvar="FATHOM_STORE", help="Override the store path.")
@click.pass_context
def main(ctx: click.Context, config: str | None, store: str | None) -> None:
    """Lineage, partition-scoped invalidation, profiling, and policy for data platforms.

    Nothing here writes to your data. `plan` prints what it would rebuild and `erase`
    prints what it would destroy; applying either needs an engine binding you supply
    deliberately, from a pipeline rather than a shell.

    \b
    First time here:
      fathom init          write a starter fathom.yml, then edit the datasets block
      fathom ingest        build the dependency graph from your SQL or dbt manifest
      fathom doctor        find what would silently make plans worse
      fathom plan --detect see what a source change would invalidate

    \b
    Stuck on a word? Every term in a warning has an entry:
      fathom explain widening
      fathom explain               list every topic

    \b
    Exit codes:
      0  success, and nothing needed your attention
      1  a check failed — drift, a policy violation, a breached contract, an
         incomplete erasure, or a missed partition in shadow mode
      2  the command was used wrongly, or the config could not be read
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["store"] = store


# -- setup ---------------------------------------------------------------------


@main.command()
@click.argument("topic", required=False)
def explain(topic: str | None) -> None:
    """Explain a concept this tool's output assumes you know.

    \b
    Examples:
      fathom explain                list every topic
      fathom explain widening       why a plan rebuilt more than expected
      fathom explain unbounded      what an unbounded mapping costs
      fathom explain shadow-mode    how to decide whether to trust the planner
    """
    if not topic:
        click.echo("Topics — pass one to `fathom explain`:\n")
        width = max(len(name) for name in explain_topics.titles())
        for name in explain_topics.titles():
            click.echo(f"  {name:<{width}}  {explain_topics.TOPICS[name].gloss}")
        click.echo("\nFull guides live in docs/. Start with docs/guide/concepts.md.")
        return

    found = explain_topics.lookup(topic)
    if found is None:
        raise click.ClickException(
            f"no topic {topic!r}"
            f"{did_you_mean(topic, explain_topics.titles())}"
            f"\nRun `fathom explain` with no argument to list them."
        )
    click.echo(found.render())


@main.command()
@click.option("--force", is_flag=True, help="Overwrite an existing config.")
def init(force: bool) -> None:
    """Write a starter fathom.yml in the current directory.

    \b
    Example:
      fathom init
      fathom init --force    replace an existing config
    """
    target = Path(CONFIG_NAMES[0])
    if target.exists() and not force:
        raise click.ClickException(
            f"{target} already exists. Pass --force to overwrite it, or edit it in "
            f"place — `fathom doctor` will tell you what it is still missing."
        )
    target.write_text(STARTER_CONFIG)
    click.echo(f"wrote {target}")
    click.echo("")
    click.echo("Next, in order:")
    click.echo("  1. Edit the `datasets` block — name your tables and how each is")
    click.echo("     partitioned. This is the declaration that makes plans precise.")
    click.echo("  2. Point the `lineage` block at your SQL or dbt manifest.")
    click.echo("  3. `fathom ingest`   build the graph")
    click.echo("  4. `fathom doctor`   check for what would silently make plans worse")
    click.echo("")
    click.echo("Commit fathom.yml. Do not commit the store — it rebuilds, and it conflicts.")


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Explain what each capability means.")
def adapters(verbose: bool) -> None:
    """List registered adapters, and what each can actually do.

    Adapters declare capabilities rather than implement everything. Most surprises
    about a plan are a declared limit showing through, so this is worth reading
    before concluding the planner is being conservative for no reason.

    \b
    Example:
      fathom adapters
      fathom adapters --verbose
    """
    names = registered()
    if not names:
        raise click.ClickException("no adapters registered; this is a broken installation")
    for name in names:
        if not verbose:
            click.echo(name)
            continue
        click.echo(name)
        caps = getattr(get_adapter(name), "capabilities", None)
        if caps is None:
            click.echo("  capabilities are declared per instance, not per class")
        else:
            for line in caps.explain():
                click.echo(f"  {line}")
        click.echo("")


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Report configuration problems that would silently degrade a plan.

    Everything here is a thing that makes plans coarser without making them fail,
    which is the failure mode worth checking for: a tool that quietly rebuilds more
    than it needs to looks like it is working.

    \b
    Example:
      fathom doctor
    """
    with _project(ctx) as project:
        click.echo(f"config   {project.config.path}")
        click.echo(f"store    {project.store.path}")
        click.echo(f"system   {project.config.system}")
        click.echo(f"datasets {len(project.config.datasets)}, edges {len(project.graph().edges)}")

        click.echo("")
        click.echo(project.graph().describe())

        problems = project.doctor()
        if not problems:
            click.echo("\nno problems found")
            return
        click.echo(f"\n{len(problems)} problem(s):")
        for problem in problems:
            click.echo(f"  {problem}")
        click.echo("")
        click.echo(
            "Each of these makes plans coarser without making them fail. "
            "`fathom explain widening` covers what that costs."
        )


# -- graph ---------------------------------------------------------------------


@main.command()
@click.option("--author", default="", help="Recorded against this ingest's revision.")
@click.option("--note", default="", help="Why the graph changed, for the history.")
@click.pass_context
def ingest(ctx: click.Context, author: str, note: str) -> None:
    """Build the dependency graph from every configured lineage source.

    Re-run this whenever the pipeline changes. Each run records a revision, so
    `fathom history` can later answer who narrowed an edge and when.

    \b
    Example:
      fathom ingest
      fathom ingest --author "$USER" --note "added the fx_rates join"
    """
    with _project(ctx) as project:
        result = project.ingest(author=author, note=note)
        click.echo(result.summary())
        for note in result.notes[:20]:
            click.echo(f"  ! {note}", err=True)
        if len(result.notes) > 20:
            click.echo(f"  ! ... and {len(result.notes) - 20} more", err=True)
        if not result.edges:
            raise click.ClickException(
                "no lineage extracted from any configured source. Check the `lineage` "
                "block in your config: `paths` must match real files, and a `dialect` "
                "the parser does not know reads as zero edges rather than as an error. "
                "`fathom doctor` reports which sources were reachable."
            )
        click.echo("")
        click.echo("Next: `fathom doctor` for what would make plans coarse, then")
        click.echo("      `fathom plan --detect` to see what a source change invalidates.")


@main.command()
@click.option("--select", "selector", help="Restrict to a selector, e.g. '+gold.monthly+'.")
@click.option(
    "--explain", "as_prose", is_flag=True, help="Say what each edge claims, in sentences."
)
@click.pass_context
def lineage(ctx: click.Context, selector: str | None, as_prose: bool) -> None:
    """Show the stored dependency graph.

    Each line is `source -> target {mapping} [evidence]`. The mapping is the part
    nobody can verify by reading the SQL, so `--explain` spells it out.

    \b
    Examples:
      fathom lineage
      fathom lineage --select '+gold.monthly+'    that model and both directions
      fathom lineage --select tag:pii             everything carrying a tag
      fathom lineage --explain                    what each edge actually claims

    Selector syntax is dbt's — `fathom explain selector` covers it.
    """
    with _project(ctx) as project:
        graph = project.graph()
        if not graph.edges:
            raise click.ClickException(
                "no lineage in the store yet; run `fathom ingest` first to build the graph"
            )
        graph = _narrow(graph, selector)
        if not graph.edges:
            raise click.ClickException(
                f"the selector {selector!r} matched no edges. An empty result usually "
                f"means the name did not resolve rather than that nothing matched — "
                f"run `fathom lineage` unfiltered to see what is there."
            )
        for edge in graph.edges:
            if as_prose:
                click.echo(edge.explain())
                click.echo("")
                continue
            click.echo(str(edge))
            for src_col, dst_col in edge.columns:
                click.echo(f"    {src_col} -> {dst_col}")


@main.command()
@click.pass_context
def detect(ctx: click.Context) -> None:
    """Ask every configured source what changed since the last run.

    Advances a resume token per source, so two runs in a row report the second as
    empty. That is the point — it is the seed for `fathom plan --detect`, not a
    listing of what exists.

    \b
    Example:
      fathom detect
    """
    with _project(ctx) as project:
        if not project.config.sources:
            raise click.ClickException(
                "no sources configured, so there is nothing to ask what changed. Add a "
                "`source:` to entries in the `datasets` block of fathom.yml — a dataset "
                "with a partition spec but no source can still be planned against, it "
                "just cannot seed a plan by itself."
            )
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
@click.option(
    "--dirty",
    "dirties",
    multiple=True,
    help="What changed: TABLE@FIELD=VALUE[,FIELD=VALUE]. Repeatable.",
)
@click.option("--detect", "auto", is_flag=True, help="Discover the seeds by scanning sources.")
@click.option("--json", "as_json", is_flag=True, help="Emit the plan as JSON for a pipeline.")
@click.option(
    "--explain",
    "explain_for",
    default="",
    metavar="DATASET",
    help="Why one dataset is in the plan, and why so much of it.",
)
@click.pass_context
def plan(
    ctx: click.Context,
    dirties: tuple[str, ...],
    auto: bool,
    as_json: bool,
    explain_for: str,
) -> None:
    """Show which partitions a set of source changes invalidates.

    Prints; never rebuilds. The output is in build order, dependencies first.

    \b
    Examples:
      fathom plan --dirty 'raw.events@dt=2026-03-14'
      fathom plan --dirty 'raw.events@dt=2026-03-14,region=eu'
      fathom plan --dirty 'raw.events@dt=2026-03-14' --dirty 'raw.fx@dt=2026-03-14'
      fathom plan --detect                      seed from what the sources report
      fathom plan --detect --json               for an orchestrator to act on
      fathom plan --detect --explain gold.monthly

    \b
    If the plan is bigger than you expected, in order:
      fathom explain widening
      fathom plan ... --explain THE_DATASET
      fathom doctor
    """
    if not dirties and not auto:
        raise click.UsageError(
            "nothing to plan from. Pass --dirty to say what changed, e.g.\n"
            "  fathom plan --dirty 'raw.events@dt=2026-03-14'\n"
            "or --detect to ask the configured sources what changed since last time."
        )

    with _project(ctx) as project:
        result = project.plan(detect=True) if auto else project.plan(_seeds(project, dirties))
        if explain_for:
            target = project.config.resolve(explain_for)
            click.echo(result.explain(target))
            return
        if as_json:
            # The plan is the one output another program acts on, and screen-scraping
            # a summary written for humans is how an orchestrator ends up rebuilding
            # the wrong thing after a wording change.
            click.echo(render.plan_to_json(result))
            return
        if result.is_empty:
            click.echo("nothing to rebuild")
            if auto:
                click.echo(
                    "No source reported a change since the last `detect`. That is the "
                    "expected result on a second run — the resume token already moved."
                )
            return

        click.echo(result.summary())
        click.echo("")
        click.echo(f"{result.total_partitions} partition(s) across {len(result)} dataset(s).")
        if result.widened:
            click.echo("\nwidened to whole dataset (no provable partition bound):", err=True)
            for ds in sorted(result.widened, key=str):
                for reason in result.reasons[ds][:2]:
                    click.echo(f"  {ds}: {reason}", err=True)
            click.echo(
                "  Run `fathom explain widening` for what this costs and how to fix it.",
                err=True,
            )
        if result.cyclic:
            cycles = ", ".join(str(d) for d in sorted(result.cyclic, key=str))
            click.echo(f"\ncycles detected in: {cycles}", err=True)
            click.echo(
                "  A dataset that reads itself cannot converge, so the planner took it "
                "whole rather than looping.",
                err=True,
            )


# -- profiling -----------------------------------------------------------------


@main.command()
@click.argument("dataset", required=False)
@click.pass_context
def profile(ctx: click.Context, dataset: str | None) -> None:
    """Profile datasets from Parquet footers alone. Reads no data pages.

    Costs a metadata read rather than a scan, which is what makes profiling
    affordable enough to run continuously. Establishes the baseline `fathom check`
    compares against — a first check with no prior profile is not a clean result,
    it is no result.

    \b
    Examples:
      fathom profile                 every path-backed dataset in the config
      fathom profile gold.monthly    just one
    """
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
            raise click.ClickException(
                "nothing could be profiled. Only path-backed datasets qualify here — "
                "Parquet, Delta, or Iceberg under a filesystem or object store. A "
                "warehouse table is profiled through its engine adapter instead, which "
                "needs a connection you supply with `register_runner`."
            )


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Emit findings as JSON for a pipeline.")
@click.pass_context
def check(ctx: click.Context, as_json: bool) -> None:
    """Compare each dataset against its last profile, and attribute any drift.

    "revenue moved 8%" is an alert. "revenue moved because fx_rates changed three
    hops upstream" is a diagnosis, and the second is what the graph adds. Exits 1
    when any finding is an error.

    \b
    Examples:
      fathom check
      fathom check --json      for a CI gate
    """
    import json as _json

    from ..core.types import ColumnRef

    with _project(ctx) as project:
        results = project.check()
        if not results:
            raise click.ClickException(
                "no path-backed datasets to check. `check` compares against stored "
                "profiles, so run `fathom profile` first — and note that a dataset "
                "profiled for the first time has nothing to be compared against yet."
            )

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
    """Infer column labels, propagate them, and check configured sink policies.

    Nobody hand-labels 40,000 columns. Inference proposes; the profile rejects the
    bad guesses (a `latitude` whose values top out at 4,000 is not a latitude); and
    labels propagate along graph edges so a derived table inherits what it carries.

    A `*` marks a label somebody confirmed. Exits 1 on a policy violation.

    \b
    Example:
      fathom label
    """
    with _project(ctx) as project:
        labels = project.labels()
        if not labels:
            raise click.ClickException(
                "no labels could be inferred. Inference reads profiles, not schemas, "
                "so run `fathom profile` on some datasets first — a column name alone "
                "is deliberately not enough evidence to label one."
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

    Erasure may under-delete and refuse; it must never over-delete. Exits 1 when the
    plan is incomplete — a model that retains the subject, or storage that cannot
    destroy anything, is reported rather than rounded up to success.

    \b
    Example:
      fathom erase --subject u1 --key-column user_id --origin raw.events
      fathom erase --subject u1 --key-column user_id --origin raw.events \\
          --proof proof.json      # needs FATHOM_SALT
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
    """Report accumulated shadow results: how much was skipped, and what was missed.

    How you decide whether to trust the planner. `missed` must be zero, and this
    exits non-zero the moment it is not. Accumulate weeks before anything writes —
    running it costs nothing, because the full rebuild happens either way.

    \b
    Example:
      fathom shadow
      fathom shadow --json     for a CI gate
    """
    with _project(ctx) as project:
        summary = project.store.shadow_summary()
        if not summary["runs"]:
            raise click.ClickException(
                "no shadow observations recorded yet. Shadow mode is fed from your "
                "pipeline: call `fathom.shadow.run(...)` alongside a full rebuild and "
                "pass the store, then this reports what accumulated. "
                "`fathom explain shadow-mode` covers why that order matters."
            )

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

    The only check that can see a partition which never arrived: it has no profile
    to drift and no rows to fail an expectation, so nothing else can.

    \b
    Example:
      fathom completeness --dataset raw.events --since 2026-03-01 --until 2026-03-31
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
    """Report who reads each dataset over a window of recorded reads.

    Nothing here says "unused". Everything says "no reads observed", and carries the
    window it observed over — query logs have retention limits, and a table read
    once a quarter looks dead over thirty days. Deleting a table read annually for a
    filing is the one mistake in this area a rebuild cannot undo.

    \b
    Examples:
      fathom usage
      fathom usage --days 180
      fathom usage --retire      datasets nothing reads whose descendants nothing reads
    """
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
    """Set each dataset's lifetime cost against whether anyone reads it.

    `--threshold` has no default on purpose: the right number is a fraction of a
    budget this tool cannot see, and a default would be a made-up one carrying the
    authority of a real one.

    \b
    Example:
      fathom value --threshold 500 --price-per-partition 0.02
    """
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
    Exits 1 when a regulatory filing is downstream, because that is a different
    conversation from a stale dashboard.

    \b
    Example:
      fathom impact --dataset gold.monthly --reason "restated fx rates"
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
    quasi-identifier *combination*, which is a scan this does not do. A birth date
    identifies nobody and a postcode identifies nobody; together they identify most
    of a population, and per-column labelling is structurally blind to it.

    \b
    Example:
      fathom risk
      fathom risk --min-k 10
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
    """Verify every contract declared in fathom.yml against what is currently true.

    A failing test says a column vanished. A breached contract says who was promised
    it and is therefore owed a conversation. Exits 1 on a breach.

    \b
    Example:
      fathom contracts
    """
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

    \b
    Examples:
      fathom history
      fathom history --unsafe                        only narrowings and removals
      fathom history --edge 'raw.events->silver.events'
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

    \b
    Examples:
      fathom dag --dirty 'raw.events@dt=2026-03-14'
      fathom dag --flavor shell --dirty 'raw.events@dt=2026-03-14'
      fathom dag --flavor dagster --out dags/rebuild.py
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

    \b
    Example:
      fathom seasonal --dataset raw.events --cycle day_of_week
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
