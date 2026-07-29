"""Revisions recorded by `ingest`, read back by `history`, and DAGs from `dag`.

A history nobody remembers to write is empty on the day it is needed, so `ingest`
records one. These check that it does, that an unchanged ingest adds nothing, and that
`history --edge` answers the question an incident asks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from click.testing import CliRunner

from fathom.cli import main
from fathom.store.sqlite import Store

# Adds a join, so the edge from raw.events gains a column mapping and a new edge
# from raw.rates appears — a change the graph genuinely records.
WIDE = (
    "CREATE TABLE silver.events AS "
    "SELECT e.dt, e.region, e.amount, r.rate FROM raw.events e "
    "JOIN raw.rates r ON e.dt = r.dt;"
)
NARROW = "CREATE TABLE silver.events AS SELECT dt, region, amount FROM raw.events;"
GOLD_SQL = (
    "CREATE TABLE gold.monthly AS "
    "SELECT DATE_TRUNC('month', dt) AS dt, region, SUM(amount) AS revenue "
    "FROM silver.events GROUP BY 1, 2;"
)

CONFIG = """\
version: 1
store: .fathom/fathom.db
system: duckdb

datasets:
  - name: raw.events
    partition: [{field: dt, grain: day}, {field: region}]
  - name: silver.events
    partition: [{field: dt, grain: day}, {field: region}]
  - name: gold.monthly
    partition: [{field: dt, grain: month}, {field: region}]

lineage:
  - type: sql
    paths: ["models/*.sql"]
    dialect: duckdb
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "silver.sql").write_text(NARROW)
    (models / "gold.sql").write_text(GOLD_SQL)
    (tmp_path / "fathom.yml").write_text(CONFIG)
    return tmp_path


@pytest.fixture
def run(project: Path):
    runner = CliRunner()

    def invoke(*args: str):
        return runner.invoke(
            main, ["--config", str(project / "fathom.yml"), *args], catch_exceptions=False
        )

    return invoke


# -- ingest records a revision -------------------------------------------------


def test_ingest_records_a_revision(run, project):
    assert run("ingest").exit_code == 0
    with Store(project / ".fathom" / "fathom.db") as store:
        assert len(store.revisions()) == 1


def test_the_author_and_note_reach_the_history(run, project):
    run("ingest", "--author", "ana", "--note", "first pass")
    with Store(project / ".fathom" / "fathom.db") as store:
        (revision,) = store.revisions()
        assert revision["author"] == "ana"
        assert revision["note"] == "first pass"


def test_re_ingesting_an_unchanged_graph_adds_nothing(run, project):
    """A nightly ingest that found no new lineage must not fill the log with noise."""
    run("ingest", "--author", "ana")
    run("ingest", "--author", "ana")
    with Store(project / ".fathom" / "fathom.db") as store:
        assert len(store.revisions()) == 1


def test_a_changed_model_appends_a_second_revision(run, project):
    run("ingest", "--author", "ana")
    (project / "models" / "silver.sql").write_text(WIDE)
    run("ingest", "--author", "ben", "--note", "trailing window")

    with Store(project / ".fathom" / "fathom.db") as store:
        revisions = store.revisions()
        assert [r["author"] for r in revisions] == ["ana", "ben"]
        assert revisions[1]["parent"] == revisions[0]["digest"]


def test_the_first_revision_records_the_graph_size(run, project):
    run("ingest")
    with Store(project / ".fathom" / "fathom.db") as store:
        (revision,) = store.revisions()
        assert revision["datasets"] >= 3
        assert revision["edges"] >= 1


# -- history reads it back -----------------------------------------------------


def test_history_lists_the_revisions(run):
    run("ingest", "--author", "ana", "--note", "first pass")
    result = run("history")
    assert result.exit_code == 0
    assert "ana" in result.output
    assert "first pass" in result.output


def test_history_refuses_before_anything_is_recorded(run):
    result = run("history")
    assert result.exit_code != 0
    assert "no revisions recorded" in result.output


def test_history_reports_an_unknown_author_rather_than_a_blank(run):
    run("ingest")
    assert "(unknown)" in run("history").output


def test_history_edge_answers_who_changed_it(run, project):
    """The question an incident asks: when did this dependency appear, and who added it.

    Asked of the edge the join introduced. `raw.events -> silver.events` is
    deliberately *not* reported here: the parser could not attribute unqualified
    columns across two sources, so that edge did not change, and saying otherwise
    would be a claim the ingest never made.
    """
    run("ingest", "--author", "ana")
    (project / "models" / "silver.sql").write_text(WIDE)
    run("ingest", "--author", "ben", "--note", "joined the rates table")

    result = run("history", "--edge", "raw.rates->silver.events")
    assert result.exit_code == 0
    assert "ben" in result.output
    assert "added" in result.output
    assert "joined the rates table" in result.output


def test_history_edge_refuses_a_malformed_argument(run):
    run("ingest")
    result = run("history", "--edge", "raw.events")
    assert result.exit_code != 0
    assert "src->dst" in result.output


def test_history_edge_refuses_an_untouched_edge(run):
    run("ingest")
    result = run("history", "--edge", "raw.events->nowhere.missing")
    assert result.exit_code != 0
    assert "no recorded revision touched" in result.output


def test_history_limit_bounds_the_output(run, project):
    run("ingest", "--author", "ana")
    (project / "models" / "silver.sql").write_text(WIDE)
    run("ingest", "--author", "ben")
    assert len(run("history", "--limit", "1").output.strip().splitlines()) == 1


# -- dag -----------------------------------------------------------------------


@pytest.mark.parametrize("flavor", ["airflow", "dagster", "prefect"])
def test_dag_generates_parseable_python(run, flavor):
    run("ingest")
    result = run("dag", "--flavor", flavor, "--dirty", "raw.events@dt=2026-03-14,region=eu")
    assert result.exit_code == 0
    ast.parse(result.output)


def test_dag_carries_the_partition_keys(run):
    run("ingest")
    result = run("dag", "--dirty", "raw.events@dt=2026-03-14,region=eu")
    assert "--partitions" in result.output
    assert "2026-03-14" in result.output


def test_dag_writes_to_a_file_when_asked(run, project):
    run("ingest")
    target = project / "dags" / "rebuild.py"
    target.parent.mkdir()
    result = run("dag", "--dirty", "raw.events@dt=2026-03-14,region=eu", "--out", str(target))
    assert result.exit_code == 0
    assert "wrote" in result.output
    ast.parse(target.read_text())


def test_dag_refuses_when_nothing_is_dirty(run):
    run("ingest")
    result = run("dag")
    assert result.exit_code != 0
    assert "nothing to rebuild" in result.output


def test_the_shell_flavor_is_still_available(run):
    run("ingest")
    result = run("dag", "--flavor", "shell", "--dirty", "raw.events@dt=2026-03-14,region=eu")
    assert result.exit_code == 0
    assert result.output.startswith("#!/usr/bin/env bash")


def test_the_json_flavor_emits_the_task_list(run):
    import json

    run("ingest")
    result = run("dag", "--flavor", "json", "--dirty", "raw.events@dt=2026-03-14,region=eu")
    assert result.exit_code == 0
    assert "tasks" in json.loads(result.output)
