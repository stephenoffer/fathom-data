"""CLI behaviour, including the exit codes CI will depend on.

Everything is driven from a `fathom.yml`, because that is how the tool is meant to
be used — flags are for overrides, not for describing a project.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from conftest import write_partition
from fathom.cli import main

SILVER_SQL = "CREATE TABLE silver.events AS SELECT dt, region, user_id, amount FROM raw.events;"
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
    (models / "silver.sql").write_text(SILVER_SQL)
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


def config_for(tmp_path: Path, body: str) -> CliRunner:
    (tmp_path / "fathom.yml").write_text(body)
    return CliRunner()


def storage_config(root: Path, *, policies: str = "") -> str:
    return f"""\
version: 1
store: .fathom/fathom.db
system: duckdb
datasets:
  - name: {root.resolve()}
    adapter: storage
    partition: [{{field: dt, grain: day}}, {{field: region}}]
{policies}"""


# -- setup ---------------------------------------------------------------------


def test_adapters_lists_the_registry():
    result = CliRunner().invoke(main, ["adapters"])
    assert result.exit_code == 0
    assert {"delta", "duckdb", "snowflake", "databricks", "bigquery", "storage"} <= set(
        result.output.split()
    )


def test_init_writes_a_config_that_is_itself_valid(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(main, ["init"]).exit_code == 0

        from fathom.config import load_config

        assert load_config("fathom.yml").system == "duckdb"


def test_init_refuses_to_clobber(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["init"])
        assert result.exit_code != 0
        assert "--force" in result.output


def test_missing_config_says_how_to_make_one(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["lineage"])
        assert result.exit_code != 0
        assert "fathom.yml" in result.output


# -- graph ---------------------------------------------------------------------


def test_ingest_persists_a_graph(run):
    result = run("ingest")
    assert result.exit_code == 0
    assert "2 edge(s)" in result.output

    listed = run("lineage")
    assert "raw.events -> duckdb/silver.events" in listed.output
    assert "dt@day->month" in listed.output


def test_lineage_before_ingest_says_what_to_run(run):
    result = run("lineage")
    assert result.exit_code != 0
    assert "run `fathom ingest` first" in result.output


def test_doctor_reports_a_healthy_project(run):
    run("ingest")
    result = run("doctor")
    assert result.exit_code == 0
    assert "edges 2" in result.output
    assert "no problems found" in result.output


def test_doctor_flags_a_dataset_with_no_partition_spec(project):
    """The most common cause of "why is it rebuilding everything"."""
    (project / "fathom.yml").write_text(
        CONFIG.replace(
            "  - name: gold.monthly\n    partition: [{field: dt, grain: month}, {field: region}]\n",
            "",
        )
    )
    runner = CliRunner()
    runner.invoke(main, ["--config", str(project / "fathom.yml"), "ingest"])
    result = runner.invoke(main, ["--config", str(project / "fathom.yml"), "doctor"])
    assert "no partition spec" in result.output


# -- planning ------------------------------------------------------------------


def test_plan_scopes_a_day_to_a_month(run):
    run("ingest")
    result = run("plan", "--dirty", "raw.events@dt=2026-03-14,region=eu")

    assert result.exit_code == 0
    assert "dt=2026-03-14T00:00:00/region=eu" in result.output
    assert "dt=2026-03-01T00:00:00/region=eu" in result.output


def test_plan_warns_when_a_seeded_table_is_unknown(run):
    """A typo would otherwise produce a confident plan containing only the typo."""
    run("ingest")
    result = run("plan", "--dirty", "other.thing@dt=2026-03-14")
    assert "is not in the graph" in result.output
    assert "silver.events" not in result.output


def test_plan_needs_seeds_or_detect(run):
    run("ingest")
    result = run("plan")
    assert result.exit_code != 0
    assert "--detect" in result.output


# -- detection -----------------------------------------------------------------


def test_detect_reports_changes_then_converges(tmp_path, lake):
    runner = config_for(tmp_path, storage_config(lake))
    args = ["--config", str(tmp_path / "fathom.yml"), "detect"]

    first = runner.invoke(main, args)
    assert first.exit_code == 0
    assert "no changes" not in first.output
    assert "no changes" in runner.invoke(main, args).output


def test_detect_picks_up_a_new_partition(tmp_path, lake):
    runner = config_for(tmp_path, storage_config(lake))
    args = ["--config", str(tmp_path / "fathom.yml"), "detect"]

    runner.invoke(main, args)
    write_partition(lake, dt=date(2026, 3, 20), region="apac", amounts=[1.0])
    assert "region=apac" in runner.invoke(main, args).output


def test_plan_can_discover_its_own_seeds(tmp_path, lake):
    runner = config_for(tmp_path, storage_config(lake))
    result = runner.invoke(main, ["--config", str(tmp_path / "fathom.yml"), "plan", "--detect"])
    assert result.exit_code == 0
    assert "dt=2026-03-14T00:00:00" in result.output


# -- profiling -----------------------------------------------------------------


def test_profile_reports_from_footers(tmp_path, lake):
    runner = config_for(tmp_path, storage_config(lake))
    result = runner.invoke(main, ["--config", str(tmp_path / "fathom.yml"), "profile"])
    assert result.exit_code == 0
    assert "8 rows across 3 file(s)" in result.output


def test_check_records_a_baseline_then_detects_drift(tmp_path, lake):
    runner = config_for(tmp_path, storage_config(lake))
    args = ["--config", str(tmp_path / "fathom.yml"), "check"]

    assert "no drift" in runner.invoke(main, args).output

    write_partition(lake, dt=date(2026, 3, 20), region="apac", amounts=[1.0] * 5000)
    assert "row count moved" in runner.invoke(main, args).output


# -- labels --------------------------------------------------------------------


def test_label_requires_profiles_first(run):
    result = run("label")
    assert result.exit_code != 0
    assert "fathom profile" in result.output


def test_label_flags_pii_reaching_a_forbidden_sink(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    users = tmp_path / "users"
    users.mkdir()
    pq.write_table(
        pa.table({"email_address": pa.array(["a@b.c"]), "amount": pa.array([1.0])}),
        users / "p.parquet",
    )
    runner = config_for(
        tmp_path,
        f"""\
version: 1
store: .fathom/fathom.db
system: duckdb
datasets:
  - name: {users.resolve()}
    adapter: storage
policies:
  - dataset: {users.resolve()}
    forbid: [pii]
    reason: not cleared for personal data
""",
    )
    base = ["--config", str(tmp_path / "fathom.yml")]

    runner.invoke(main, [*base, "profile"])
    result = runner.invoke(main, [*base, "label"])

    assert "email_address" in result.output
    assert "policy: 1 violation(s)" in result.output
    assert result.exit_code == 1


# -- erasure -------------------------------------------------------------------


def test_erase_plans_across_derived_datasets(run, project):
    run("ingest")
    proof = project / "proof.json"
    result = run(
        "erase",
        "--subject",
        "u1",
        "--key-column",
        "user_id",
        "--origin",
        "raw.events",
        "--partition",
        "dt=2026-03-14",
        "--reference",
        "DSR-1",
        "--proof",
        str(proof),
    )

    assert "silver.events" in result.output and "gold.monthly" in result.output

    body = json.loads(proof.read_text())
    assert body["reference"] == "DSR-1"
    assert "u1" not in proof.read_text()
    assert body["complete"] is False  # a dry run is never complete


def test_erase_orders_sources_before_derived(run):
    run("ingest")
    result = run("erase", "--subject", "u1", "--key-column", "user_id", "--origin", "raw.events")
    lines = [ln for ln in result.output.splitlines() if "duckdb/" in ln]
    positions = {
        name: i
        for i, ln in enumerate(lines)
        for name in ("raw", "silver", "gold")
        if f"{name}." in ln
    }
    assert positions["raw"] < positions["silver"] < positions["gold"]


def test_erase_on_unconfigured_datasets_reports_incomplete(run):
    """Unconfigured datasets block rather than being assumed erasable."""
    run("ingest")
    result = run("erase", "--subject", "u1", "--key-column", "user_id", "--origin", "raw.events")
    assert result.exit_code == 1
    assert "INCOMPLETE" in result.output


# -- shadow --------------------------------------------------------------------


def test_shadow_reports_nothing_recorded(run):
    result = run("shadow")
    assert result.exit_code != 0
    assert "no shadow observations" in result.output


def test_shadow_summarizes_and_fails_on_a_miss(project):
    from datetime import UTC, datetime

    from fathom.ids import normalize_table
    from fathom.store import ShadowObservation, Store

    store_path = project / ".fathom" / "fathom.db"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = normalize_table("gold.monthly", system="duckdb")

    def record(missed: int, planned: int) -> None:
        with Store(store_path) as store:
            store.record_shadow(
                ShadowObservation(
                    dataset=dataset,
                    observed=datetime.now(UTC),
                    planned=planned,
                    actual=2,
                    missed=missed,
                    total=10,
                )
            )

    runner = CliRunner()
    args = ["--config", str(project / "fathom.yml"), "shadow"]

    record(missed=0, planned=2)
    ok = runner.invoke(main, args)
    assert ok.exit_code == 0
    assert "savings     80%" in ok.output

    record(missed=1, planned=1)
    bad = runner.invoke(main, args)
    assert bad.exit_code == 1
    assert "SOUNDNESS FAILURE" in bad.output
