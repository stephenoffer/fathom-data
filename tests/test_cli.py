"""CLI behaviour, including the exit codes CI will depend on."""

from __future__ import annotations

import json
from datetime import UTC, date
from pathlib import Path

import pytest
from click.testing import CliRunner

from conftest import write_partition
from fathom.cli import main

SILVER_SQL = "CREATE TABLE silver.events AS SELECT dt, region, amount FROM raw.events;"
GOLD_SQL = (
    "CREATE TABLE gold.monthly AS "
    "SELECT DATE_TRUNC('month', dt) AS dt, region, SUM(amount) AS revenue "
    "FROM silver.events GROUP BY 1, 2;"
)

SPECS = [
    "--spec",
    "raw.events:dt:day",
    "--spec",
    "raw.events:region",
    "--spec",
    "silver.events:dt:day",
    "--spec",
    "silver.events:region",
    "--spec",
    "gold.monthly:dt:month",
    "--spec",
    "gold.monthly:region",
]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "silver.sql").write_text(SILVER_SQL)
    (tmp_path / "gold.sql").write_text(GOLD_SQL)
    return tmp_path


@pytest.fixture
def run(tmp_path: Path):
    runner = CliRunner()
    store = tmp_path / "state" / "fathom.db"

    def invoke(*args: str):
        return runner.invoke(main, ["--store", str(store), *args], catch_exceptions=False)

    return invoke


def ingest(run, project: Path):
    return run("ingest", str(project / "silver.sql"), str(project / "gold.sql"), *SPECS)


# -- basics --------------------------------------------------------------------


def test_adapters_lists_the_registry(run):
    result = run("adapters")
    assert result.exit_code == 0
    assert {"delta", "duckdb", "local"} <= set(result.output.split())


def test_ingest_persists_a_graph(run, project):
    result = ingest(run, project)
    assert result.exit_code == 0
    assert "2 edge(s)" in result.output

    listed = run("lineage")
    assert "raw.events -> duckdb/silver.events" in listed.output
    assert "dt@day->month" in listed.output


def test_commands_needing_a_graph_say_so(run):
    result = run("lineage")
    assert result.exit_code != 0
    assert "run `fathom ingest` first" in result.output


def test_ingest_fails_when_nothing_parses(run, tmp_path):
    (tmp_path / "junk.sql").write_text("not sql at all !!!")
    result = run("ingest", str(tmp_path / "junk.sql"))
    assert result.exit_code != 0
    assert "no lineage extracted" in result.output


# -- planning ------------------------------------------------------------------


def test_plan_scopes_a_day_to_a_month(run, project):
    ingest(run, project)
    result = run("plan", "--dirty", "raw.events@dt=2026-03-14,region=eu")

    assert result.exit_code == 0
    assert "dt=2026-03-14T00:00:00/region=eu" in result.output
    assert "dt=2026-03-01T00:00:00/region=eu" in result.output


def test_plan_warns_when_a_seeded_table_is_unknown(run, project):
    """A typo would otherwise produce a confident plan containing only the typo."""
    ingest(run, project)
    result = run("plan", "--dirty", "other.thing@dt=2026-03-14")
    assert "is not in the graph" in result.output
    assert "silver.events" not in result.output


# -- change detection ----------------------------------------------------------


def test_changed_detects_delta_tables_automatically(run, tmp_path):
    root = tmp_path / "events"
    log = root / "_delta_log"
    log.mkdir(parents=True)
    (log / f"{0:020d}.json").write_text(
        json.dumps(
            {
                "metaData": {
                    "id": "t",
                    "format": {"provider": "parquet"},
                    "schemaString": json.dumps(
                        {
                            "type": "struct",
                            "fields": [
                                {"name": "dt", "type": "date", "nullable": True, "metadata": {}}
                            ],
                        }
                    ),
                    "partitionColumns": ["dt"],
                    "configuration": {},
                }
            }
        )
        + "\n"
        + json.dumps(
            {
                "add": {
                    "path": "dt=2026-03-14/p.parquet",
                    "partitionValues": {"dt": "2026-03-14"},
                    "size": 1,
                    "modificationTime": 1,
                    "dataChange": True,
                }
            }
        )
    )

    result = run("changed", str(root))
    assert result.exit_code == 0
    assert "[delta]" in result.output
    assert "dt=2026-03-14T00:00:00" in result.output


def test_changed_token_advances_between_runs(run, lake):
    first = run("changed", str(lake), "--spec", "dt:day", "--spec", "region")
    assert "no changes" not in first.output
    second = run("changed", str(lake), "--spec", "dt:day", "--spec", "region")
    assert "no changes" in second.output


def test_reset_ignores_the_stored_token(run, lake):
    run("changed", str(lake), "--spec", "dt:day")
    result = run("changed", str(lake), "--spec", "dt:day", "--reset")
    assert "no changes" not in result.output


# -- profiling -----------------------------------------------------------------


def test_profile_reports_from_footers(run, lake):
    result = run("profile", str(lake), "--spec", "dt:day", "--spec", "region")
    assert result.exit_code == 0
    assert "8 rows across 3 file(s)" in result.output
    assert "source=footer" in result.output


def test_check_records_a_baseline_then_compares(run, lake, tmp_path):
    first = run("check", str(lake), "--spec", "dt:day", "--spec", "region")
    assert "baseline recorded" in first.output

    write_partition(lake, dt=date(2026, 3, 20), region="apac", amounts=[1.0] * 5000)
    second = run("check", str(lake), "--spec", "dt:day", "--spec", "region")
    assert "row count moved" in second.output


def test_check_exits_nonzero_on_an_error_finding(run, lake, tmp_path):
    run("check", str(lake), "--spec", "dt:day")

    import pyarrow as pa
    import pyarrow.parquet as pq

    stripped = tmp_path / "stripped"
    stripped.mkdir()
    pq.write_table(pa.table({"id": pa.array(["a"])}), stripped / "p.parquet")

    result = run("check", str(stripped), "--spec", "dt:day")
    assert "baseline recorded" in result.output  # different dataset, own baseline


# -- labels --------------------------------------------------------------------


def test_label_requires_profiles_first(run):
    result = run("label")
    assert result.exit_code != 0
    assert "fathom profile --save" in result.output


def test_label_flags_pii_reaching_a_forbidden_sink(run, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    users = tmp_path / "users"
    users.mkdir()
    pq.write_table(
        pa.table({"email_address": pa.array(["a@b.c"]), "amount": pa.array([1.0])}),
        users / "p.parquet",
    )

    run("profile", str(users), "--save")
    result = run("label", "--sink", f"{users.resolve()}:forbid=pii")

    assert "email_address" in result.output
    assert "policy: 1 violation(s)" in result.output
    assert result.exit_code == 1


def test_sink_paths_are_resolved_before_matching(run, tmp_path):
    """A symlinked path must still match the dataset it points at."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    real = tmp_path / "real"
    real.mkdir()
    pq.write_table(pa.table({"email_address": pa.array(["a@b.c"])}), real / "p.parquet")
    link = tmp_path / "link"
    link.symlink_to(real)

    run("profile", str(real), "--save")
    result = run("label", "--sink", f"{link}:forbid=pii")

    assert "policy: 1 violation(s)" in result.output
    assert result.exit_code == 1


# -- erasure -------------------------------------------------------------------


def test_erase_plans_across_derived_datasets(run, project, tmp_path):
    ingest(run, project)
    proof = tmp_path / "proof.json"
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

    assert result.exit_code == 0
    assert "silver.events" in result.output and "gold.monthly" in result.output

    body = json.loads(proof.read_text())
    assert body["reference"] == "DSR-1"
    assert "u1" not in proof.read_text()
    assert body["complete"] is False  # a dry run is never complete
    assert all(e["status"] == "planned" for e in body["entries"])


def test_erase_exits_nonzero_when_a_target_is_worm(run, project):
    ingest(run, project)
    result = run(
        "erase",
        "--subject",
        "u1",
        "--key-column",
        "user_id",
        "--origin",
        "raw.events",
        "--worm",
        "gold.monthly",
    )
    assert result.exit_code == 1
    assert "INCOMPLETE" in result.output


def test_erase_orders_sources_before_derived(run, project):
    ingest(run, project)
    result = run("erase", "--subject", "u1", "--key-column", "user_id", "--origin", "raw.events")
    lines = [ln for ln in result.output.splitlines() if "duckdb/" in ln]
    positions = {
        name: i
        for i, ln in enumerate(lines)
        for name in ("raw", "silver", "gold")
        if f"{name}." in ln
    }
    assert positions["raw"] < positions["silver"] < positions["gold"]


# -- shadow --------------------------------------------------------------------


def test_shadow_reports_nothing_recorded(run):
    result = run("shadow")
    assert result.exit_code != 0
    assert "no shadow observations" in result.output


def test_shadow_summarizes_recorded_runs(run, tmp_path):
    from datetime import datetime

    from fathom.ids import normalize_table
    from fathom.store import ShadowObservation, Store

    store_path = tmp_path / "state" / "fathom.db"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with Store(store_path) as store:
        store.record_shadow(
            ShadowObservation(
                dataset=normalize_table("gold.monthly", system="duckdb"),
                observed=datetime.now(UTC),
                planned=2,
                actual=2,
                missed=0,
                total=10,
            )
        )

    result = run("shadow")
    assert result.exit_code == 0
    assert "savings     80%" in result.output
    assert "missed      0" in result.output


def test_shadow_fails_loudly_on_a_missed_partition(run, tmp_path):
    from datetime import datetime

    from fathom.ids import normalize_table
    from fathom.store import ShadowObservation, Store

    store_path = tmp_path / "state" / "fathom.db"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with Store(store_path) as store:
        store.record_shadow(
            ShadowObservation(
                dataset=normalize_table("gold.monthly", system="duckdb"),
                observed=datetime.now(UTC),
                planned=1,
                actual=2,
                missed=1,
                total=10,
            )
        )

    result = run("shadow")
    assert result.exit_code == 1
    assert "SOUNDNESS FAILURE" in result.output
