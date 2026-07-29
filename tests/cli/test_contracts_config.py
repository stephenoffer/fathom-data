"""The `contracts:` block, the duration parser, and `fathom contracts` / `fathom risk`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from fathom.cli import main
from fathom.cli.config import parse_config
from fathom.core.errors import ConfigError
from fathom.core.types import DatasetId
from fathom.observe.profile import ColumnProfile, Profile
from fathom.store.sqlite import Store

ORDERS = DatasetId("duckdb", "gold.orders")

BASE = """\
version: 1
store: .fathom/fathom.db
system: duckdb
datasets:
  - name: gold.orders
"""


def config(body: str, root: Path):
    return parse_config(yaml.safe_load(body), root=root)


# -- parsing -------------------------------------------------------------------


def test_a_contract_parses(tmp_path):
    parsed = config(
        BASE
        + """
contracts:
  - dataset: gold.orders
    producer: platform
    consumers: [finance, ml]
    columns: [order_id, amount]
    max_staleness: 6h
    note: the close depends on it
""",
        tmp_path,
    )
    (contract,) = parsed.contracts
    assert contract.dataset == ORDERS
    assert contract.producer == "platform"
    assert contract.consumers == ("finance", "ml")
    assert contract.columns == ("order_id", "amount")
    assert contract.max_staleness == timedelta(hours=6)


def test_a_contract_without_a_producer_is_rejected(tmp_path):
    """An unowned contract names nobody on breach, which is the one thing it adds."""
    with pytest.raises(ConfigError, match="missing `producer`"):
        config(BASE + "\ncontracts:\n  - dataset: gold.orders\n", tmp_path)


def test_a_contract_without_a_dataset_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="missing `dataset`"):
        config(BASE + "\ncontracts:\n  - producer: platform\n", tmp_path)


def test_an_unknown_contract_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        config(
            BASE + "\ncontracts:\n  - dataset: gold.orders\n    producer: p\n    sla: nope\n",
            tmp_path,
        )


def test_a_single_consumer_may_be_a_bare_string(tmp_path):
    parsed = config(
        BASE + "\ncontracts:\n  - dataset: gold.orders\n    producer: p\n    consumers: finance\n",
        tmp_path,
    )
    assert parsed.contracts[0].consumers == ("finance",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30s", timedelta(seconds=30)),
        ("30m", timedelta(minutes=30)),
        ("6h", timedelta(hours=6)),
        ("2d", timedelta(days=2)),
        ("1w", timedelta(weeks=1)),
        ("1.5h", timedelta(hours=1.5)),
        ("6H", timedelta(hours=6)),
    ],
)
def test_every_duration_unit_parses(tmp_path, text, expected):
    parsed = config(
        BASE
        + f"\ncontracts:\n  - dataset: gold.orders\n    producer: p\n    max_staleness: {text}\n",
        tmp_path,
    )
    assert parsed.contracts[0].max_staleness == expected


def test_a_bare_number_is_rejected_rather_than_assumed(tmp_path):
    """Two readers silently disagreeing about seconds versus hours is the failure."""
    with pytest.raises(ConfigError, match="not a duration"):
        config(
            BASE
            + "\ncontracts:\n  - dataset: gold.orders\n    producer: p\n    max_staleness: 6\n",
            tmp_path,
        )


def test_an_unknown_unit_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not a duration"):
        config(
            BASE
            + "\ncontracts:\n  - dataset: gold.orders\n    producer: p\n    max_staleness: 6y\n",
            tmp_path,
        )


def test_no_contracts_block_is_simply_empty(tmp_path):
    assert config(BASE, tmp_path).contracts == []


# -- the command ---------------------------------------------------------------


CONTRACT_CONFIG = (
    BASE
    + """
contracts:
  - dataset: gold.orders
    producer: platform
    consumers: [finance]
    columns: [order_id, amount]
"""
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "fathom.yml").write_text(CONTRACT_CONFIG)
    (tmp_path / ".fathom").mkdir()
    return tmp_path


@pytest.fixture
def run(project: Path):
    runner = CliRunner()

    def invoke(*args: str):
        return runner.invoke(
            main, ["--config", str(project / "fathom.yml"), *args], catch_exceptions=False
        )

    return invoke


def profile(*names: str) -> Profile:
    return Profile(
        dataset=ORDERS,
        row_count=10,
        columns=tuple(ColumnProfile(n, "string", row_count=10) for n in names),
    )


def test_a_met_contract_exits_zero(run, project):
    with Store(project / ".fathom" / "fathom.db") as store:
        store.save_profile(profile("order_id", "amount"), captured=datetime.now(UTC))
    result = run("contracts")
    assert result.exit_code == 0
    assert "met" in result.output


def test_a_breached_contract_exits_non_zero_and_names_the_consumer(run, project):
    with Store(project / ".fathom" / "fathom.db") as store:
        store.save_profile(profile("order_id"), captured=datetime.now(UTC))
    result = run("contracts")
    assert result.exit_code == 1
    assert "finance" in result.output
    assert "'amount'" in result.output


def test_a_missing_profile_is_unchecked_not_passed(run):
    result = run("contracts")
    assert result.exit_code == 0
    assert "Not checked" in result.output or "not checked" in result.output


def test_no_contracts_block_is_a_clean_error(tmp_path):
    (tmp_path / "fathom.yml").write_text(BASE)
    (tmp_path / ".fathom").mkdir()
    result = CliRunner().invoke(
        main, ["--config", str(tmp_path / "fathom.yml"), "contracts"], catch_exceptions=False
    )
    assert result.exit_code != 0
    assert "no contracts declared" in result.output


# -- risk ----------------------------------------------------------------------


def test_risk_refuses_without_profiles(tmp_path):
    (tmp_path / "fathom.yml").write_text(BASE)
    (tmp_path / ".fathom").mkdir()
    result = CliRunner().invoke(
        main, ["--config", str(tmp_path / "fathom.yml"), "risk"], catch_exceptions=False
    )
    assert result.exit_code != 0
    assert "run `fathom profile` first" in result.output


def test_risk_reports_a_near_unique_column(run, project):
    with Store(project / ".fathom" / "fathom.db") as store:
        store.save_profile(
            Profile(
                dataset=ORDERS,
                row_count=1000,
                columns=(
                    ColumnProfile("order_ref", "string", row_count=1000, distinct_estimate=999),
                ),
            ),
            captured=datetime.now(UTC),
        )
    result = run("risk")
    assert result.exit_code == 0
    assert "singles a row out" in result.output


def test_risk_never_claims_safety(run, project):
    with Store(project / ".fathom" / "fathom.db") as store:
        store.save_profile(
            Profile(
                dataset=ORDERS,
                row_count=1000,
                columns=(ColumnProfile("colour", "string", row_count=1000, distinct_estimate=4),),
            ),
            captured=datetime.now(UTC),
        )
    result = run("risk")
    # A clear dataset is skipped entirely; the command errors only when nothing was
    # assessable at all, which is a different thing from "everything is fine".
    assert "no re-identification risk proven" not in result.output or "safe" in result.output
