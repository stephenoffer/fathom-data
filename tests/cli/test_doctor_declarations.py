"""`doctor` over the declared blocks, which fail by being silently vacuous.

Everything in `fathom.yml` that is declared rather than discovered can point at
nothing and still parse. A contract on a never-profiled dataset reports "met" forever;
a publication whose input left the graph names the wrong blast radius in a restatement
notice. Neither raises, which is exactly why `doctor` has to say so.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from fathom.cli import main
from fathom.cli.config import parse_config
from fathom.cli.project import Project
from fathom.core.partitions import PartitionMapping
from fathom.core.types import UNPARTITIONED, DatasetId
from fathom.graph import Edge, Graph
from fathom.observe.profile import ColumnProfile, Profile
from fathom.store.sqlite import Store

ORDERS = DatasetId("duckdb", "gold.orders")
RAW = DatasetId("duckdb", "raw.events")

BASE = """\
version: 1
store: .fathom/fathom.db
system: duckdb
datasets:
  - name: raw.events
  - name: gold.orders
"""


def project_for(body: str, tmp_path: Path, *, store: Store) -> Project:
    return Project(config=parse_config(yaml.safe_load(body), root=tmp_path), store=store)


@pytest.fixture
def store():
    with Store(":memory:") as s:
        # A minimal real graph, so "not in the graph" findings are about the
        # declaration rather than about an empty store.
        graph = Graph()
        graph.add_edge(Edge(RAW, ORDERS, PartitionMapping.unknown(UNPARTITIONED), evidence="sql:1"))
        s.save_graph(graph)
        yield s


def problems_of(project: Project) -> list[str]:
    return project.doctor()


# -- publications --------------------------------------------------------------


def test_a_publication_whose_input_is_missing_is_reported(tmp_path, store):
    body = (
        BASE
        + """
publications:
  - name: exec
    kind: dashboard
    inputs: [nowhere.missing]
"""
    )
    found = problems_of(project_for(body, tmp_path, store=store))
    assert any("publication 'exec'" in p and "not in the graph" in p for p in found)


def test_a_publication_with_real_inputs_is_not_reported(tmp_path, store):
    body = (
        BASE
        + """
publications:
  - name: exec
    kind: dashboard
    inputs: [gold.orders]
"""
    )
    found = problems_of(project_for(body, tmp_path, store=store))
    assert not any("publication" in p for p in found)


def test_the_finding_names_the_missing_input(tmp_path, store):
    body = (
        BASE
        + """
publications:
  - name: exec
    kind: filing
    inputs: [gold.orders, nowhere.missing]
"""
    )
    found = [p for p in problems_of(project_for(body, tmp_path, store=store)) if "publication" in p]
    assert len(found) == 1
    assert "nowhere.missing" in found[0]
    assert "gold.orders" not in found[0]


# -- contracts -----------------------------------------------------------------


def test_a_contract_on_an_unknown_dataset_is_reported(tmp_path, store):
    body = (
        BASE
        + """
contracts:
  - dataset: nowhere.missing
    producer: platform
    consumers: [finance]
"""
    )
    found = problems_of(project_for(body, tmp_path, store=store))
    assert any("not in the graph" in p and "nothing this contract promises" in p for p in found)


def test_a_contract_with_no_consumers_is_reported(tmp_path, store):
    """A breach with no consumer is only a warning, and escalates to nobody."""
    body = (
        BASE
        + """
contracts:
  - dataset: gold.orders
    producer: platform
"""
    )
    found = problems_of(project_for(body, tmp_path, store=store))
    assert any("no consumers named" in p for p in found)


def test_a_contract_on_a_never_profiled_dataset_is_reported(tmp_path, store):
    """It would report 'unchecked' on every run, which reads as passing."""
    body = (
        BASE
        + """
contracts:
  - dataset: gold.orders
    producer: platform
    consumers: [finance]
    columns: [order_id]
"""
    )
    found = problems_of(project_for(body, tmp_path, store=store))
    assert any("never profiled" in p for p in found)


def test_a_profiled_dataset_with_a_full_contract_is_clean(tmp_path, store):
    store.save_profile(
        Profile(dataset=ORDERS, row_count=1, columns=(ColumnProfile("order_id", "string"),)),
        captured=datetime.now(UTC),
    )
    body = (
        BASE
        + """
contracts:
  - dataset: gold.orders
    producer: platform
    consumers: [finance]
    columns: [order_id]
"""
    )
    found = problems_of(project_for(body, tmp_path, store=store))
    assert not any("contract" in p for p in found)


def test_a_contract_promising_nothing_measurable_needs_no_profile(tmp_path, store):
    """Only column and staleness promises need one; a bare ownership record does not."""
    body = (
        BASE
        + """
contracts:
  - dataset: gold.orders
    producer: platform
    consumers: [finance]
"""
    )
    found = problems_of(project_for(body, tmp_path, store=store))
    assert not any("never profiled" in p for p in found)


def test_an_unknown_dataset_is_reported_once_not_twice(tmp_path, store):
    """The later checks are skipped, because they would all say the same thing."""
    body = (
        BASE
        + """
contracts:
  - dataset: nowhere.missing
    producer: platform
    columns: [x]
"""
    )
    found = [p for p in problems_of(project_for(body, tmp_path, store=store)) if "nowhere" in p]
    assert len(found) == 1


def test_no_declarations_produce_no_declaration_findings(tmp_path, store):
    found = problems_of(project_for(BASE, tmp_path, store=store))
    assert not any("contract" in p or "publication" in p for p in found)


# -- through the command -------------------------------------------------------


def test_doctor_surfaces_a_vacuous_contract(tmp_path):
    (tmp_path / ".fathom").mkdir()
    (tmp_path / "fathom.yml").write_text(
        BASE
        + """
contracts:
  - dataset: gold.orders
    producer: platform
"""
    )
    with Store(tmp_path / ".fathom" / "fathom.db") as store:
        graph = Graph()
        graph.add_edge(Edge(RAW, ORDERS, PartitionMapping.unknown(UNPARTITIONED), evidence="sql:1"))
        store.save_graph(graph)

    result = CliRunner().invoke(
        main, ["--config", str(tmp_path / "fathom.yml"), "doctor"], catch_exceptions=False
    )
    assert result.exit_code == 0
    assert "no consumers named" in result.output
