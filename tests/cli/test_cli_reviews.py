"""The four commands over the newer artifacts, and the exit codes CI depends on.

`completeness` and `impact` exit non-zero on a finding, because both describe a
condition somebody has to act on. `usage` and `value` do not, because "these datasets
look quiet" is a review list and failing a build over it would train people to pass
`|| true`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from fathom.cli import main
from fathom.core.partitions import PartitionMapping
from fathom.core.types import UNPARTITIONED, DatasetId, KeyPredicate
from fathom.graph import sinks
from fathom.graph.model import Edge, Graph
from fathom.graph.plan.lifetime import RunRecord
from fathom.observe.completeness import Arrival
from fathom.observe.usage import ReadEvent
from fathom.store.sqlite import Store

CONFIG = """\
version: 1
store: .fathom/fathom.db
system: duckdb

datasets:
  - name: raw.events
    partition: [{field: dt, grain: day}]
  - name: gold.monthly
    partition: [{field: dt, grain: month}]
  - name: gold.flat
"""

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.monthly")
FLAT = DatasetId("duckdb", "gold.flat")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "fathom.yml").write_text(CONFIG)
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


@pytest.fixture
def store(project: Path):
    with Store(project / ".fathom" / "fathom.db") as s:
        yield s


def day(n: int) -> KeyPredicate:
    return KeyPredicate.of(dt=datetime(2026, 3, n))


# -- completeness --------------------------------------------------------------


def test_completeness_reports_a_gap_and_exits_non_zero(run, store):
    for n in (1, 2, 5):
        store.record_arrival(Arrival(RAW, day(n), datetime(2026, 3, n, tzinfo=UTC)))
    store.close()

    result = run(
        "completeness", "--dataset", "raw.events", "--since", "2026-03-01", "--until", "2026-03-05"
    )
    assert result.exit_code == 1
    assert "incomplete" in result.output
    assert "2 of 5" in result.output


def test_a_complete_dataset_exits_zero(run, store):
    for n in range(1, 6):
        store.record_arrival(Arrival(RAW, day(n), datetime(2026, 3, n, tzinfo=UTC)))
    store.close()

    result = run(
        "completeness", "--dataset", "raw.events", "--since", "2026-03-01", "--until", "2026-03-05"
    )
    assert result.exit_code == 0
    assert "complete" in result.output


def test_completeness_refuses_without_recorded_arrivals(run):
    result = run(
        "completeness", "--dataset", "raw.events", "--since", "2026-03-01", "--until", "2026-03-05"
    )
    assert result.exit_code != 0
    assert "no arrivals recorded" in result.output


def test_completeness_refuses_on_an_unpartitioned_dataset(run, store):
    store.record_arrival(Arrival(FLAT, KeyPredicate(), datetime(2026, 3, 1, tzinfo=UTC)))
    store.close()

    result = run(
        "completeness", "--dataset", "gold.flat", "--since", "2026-03-01", "--until", "2026-03-05"
    )
    assert result.exit_code != 0
    assert "no partition spec" in result.output


def test_an_oversized_range_is_a_clean_error_not_a_traceback(run, store):
    store.record_arrival(Arrival(RAW, day(1), datetime(2026, 3, 1, tzinfo=UTC)))
    store.close()

    result = run(
        "completeness", "--dataset", "raw.events", "--since", "1000-01-01", "--until", "2026-03-05"
    )
    assert result.exit_code != 0
    # Either bound may fire first — `grains.span`'s walk limit or `expected_keys`'
    # own ceiling — and either may reword its message. The claim under test is that
    # an oversized range is refused cleanly rather than raising, so this asserts the
    # shape of the failure and not the prose, which belongs to those modules.
    assert result.output.startswith("Error:")
    assert "Traceback" not in result.output


# -- usage ---------------------------------------------------------------------


def test_usage_ranks_the_datasets_read(run, store):
    now = datetime.now(UTC)
    store.record_reads([ReadEvent(GOLD, "ana", now), ReadEvent(GOLD, "ben", now)])
    store.record_read(ReadEvent(RAW, "ana", now))
    store.close()

    result = run("usage")
    assert result.exit_code == 0
    assert result.output.index("gold.monthly") < result.output.index("raw.events")


def test_usage_refuses_when_nothing_was_recorded(run):
    result = run("usage")
    assert result.exit_code != 0
    assert "not evidence a dataset is unused" in result.output


def test_usage_respects_the_window(run, store):
    store.record_read(ReadEvent(GOLD, "ana", datetime(2020, 1, 1, tzinfo=UTC)))
    store.close()

    result = run("usage", "--days", "30")
    assert result.exit_code != 0  # nothing inside the window


def test_retire_keeps_the_review_list_caveat(run, store):
    store.record_read(ReadEvent(GOLD, "ana", datetime.now(UTC)))
    store.close()

    result = run("usage", "--retire")
    assert result.exit_code == 0
    assert "review list, not a delete list" in result.output


# -- value ---------------------------------------------------------------------


def test_value_flags_the_unread_and_expensive(run, store):
    now = datetime.now(UTC)
    store.record_run(RunRecord(GOLD, now, partitions=100))
    store.record_run(RunRecord(RAW, now, partitions=100))
    store.record_read(ReadEvent(RAW, "ana", now))
    store.close()

    result = run("value", "--threshold", "50", "--price-per-partition", "1.0")
    assert result.exit_code == 0
    assert "1 dataset(s) unread and above the threshold" in result.output
    assert "gold.monthly" in result.output


def test_value_states_the_measured_versus_observed_asymmetry(run, store):
    store.record_run(RunRecord(GOLD, datetime.now(UTC), partitions=100))
    store.close()

    result = run("value", "--threshold", "10", "--price-per-partition", "1.0")
    assert "read once a year for a" in result.output


def test_value_refuses_without_recorded_runs(run):
    result = run("value", "--threshold", "10")
    assert result.exit_code != 0
    assert "no runs recorded" in result.output


def test_the_threshold_is_required(run):
    runner = CliRunner()
    result = runner.invoke(main, ["value"], catch_exceptions=False)
    assert result.exit_code != 0


# -- impact --------------------------------------------------------------------


def _graph_with_a_filing() -> Graph:
    g = Graph()
    g.add_dataset(RAW, UNPARTITIONED)
    g.add_dataset(GOLD, UNPARTITIONED)
    g.add_edge(Edge(RAW, GOLD, PartitionMapping.unknown(UNPARTITIONED), evidence="declared"))
    sinks.record_publication(g, sinks.filing("10-K/2026", regulator="sec"), [GOLD])
    return g


def test_impact_names_the_artefacts_and_exits_non_zero_on_a_filing(run, store):
    store.save_graph(_graph_with_a_filing())
    store.close()

    result = run("impact", "--dataset", "raw.events", "--reason", "fx rates were wrong")
    assert result.exit_code == 1
    assert "filing 10-K/2026 on sec" in result.output
    assert "fx rates were wrong" in result.output
    assert "formal amendment" in result.output


def test_impact_exits_zero_when_only_a_dashboard_is_downstream(run, store):
    g = Graph()
    g.add_dataset(GOLD, UNPARTITIONED)
    sinks.record_publication(g, sinks.dashboard("exec", tool="looker"), [GOLD])
    store.save_graph(g)
    store.close()

    result = run("impact", "--dataset", "gold.monthly")
    assert result.exit_code == 0
    assert "dashboard exec on looker" in result.output


def test_impact_refuses_for_a_dataset_not_in_the_graph(run, store):
    """A name that is in neither the config nor the store cannot be reasoned about."""
    store.save_graph(Graph())
    store.close()

    result = run("impact", "--dataset", "nowhere.missing")
    assert result.exit_code != 0
    assert "not in the stored graph" in result.output


def test_impact_says_so_when_nothing_is_published(run, store):
    g = Graph()
    g.add_dataset(GOLD, UNPARTITIONED)
    store.save_graph(g)
    store.close()

    result = run("impact", "--dataset", "gold.monthly")
    assert result.exit_code == 0
    assert "nothing has been told to anyone" in result.output


# -- help ----------------------------------------------------------------------


@pytest.mark.parametrize("command", ["completeness", "usage", "value", "impact"])
def test_every_new_command_documents_itself(command):
    result = CliRunner().invoke(main, [command, "--help"])
    assert result.exit_code == 0
    assert len(result.output.splitlines()[2].strip()) > 0
