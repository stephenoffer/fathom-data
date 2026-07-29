"""Seasonal baselines end to end: profile history in, a checked baseline out.

The chain that matters here is that observations are bucketed by the *partition's*
moment rather than by when profiling ran. Bucketing by run time puts a Monday
partition backfilled on Saturday into the Saturday band, which is how a backfill
starts failing its own checks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from fathom.cli import main
from fathom.core.types import DatasetId, KeyPredicate
from fathom.observe.profile import ColumnProfile, Profile
from fathom.store.sqlite import Store

EVENTS = DatasetId("duckdb", "raw.events")

CONFIG = """\
version: 1
store: .fathom/fathom.db
system: duckdb
datasets:
  - name: raw.events
    partition: [{field: dt, grain: day}]
"""

# A Monday, so weekday offsets line up.
MONDAY = datetime(2026, 3, 2)


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


def seed(
    store: Store,
    *,
    weeks: int = 5,
    weekday: int = 1000,
    weekend: int = 200,
    extra_weekdays: int = 0,
) -> None:
    """Weeks of daily profiles with a weekday/weekend split.

    `extra_weekdays` adds further Monday-to-Friday observations without adding
    weekend ones, so a `--min-observations` between the two counts leaves the weekend
    buckets unmodelled while the weekday buckets are modelled — which is the mixed
    state the reporting has to distinguish.
    """
    days = [(w, d) for w in range(weeks) for d in range(7)]
    days += [(weeks + w, d) for w in range(extra_weekdays) for d in range(5)]
    for week, offset in days:
        when = MONDAY + timedelta(days=week * 7 + offset)
        rows = weekend if offset >= 5 else weekday
        # A little jitter inside the bucket, uncorrelated with the weekday, so a band
        # has width without the cycle explaining it.
        rows += (week * 7 + offset) % 3
        store.save_profile(
            Profile(
                dataset=EVENTS,
                partition=KeyPredicate.of(dt=when),
                row_count=rows,
                columns=(ColumnProfile("amount", "double", row_count=rows, min=0, max=10),),
            ),
            # Deliberately *not* the partition's own moment, so a test that
            # accidentally buckets by capture time fails.
            captured=datetime(2026, 5, 1, tzinfo=UTC),
        )


# -- the store seam ------------------------------------------------------------


def test_observations_are_dated_by_the_partition_not_the_capture(project):
    """Every profile here was captured on the same day; the buckets must still differ."""
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store)
        history = store.seasonal_observations(EVENTS)
        assert len({o.when for o in history}) == 35


def test_a_profile_with_no_dated_partition_is_skipped(project):
    """A whole-dataset profile has no position in a weekly cycle."""
    with Store(project / ".fathom" / "fathom.db") as store:
        store.save_profile(Profile(dataset=EVENTS, row_count=5), captured=datetime.now(UTC))
        assert store.seasonal_observations(EVENTS) == []


def test_the_time_field_can_be_named_explicitly(project):
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store, weeks=1)
        assert store.seasonal_observations(EVENTS, field_name="dt")
        assert store.seasonal_observations(EVENTS, field_name="absent") == []


def test_an_unprofiled_dataset_has_no_observations(project):
    with Store(project / ".fathom" / "fathom.db") as store:
        assert store.seasonal_observations(EVENTS) == []


# -- the command ---------------------------------------------------------------


def test_seasonal_learns_a_band_per_weekday(run, project):
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store)
    result = run("seasonal", "--dataset", "raw.events")
    assert result.exit_code == 0
    assert "Seasonal baseline" in result.output
    assert "Mon" in result.output and "Sat" in result.output


def test_seasonal_reports_how_much_the_cycle_explains(run, project):
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store)
    result = run("seasonal", "--dataset", "raw.events")
    assert "seasonality:" in result.output
    assert "explained by day_of_week" in result.output


def test_flat_data_is_told_to_use_the_simpler_tool(run, project):
    """Reaching for this over `fathom check` should stay a decision."""
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store, weekday=1000, weekend=1000)
    result = run("seasonal", "--dataset", "raw.events")
    assert "Low." in result.output
    assert "better tool here" in result.output


def test_unmodelled_buckets_are_named_and_flagged_as_unchecked(run, project):
    """Silence on a bucket has to mean unmodelled, never passing."""
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store, weeks=5, extra_weekdays=2)
    result = run("seasonal", "--dataset", "raw.events", "--min-observations", "6")
    assert "Not modelled" in result.output
    assert "not checked" in result.output
    assert "Sat" in result.output


def test_a_baseline_with_nothing_modelled_says_so(run, project):
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store, weeks=2)
    result = run("seasonal", "--dataset", "raw.events", "--min-observations", "4")
    assert result.exit_code == 0
    assert "none reaching the minimum" in result.output


def test_seasonal_refuses_without_dated_profiles(run):
    result = run("seasonal", "--dataset", "raw.events")
    assert result.exit_code != 0
    assert "no partition-dated profiles" in result.output


@pytest.mark.parametrize("cycle", ["day_of_week", "hour_of_day", "day_of_month", "month_of_year"])
def test_every_cycle_is_accepted(run, project, cycle):
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store)
    assert run("seasonal", "--dataset", "raw.events", "--cycle", cycle).exit_code == 0


def test_an_unknown_cycle_is_rejected(run, project):
    with Store(project / ".fathom" / "fathom.db") as store:
        seed(store)
    assert run("seasonal", "--dataset", "raw.events", "--cycle", "fortnight").exit_code != 0
