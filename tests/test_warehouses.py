"""Snowflake, Databricks, and BigQuery adapters.

Driven by `RecordedRunner` with rows shaped the way each platform actually returns
them — Snowflake uppercases column names and hands back `ACCESS_HISTORY` arrays as
either JSON text or parsed lists; BigQuery encodes partition grain in the id format.
A warehouse adapter whose only test path is a live account has no tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from fathom.adapters.bigquery import BigQueryAdapter, grain_of, parse_partition_id
from fathom.adapters.databricks import DatabricksAdapter
from fathom.adapters.predicates import render_predicate
from fathom.adapters.snowflake import SnowflakeAdapter
from fathom.adapters.sql_runner import DBAPIRunner, QueryError, RecordedRunner, quote_identifier
from fathom.errors import ConfigError
from fathom.grains import Grain
from fathom.ids import normalize_table
from fathom.types import ANY, KeyPredicate, PartitionField, PartitionSpec

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))


# -- the runner ----------------------------------------------------------------


class FakeCursor:
    def __init__(self, rows, columns, fail=None):
        self._rows, self._columns, self._fail = rows, columns, fail
        self.description = [(c,) for c in columns]
        self.closed = False

    def execute(self, sql, params=None):
        if self._fail:
            raise self._fail

    def fetchall(self):
        return self._rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, rows, columns, fail=None):
        self._cursor = FakeCursor(rows, columns, fail)

    def cursor(self):
        return self._cursor


def test_dbapi_runner_returns_dicts():
    runner = DBAPIRunner(FakeConnection([(1, "a")], ["ID", "NAME"]))
    assert runner.rows("SELECT 1") == [{"ID": 1, "NAME": "a"}]


def test_dbapi_runner_closes_the_cursor_even_on_failure():
    connection = FakeConnection([], [], fail=RuntimeError("boom"))
    with pytest.raises(QueryError):
        DBAPIRunner(connection).rows("SELECT 1")
    assert connection._cursor.closed


def test_query_errors_carry_the_statement():
    connection = FakeConnection([], [], fail=RuntimeError("syntax error"))
    with pytest.raises(QueryError) as caught:
        DBAPIRunner(connection).rows("SELECT * FROM nope")
    assert "SELECT * FROM nope" in str(caught.value)


def test_row_caps_are_enforced():
    runner = DBAPIRunner(FakeConnection([(i,) for i in range(10)], ["N"]), max_rows=5)
    with pytest.raises(QueryError, match="over the 5 rows cap|over the 5 cap"):
        runner.rows("SELECT 1")


def test_identifiers_are_validated_not_interpolated_blindly():
    """Identifiers cannot be parameterized, so injection is checked at the boundary."""
    assert quote_identifier("user_id") == '"user_id"'
    with pytest.raises(ValueError, match="not a valid identifier"):
        quote_identifier("id; DROP TABLE users--")


def test_recorded_runner_reports_unmatched_queries():
    with pytest.raises(QueryError, match="no recorded response matched"):
        RecordedRunner({"SELECT a": [{"a": 1}]}).rows("SELECT b FROM t")


# -- Snowflake -----------------------------------------------------------------


def access_history_row(*, as_json: bool = False):
    modified = [
        {
            "objectName": "PROD.GOLD.MONTHLY",
            "columns": [
                {
                    "columnName": "REVENUE",
                    "directSources": [{"objectName": "PROD.SILVER.EVENTS", "columnName": "AMOUNT"}],
                }
            ],
        }
    ]
    accessed = [{"objectName": "PROD.SILVER.EVENTS"}, {"objectName": "PROD.DIM.REGION"}]
    return {
        "QUERY_ID": "q1",
        "QUERY_START_TIME": datetime(2026, 3, 14, 12, tzinfo=UTC),
        "OBJECTS_MODIFIED": json.dumps(modified) if as_json else modified,
        "BASE_OBJECTS_ACCESSED": json.dumps(accessed) if as_json else accessed,
    }


@pytest.mark.parametrize("as_json", [False, True])
def test_snowflake_column_lineage_from_access_history(as_json):
    """Drivers differ on whether variant columns arrive parsed or as JSON text."""
    runner = RecordedRunner({"access_history": [access_history_row(as_json=as_json)]})
    events = list(SnowflakeAdapter(runner=runner, account="ac1").fetch_lineage(None))

    by_source = {e.src.name: e for e in events}
    assert by_source["PROD.SILVER.EVENTS"].columns == (("AMOUNT", "REVENUE"),)
    assert by_source["PROD.SILVER.EVENTS"].dst.name == "PROD.GOLD.MONTHLY"


def test_snowflake_keeps_tables_read_but_not_projected():
    """A join key or filter source is a real dependency with no column edge."""
    runner = RecordedRunner({"access_history": [access_history_row()]})
    events = list(SnowflakeAdapter(runner=runner, account="ac1").fetch_lineage(None))
    dimension = next(e for e in events if e.src.name == "PROD.DIM.REGION")
    assert dimension.columns == ()


def test_snowflake_token_is_held_back_by_the_account_usage_lag():
    """Advancing to the newest row would permanently skip rows still in flight."""
    runner = RecordedRunner({"access_history": [access_history_row()]})
    adapter = SnowflakeAdapter(runner=runner, account="ac1")
    events = list(adapter.fetch_lineage(None))
    token = adapter.lineage_token(events)
    assert token == "2026-03-14 09:00:00"  # 12:00 minus the three-hour lag


def test_snowflake_watermark_gives_partition_granularity():
    table = normalize_table("prod.raw.events", system="snowflake", instance="ac1")
    runner = RecordedRunner(
        {
            "SELECT DISTINCT": [
                {
                    "DT": datetime(2026, 3, 14, 8, tzinfo=UTC),
                    "_FATHOM_HIGH_WATER": datetime(2026, 3, 14, 9, tzinfo=UTC),
                }
            ]
        }
    )
    adapter = SnowflakeAdapter(runner=runner, account="ac1")
    adapter.declare(table, DAY, watermark="updated_at")

    changes = adapter.changed(table, None)
    assert changes.partitions == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 14))})
    assert changes.token == "2026-03-14 09:00:00"


def test_snowflake_without_a_watermark_falls_back_to_whole_table():
    """Honest and coarse beats precise and invented."""
    table = normalize_table("prod.raw.events", system="snowflake", instance="ac1")
    runner = RecordedRunner({"last_altered": [{"LAST_ALTERED": datetime(2026, 3, 14, tzinfo=UTC)}]})
    adapter = SnowflakeAdapter(runner=runner, account="ac1")
    adapter.declare(table, DAY)

    changes = adapter.changed(table, None)
    assert changes.partitions == frozenset({KeyPredicate.of(dt=ANY)})


def test_snowflake_unchanged_table_reports_nothing():
    table = normalize_table("prod.raw.events", system="snowflake", instance="ac1")
    runner = RecordedRunner({"last_altered": [{"LAST_ALTERED": datetime(2026, 3, 14, tzinfo=UTC)}]})
    adapter = SnowflakeAdapter(runner=runner, account="ac1")
    adapter.declare(table, DAY)
    first = adapter.changed(table, None)
    assert adapter.changed(table, first.token).is_empty


def test_snowflake_needs_a_runner_and_says_how_to_supply_one():
    with pytest.raises(ConfigError, match="DBAPIRunner"):
        list(SnowflakeAdapter().fetch_lineage(None))


def test_snowflake_rejects_partially_qualified_names():
    adapter = SnowflakeAdapter(runner=RecordedRunner({}), account="ac1")
    partial = normalize_table("events", system="snowflake", instance="ac1")
    with pytest.raises(ConfigError, match="fully qualified"):
        adapter.changed(partial, None)


# -- Databricks ----------------------------------------------------------------


def test_databricks_column_lineage_from_unity_catalog():
    runner = RecordedRunner(
        {
            "system.access.column_lineage": [
                {
                    "source_table_full_name": "main.silver.events",
                    "target_table_full_name": "main.gold.monthly",
                    "source_column_name": "amount",
                    "target_column_name": "revenue",
                    "event_time": datetime(2026, 3, 14, 12, tzinfo=UTC),
                }
            ],
            "system.access.table_lineage": [],
        }
    )
    events = list(DatabricksAdapter(runner=runner, workspace="ws1").fetch_lineage(None))
    assert events[0].columns == (("amount", "revenue"),)
    assert events[0].src.name == "main.silver.events"


def test_databricks_table_lineage_catches_edges_column_lineage_misses():
    runner = RecordedRunner(
        {
            "system.access.column_lineage": [],
            "system.access.table_lineage": [
                {
                    "source_table_full_name": "main.dim.region",
                    "target_table_full_name": "main.gold.monthly",
                    "event_time": datetime(2026, 3, 14, tzinfo=UTC),
                }
            ],
        }
    )
    events = list(DatabricksAdapter(runner=runner, workspace="ws1").fetch_lineage(None))
    assert len(events) == 1
    assert events[0].columns == ()


def test_databricks_partition_spec_from_describe_detail():
    runner = RecordedRunner(
        {
            "DESCRIBE DETAIL": [{"partitionColumns": ["dt", "region"], "location": "s3://lake/t"}],
            "DESCRIBE TABLE": [
                {"col_name": "dt", "data_type": "date"},
                {"col_name": "region", "data_type": "string"},
                {"col_name": "# Partitioning", "data_type": ""},
            ],
        }
    )
    adapter = DatabricksAdapter(runner=runner, workspace="ws1")
    table = normalize_table("main.silver.events", system="databricks", instance="ws1")
    spec = adapter.describe_partitioning(table)

    assert spec.field("dt").grain is Grain.DAY
    assert spec.field("region").kind == "value"


def test_databricks_delegates_change_detection_to_the_delta_log(tmp_path):
    """Reusing the Delta adapter beats a second, weaker implementation."""
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

    runner = RecordedRunner(
        {
            "DESCRIBE DETAIL": [{"partitionColumns": ["dt"], "location": str(root.resolve())}],
            "DESCRIBE TABLE": [{"col_name": "dt", "data_type": "date"}],
        }
    )
    adapter = DatabricksAdapter(runner=runner, workspace="ws1")
    table = normalize_table("main.silver.events", system="databricks", instance="ws1")

    changes = adapter.changed(table, None)
    assert changes.partitions == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 14))})


def test_databricks_without_a_location_says_why_it_cannot_detect_changes():
    runner = RecordedRunner({"DESCRIBE DETAIL": [{"partitionColumns": [], "location": ""}]})
    adapter = DatabricksAdapter(runner=runner, workspace="ws1")
    table = normalize_table("main.silver.events", system="databricks", instance="ws1")
    with pytest.raises(ConfigError, match="cannot locate storage"):
        adapter.changed(table, None)


def test_databricks_rebuilds_use_backtick_quoting():
    adapter = DatabricksAdapter(runner=RecordedRunner({}), workspace="ws1")
    table = normalize_table("main.gold.monthly", system="databricks", instance="ws1")
    adapter.register_model(table, "SELECT dt FROM main.silver.events", DAY)
    statements = adapter.render_rebuild(table, [KeyPredicate.of(dt=datetime(2026, 3, 14))])
    assert "`dt`" in statements[0]


# -- BigQuery ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("partition_id", "grain"),
    [
        ("2026", Grain.YEAR),
        ("202603", Grain.MONTH),
        ("20260314", Grain.DAY),
        ("2026031409", Grain.HOUR),
        ("__NULL__", None),
        ("__UNPARTITIONED__", None),
        ("not-a-date", None),
    ],
)
def test_bigquery_grain_is_read_from_the_partition_id(partition_id, grain):
    """The id format is the only place BigQuery records granularity."""
    assert grain_of(partition_id) is grain


def test_bigquery_partition_ids_decode():
    assert parse_partition_id("20260314") == datetime(2026, 3, 14)
    assert parse_partition_id("202603") == datetime(2026, 3, 1)
    assert parse_partition_id("__NULL__") is None


def test_bigquery_change_detection_is_exact():
    runner = RecordedRunner(
        {
            "is_partitioning_column": [{"column_name": "dt", "data_type": "DATE"}],
            "INFORMATION_SCHEMA.PARTITIONS": [
                {
                    "table_name": "events",
                    "partition_id": "20260314",
                    "last_modified_time": datetime(2026, 3, 14, 9, tzinfo=UTC),
                },
                {
                    "table_name": "events",
                    "partition_id": "20260315",
                    "last_modified_time": datetime(2026, 3, 15, 9, tzinfo=UTC),
                },
            ],
        }
    )
    adapter = BigQueryAdapter(runner=runner, project="proj")
    table = normalize_table("proj.raw.events", system="bigquery")

    changes = adapter.changed(table, "2026-03-14T12:00:00+00:00")
    assert changes.partitions == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 15))})


def test_bigquery_null_partition_is_a_real_partition():
    """`__NULL__` holds rows whose partition column is null. Dropping it loses data."""
    runner = RecordedRunner(
        {
            "is_partitioning_column": [{"column_name": "dt", "data_type": "DATE"}],
            "INFORMATION_SCHEMA.PARTITIONS": [
                {
                    "table_name": "events",
                    "partition_id": "__NULL__",
                    "last_modified_time": datetime(2026, 3, 15, tzinfo=UTC),
                }
            ],
        }
    )
    adapter = BigQueryAdapter(runner=runner, project="proj")
    table = normalize_table("proj.raw.events", system="bigquery")
    changes = adapter.changed(table, None)
    assert next(iter(changes.partitions)).get("dt") is None


def test_bigquery_unpartitioned_table_reports_the_whole_dataset():
    runner = RecordedRunner(
        {
            "is_partitioning_column": [],
            "INFORMATION_SCHEMA.PARTITIONS": [
                {
                    "table_name": "events",
                    "partition_id": "__UNPARTITIONED__",
                    "last_modified_time": datetime(2026, 3, 15, tzinfo=UTC),
                }
            ],
        }
    )
    adapter = BigQueryAdapter(runner=runner, project="proj")
    table = normalize_table("proj.raw.events", system="bigquery")
    assert adapter.changed(table, None).partitions == frozenset({KeyPredicate()})


def test_bigquery_referenced_tables_give_lineage_without_parsing():
    runner = RecordedRunner(
        {
            "INFORMATION_SCHEMA.JOBS": [
                {
                    "job_id": "j1",
                    "query": "SELECT 1",
                    "creation_time": datetime(2026, 3, 14, tzinfo=UTC),
                    "destination_table": {
                        "project_id": "proj",
                        "dataset_id": "gold",
                        "table_id": "monthly",
                    },
                    "referenced_tables": [
                        {"project_id": "proj", "dataset_id": "silver", "table_id": "events"}
                    ],
                }
            ]
        }
    )
    events = list(BigQueryAdapter(runner=runner, project="proj").fetch_lineage(None))
    assert events[0].src.name == "proj.silver.events"
    assert events[0].dst.name == "proj.gold.monthly"


def test_bigquery_rejects_names_it_cannot_qualify():
    adapter = BigQueryAdapter(runner=RecordedRunner({}))
    with pytest.raises(ConfigError, match="project.dataset.table"):
        adapter.changed(normalize_table("events", system="bigquery"), None)


# -- shared predicate rendering ------------------------------------------------


def test_typed_literals_are_ansi_and_portable():
    sql = render_predicate(DAY, [KeyPredicate.of(dt=datetime(2026, 3, 14))])
    assert "TIMESTAMP '2026-03-14 00:00:00'" in sql


def test_quoting_is_configurable_per_dialect():
    keys = [KeyPredicate.of(dt=datetime(2026, 3, 14))]
    assert '"dt"' in render_predicate(DAY, keys)
    assert "`dt`" in render_predicate(DAY, keys, quote="`")


def test_null_partitions_render_as_is_null():
    spec = PartitionSpec.of(PartitionField.value("region"))
    assert "IS NULL" in render_predicate(spec, [KeyPredicate(bindings=(("region", None),))])
