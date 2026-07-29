"""dbt manifest ingest.

Manifests are built to match what dbt actually writes, including the parts that
differ per adapter: BigQuery's `partition_by` dict versus Spark's list, and
Snowflake's absence of partitioning entirely.
"""

from __future__ import annotations

import json

import pytest

from fathom.core.errors import ConfigError
from fathom.core.grains import Grain
from fathom.core.ids import normalize_table
from fathom.core.partitions import TimeWindow
from fathom.core.types import PartitionField, PartitionSpec
from fathom.ingest import ingest_dbt, load_manifest, parse_manifest


def manifest(
    *,
    adapter: str = "bigquery",
    partition_by=None,
    meta=None,
    compiled: str | None = None,
    materialized: str = "table",
) -> dict:
    return {
        "metadata": {
            "adapter_type": adapter,
            "project_name": "analytics",
            "dbt_version": "1.9.0",
        },
        "sources": {
            "source.analytics.raw.events": {
                "resource_type": "source",
                "database": "prod",
                "schema": "raw",
                "name": "events",
                "config": {"meta": {"fathom": {"partition": [{"field": "dt", "grain": "day"}]}}},
            }
        },
        "nodes": {
            "model.analytics.gold_monthly": {
                "resource_type": "model",
                "database": "prod",
                "schema": "gold",
                "name": "gold_monthly",
                "alias": "monthly",
                "depends_on": {"nodes": ["source.analytics.raw.events"]},
                "config": {
                    "materialized": materialized,
                    "partition_by": partition_by,
                    "meta": meta or {},
                },
                "columns": {"dt": {"name": "dt", "data_type": "date"}},
                "compiled_code": compiled
                or "select date_trunc(dt, MONTH) as dt, sum(amount) as revenue "
                "from prod.raw.events group by 1",
            },
            "test.analytics.not_null": {
                "resource_type": "test",
                "database": "prod",
                "schema": "gold",
                "name": "not_null",
                "depends_on": {"nodes": ["model.analytics.gold_monthly"]},
                "config": {},
            },
        },
    }


def source_id(adapter: str = "bigquery"):
    return normalize_table("prod.raw.events", system=adapter)


def model_id(adapter: str = "bigquery"):
    return normalize_table("prod.gold.monthly", system=adapter)


# -- parsing -------------------------------------------------------------------


def test_resolves_relation_names_through_database_and_schema():
    parsed = parse_manifest(manifest())
    assert model_id() in parsed.datasets
    assert source_id() in parsed.datasets


def test_relation_name_wins_when_dbt_provides_one():
    blob = manifest()
    blob["nodes"]["model.analytics.gold_monthly"]["relation_name"] = "`other`.`x`.`y`"
    parsed = parse_manifest(blob)
    assert normalize_table("other.x.y", system="bigquery") in parsed.datasets


def test_tests_are_not_part_of_the_data_graph():
    """A `not_null` test is not a dependency any rebuild plan cares about."""
    parsed = parse_manifest(manifest())
    assert all("not_null" not in str(d) for d in parsed.datasets)


def test_adapter_type_selects_the_identity_system():
    parsed = parse_manifest(manifest(adapter="snowflake"))
    assert parsed.system == "snowflake"
    # Snowflake folds identifiers up, so the relation must too.
    assert any(str(d).endswith("PROD.GOLD.MONTHLY") for d in parsed.datasets)


def test_spark_maps_onto_the_databricks_identity_system():
    assert parse_manifest(manifest(adapter="spark")).system == "databricks"


# -- partition specs -----------------------------------------------------------


def test_bigquery_partition_by_dict_becomes_a_time_field():
    parsed = parse_manifest(
        manifest(partition_by={"field": "dt", "data_type": "date", "granularity": "month"})
    )
    spec = parsed.specs[model_id()]
    assert spec.field("dt").grain is Grain.MONTH


def test_granularity_falls_back_to_the_column_type():
    parsed = parse_manifest(manifest(partition_by={"field": "dt", "data_type": "date"}))
    assert parsed.specs[model_id()].field("dt").grain is Grain.DAY


def test_spark_partition_by_list_uses_column_types():
    blob = manifest(adapter="spark", partition_by=["dt", "region"])
    blob["nodes"]["model.analytics.gold_monthly"]["columns"]["region"] = {
        "name": "region",
        "data_type": "string",
    }
    parsed = parse_manifest(blob)
    spec = parsed.specs[model_id("databricks")]
    assert spec.field("dt").kind == "time"
    assert spec.field("region").kind == "value"


def test_meta_declaration_overrides_dbt_config():
    """The escape hatch: dbt cannot express grain on Snowflake, so users declare it."""
    parsed = parse_manifest(
        manifest(
            partition_by={"field": "dt", "data_type": "date", "granularity": "day"},
            meta={"fathom": {"partition": [{"field": "dt", "grain": "year"}]}},
        )
    )
    assert parsed.specs[model_id()].field("dt").grain is Grain.YEAR


def test_no_partition_config_means_unpartitioned():
    parsed = parse_manifest(manifest(adapter="snowflake"))
    assert model_id("snowflake") not in parsed.specs


# -- graph building ------------------------------------------------------------


def test_builds_edges_from_depends_on():
    result = ingest_dbt(manifest())
    assert [(e.src, e.dst) for e in result.graph.edges] == [(source_id(), model_id())]
    assert result.graph.edges[0].evidence == "dbt:model.analytics.gold_monthly"


def test_compiled_sql_supplies_the_partition_mapping():
    """dbt gives the skeleton; parsing the compiled SQL fills in the detail."""
    result = ingest_dbt(
        manifest(partition_by={"field": "dt", "data_type": "date", "granularity": "month"})
    )
    mapping = result.graph.edges[0].mapping.get("dt")
    assert isinstance(mapping, TimeWindow)
    assert (mapping.in_grain, mapping.out_grain) == (Grain.DAY, Grain.MONTH)


def test_compiled_sql_supplies_column_edges():
    result = ingest_dbt(
        manifest(partition_by={"field": "dt", "data_type": "date", "granularity": "month"})
    )
    assert ("amount", "revenue") in result.graph.edges[0].columns


def test_parsing_can_be_disabled_for_speed():
    result = ingest_dbt(manifest(), parse_sql=False)
    assert result.graph.edges[0].columns == ()


def test_bare_select_is_wrapped_before_parsing():
    """dbt compiles models to a bare SELECT with no target."""
    result = ingest_dbt(manifest(compiled="select dt, amount from prod.raw.events"))
    assert result.graph.edges


def test_unparseable_compiled_sql_still_yields_the_dbt_edge():
    """dbt's own dependency data must survive a parse failure."""
    result = ingest_dbt(manifest(compiled="{{ this is not sql }}"))
    assert len(result.graph.edges) == 1
    assert result.graph.edges[0].mapping is not None


def test_incremental_models_without_a_spec_are_flagged():
    """Silently widening every incremental model to a full rebuild would be worse."""
    result = ingest_dbt(manifest(adapter="snowflake", materialized="incremental"))
    assert any("incremental model(s) have no partition spec" in n for n in result.notes)


def test_explicit_specs_override_the_manifest():
    override = PartitionSpec.of(PartitionField.time("dt", Grain.YEAR))
    result = ingest_dbt(
        manifest(partition_by={"field": "dt", "data_type": "date", "granularity": "day"}),
        specs={model_id(): override},
    )
    assert result.graph.spec(model_id()) == override


def test_end_to_end_plan_from_a_manifest():
    """The whole point: a dbt project becomes a partition-scoped rebuild plan."""
    from datetime import datetime

    from fathom.core.types import KeyPredicate

    result = ingest_dbt(
        manifest(partition_by={"field": "dt", "data_type": "date", "granularity": "month"})
    )
    plan = result.graph.invalidate({source_id(): [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
    assert plan.partitions(model_id()) == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 1))})


# -- loading -------------------------------------------------------------------


def test_loads_from_a_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()))
    assert load_manifest(str(path))["metadata"]["project_name"] == "analytics"


def test_finds_the_manifest_inside_a_target_directory(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps(manifest()))
    assert load_manifest(str(tmp_path))["metadata"]["adapter_type"] == "bigquery"


def test_missing_manifest_says_to_run_dbt_compile(tmp_path):
    (tmp_path / "models").mkdir()
    with pytest.raises(ConfigError, match="dbt compile"):
        load_manifest(str(tmp_path))


def test_wrong_json_is_rejected_clearly(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"not": "a manifest"}))
    with pytest.raises(ConfigError, match="does not look like a dbt manifest"):
        load_manifest(str(path))


def test_malformed_json_is_rejected_clearly(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{ broken")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_manifest(str(path))
