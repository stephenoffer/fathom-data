"""Dataset identity normalization.

Each of these is a case where the same bytes get spelled two ways in the wild. If
any of them fails to unify, the dependency graph fragments silently.
"""

from __future__ import annotations

import pytest

from fathom.core.ids import (
    AliasRegistry,
    dataset_uri,
    is_path_dataset,
    normalize,
    normalize_path,
    normalize_table,
)
from fathom.core.types import DatasetId


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("s3a://lake/raw/events", "s3://lake/raw/events"),
        ("s3n://lake/raw/events", "s3://lake/raw/events"),
        ("s3://lake//raw//events/", "s3://lake/raw/events"),
        ("gcs://lake/raw", "gs://lake/raw"),
        ("S3://LAKE/raw", "s3://lake/raw"),
        (
            "wasbs://data@acct.blob.core.windows.net/raw",
            "abfss://data@acct.dfs.core.windows.net/raw",
        ),
    ],
)
def test_equivalent_spellings_unify(left, right):
    assert normalize_path(left) == normalize_path(right)


def test_bucket_becomes_namespace_and_key_becomes_name():
    assert normalize_path("s3://lake/raw/events") == DatasetId("s3://lake", "raw/events")


def test_case_is_preserved_in_object_keys():
    """Bucket names fold; keys do not. S3 keys are case sensitive."""
    assert normalize_path("s3://Lake/Raw/Events").name == "Raw/Events"


def test_dbfs_mount_resolves_to_underlying_storage():
    mounts = {"/mnt/lake": "s3://lake"}
    assert normalize_path("/dbfs/mnt/lake/raw/events", mounts=mounts) == DatasetId(
        "s3://lake", "raw/events"
    )
    assert normalize_path("dbfs:/mnt/lake/raw/events", mounts=mounts) == DatasetId(
        "s3://lake", "raw/events"
    )


def test_longest_mount_prefix_wins():
    mounts = {"/mnt/lake": "s3://general", "/mnt/lake/raw": "s3://raw-only"}
    assert normalize_path("/dbfs/mnt/lake/raw/events", mounts=mounts).namespace == "s3://raw-only"


def test_snowflake_folds_unquoted_identifiers_up():
    a = normalize_table(
        "orders", system="snowflake", instance="ac1", default_database="db", default_schema="public"
    )
    b = normalize_table("DB.PUBLIC.Orders", system="snowflake", instance="ac1")
    assert a == b == DatasetId("snowflake://ac1", "DB.PUBLIC.ORDERS")


def test_quoted_identifiers_keep_their_case():
    got = normalize_table('db.public."MixedCase"', system="snowflake", instance="ac1")
    assert got.name == "DB.PUBLIC.MixedCase"


def test_databricks_folds_down():
    got = normalize_table("Main.Sales.Orders", system="databricks", instance="ws1")
    assert got.name == "main.sales.orders"


def test_bigquery_is_case_sensitive():
    got = normalize_table("Proj.Dataset.Table", system="bigquery")
    assert got.name == "Proj.Dataset.Table"


def test_uri_wins_over_table_interpretation():
    assert normalize("s3://lake/raw", system="snowflake").namespace == "s3://lake"


def test_bare_name_without_system_is_an_error():
    with pytest.raises(ValueError, match="Pass `system=`"):
        normalize("orders")


def test_alias_registry_resolves_transitively():
    reg = AliasRegistry()
    hive = DatasetId("hive://cluster", "raw.events")
    s3 = DatasetId("s3://lake", "raw/events")
    trino = DatasetId("trino://cluster", "hive.raw.events")
    reg.alias(hive, s3)
    reg.alias(trino, hive)
    assert reg.resolve(trino) == s3


def test_alias_cycles_are_rejected():
    reg = AliasRegistry()
    a, b = DatasetId("x", "a"), DatasetId("x", "b")
    reg.alias(a, b)
    with pytest.raises(ValueError, match="cycle"):
        reg.alias(b, a)


def test_local_paths_render_as_file_uris():
    """`file` + an absolute name would otherwise print as `file//private/...`."""
    assert str(normalize_path("/private/tmp/events")) == "file:///private/tmp/events"


def test_bucket_datasets_render_with_a_single_separator():
    assert str(normalize_path("s3://lake/raw/events")) == "s3://lake/raw/events"


# -- classifying and reversing identities --------------------------------------


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        (normalize_path("s3://lake/events"), True),
        (normalize_path("gs://lake/events"), True),
        (normalize_path("/tmp/events"), True),
        (normalize_path("abfss://c@acct/events"), True),
        (normalize_path("hdfs://nn/events"), True),
        (normalize_table("db.schema.t", system="snowflake"), False),
        (normalize_table("raw.events", system="duckdb"), False),
    ],
)
def test_path_datasets_are_distinguished_from_catalog_entries(dataset, expected):
    assert is_path_dataset(dataset) is expected


@pytest.mark.parametrize(
    "uri",
    ["s3://lake/raw/events", "gs://lake/raw/events", "/tmp/lake/events"],
)
def test_dataset_uri_inverts_normalize_path(uri):
    assert normalize_path(dataset_uri(normalize_path(uri))) == normalize_path(uri)


def test_dataset_uri_refuses_a_catalog_dataset():
    """Returning something path-shaped would send an adapter looking for bytes."""
    table = normalize_table("raw.events", system="snowflake", instance="acct")
    with pytest.raises(ValueError, match="not a location"):
        dataset_uri(table)


def test_dataset_uri_of_a_bare_bucket():
    assert dataset_uri(normalize_path("s3://lake")) == "s3://lake"


# -- the alias registry as a collection ----------------------------------------


def test_an_alias_registry_reports_what_it_holds():
    registry = AliasRegistry()
    hive = DatasetId("hive", "raw.events")
    stored = DatasetId("s3://lake", "raw/events")
    registry.alias(hive, stored)

    assert hive in registry
    assert stored not in registry  # the canonical side is not itself an alias
    assert registry.items() == [(hive, stored)]


def test_aliasing_an_identity_to_itself_is_a_no_op():
    """Callers loop over declarations; making them filter this is busywork."""
    registry = AliasRegistry()
    same = DatasetId("hive", "raw.events")
    registry.alias(same, same)
    assert len(registry) == 0


def test_a_cycle_says_which_declaration_to_drop():
    registry = AliasRegistry()
    a, b = DatasetId("hive", "a"), DatasetId("hive", "b")
    registry.alias(a, b)
    with pytest.raises(ValueError) as exc:
        registry.alias(b, a)
    assert "would form a cycle" in str(exc.value)
    assert "canonical identity" in str(exc.value)


def test_a_catalog_dataset_says_how_to_reach_it_instead():
    with pytest.raises(ValueError) as exc:
        dataset_uri(DatasetId("snowflake://xy12345", "db.schema.orders"))
    message = str(exc.value)
    assert "engine adapter" in message
    assert "alias" in message
