"""Dataset identity normalization.

Each of these is a case where the same bytes get spelled two ways in the wild. If
any of them fails to unify, the dependency graph fragments silently.
"""

from __future__ import annotations

import pytest

from fathom.ids import AliasRegistry, normalize, normalize_path, normalize_table
from fathom.types import DatasetId


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
    with pytest.raises(ValueError, match="pass system="):
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
