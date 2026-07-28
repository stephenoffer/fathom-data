"""Deriving partition keys from object paths."""

from __future__ import annotations

from datetime import datetime

from fathom.grains import Grain
from fathom.paths import PathTemplate, key_from_path, parse_hive_partitions
from fathom.types import ANY, KeyPredicate, PartitionField, PartitionSpec

DAY_REGION = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))


def test_hive_segments_parse():
    got = parse_hive_partitions("warehouse/dt=2026-01-15/region=eu/part-0.parquet")
    assert got == {"dt": "2026-01-15", "region": "eu"}


def test_filename_is_not_mistaken_for_a_partition():
    assert parse_hive_partitions("dt=2026-01-15/part-0.parquet") == {"dt": "2026-01-15"}


def test_hive_layout_yields_a_full_key():
    got = key_from_path("dt=2026-01-15/region=eu/part-0.parquet", DAY_REGION)
    assert got == KeyPredicate.of(dt=datetime(2026, 1, 15), region="eu")


def test_unresolvable_field_widens_rather_than_guessing():
    got = key_from_path("dt=2026-01-15/part-0.parquet", DAY_REGION)
    assert got.get("region") is ANY


def test_non_hive_path_without_a_template_binds_nothing():
    """Guessing which path segment is a month is how a rebuild plan silently rots."""
    got = key_from_path("events/2026/01/15/part-0.parquet", DAY_REGION)
    assert got.get("dt") is ANY and got.get("region") is ANY


def test_template_recovers_a_non_hive_layout():
    tmpl = PathTemplate("events/{yyyy}/{MM}/{dd}")
    got = key_from_path("events/2026/01/15/part-0.parquet", DAY_REGION, template=tmpl)
    assert got.get("dt") == datetime(2026, 1, 15)


def test_template_can_bind_value_fields_too():
    tmpl = PathTemplate("events/{region}/{yyyy}/{MM}/{dd}")
    got = key_from_path("events/eu/2026/01/15/f.parquet", DAY_REGION, template=tmpl)
    assert got == KeyPredicate.of(dt=datetime(2026, 1, 15), region="eu")


def test_time_values_truncate_to_the_declared_grain():
    spec = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))
    got = key_from_path("dt=2026-01-15/f.parquet", spec)
    assert got.get("dt") == datetime(2026, 1, 1)


def test_compact_date_formats_are_accepted():
    spec = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
    assert key_from_path("dt=20260115/f.parquet", spec).get("dt") == datetime(2026, 1, 15)


def test_unpartitioned_spec_yields_the_empty_key():
    assert key_from_path("anything/at/all.parquet", PartitionSpec()) == KeyPredicate()
