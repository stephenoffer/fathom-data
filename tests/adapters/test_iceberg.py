"""Iceberg catalog adapter, exercised against real tables written by pyiceberg.

Tables are created and appended to through pyiceberg so the manifests, snapshots,
and partition encodings are genuine. Hand-built Avro would only test our own
assumptions back at us.
"""

from __future__ import annotations

from datetime import datetime

import pytest

pyiceberg = pytest.importorskip("pyiceberg", reason="needs the 'iceberg' extra")
pytest.importorskip("sqlalchemy", reason="the in-memory catalog needs sqlalchemy")

import pyarrow as pa  # noqa: E402
from pyiceberg.catalog.memory import InMemoryCatalog  # noqa: E402
from pyiceberg.partitioning import PartitionField as IcePartitionField  # noqa: E402
from pyiceberg.partitioning import PartitionSpec as IcePartitionSpec  # noqa: E402
from pyiceberg.schema import Schema  # noqa: E402
from pyiceberg.transforms import DayTransform, IdentityTransform, MonthTransform  # noqa: E402
from pyiceberg.types import (  # noqa: E402
    DoubleType,
    NestedField,
    StringType,
    TimestampType,
)

from fathom.adapters.catalogs.iceberg import IcebergCatalog, _decode  # noqa: E402
from fathom.core.grains import Grain  # noqa: E402
from fathom.core.ids import normalize  # noqa: E402
from fathom.core.types import KeyPredicate  # noqa: E402

SCHEMA = Schema(
    NestedField(1, "dt", TimestampType(), required=False),
    NestedField(2, "region", StringType(), required=False),
    NestedField(3, "amount", DoubleType(), required=False),
)

ARROW_SCHEMA = pa.schema(
    [
        pa.field("dt", pa.timestamp("us"), nullable=True),
        pa.field("region", pa.string(), nullable=True),
        pa.field("amount", pa.float64(), nullable=True),
    ]
)


def rows(*items: tuple[datetime, str, float]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"dt": dt, "region": region, "amount": amount} for dt, region, amount in items],
        schema=ARROW_SCHEMA,
    )


@pytest.fixture
def warehouse(tmp_path):
    root = tmp_path / "warehouse"
    root.mkdir()
    catalog = InMemoryCatalog("test", warehouse=str(root))
    catalog.create_namespace("db")
    return catalog, root


def make_table(catalog, root, *, spec: IcePartitionSpec, name: str = "db.events"):
    table = catalog.create_table(name, schema=SCHEMA, partition_spec=spec)
    return table


DAY_REGION_SPEC = IcePartitionSpec(
    IcePartitionField(source_id=1, field_id=1000, transform=DayTransform(), name="dt_day"),
    IcePartitionField(source_id=2, field_id=1001, transform=IdentityTransform(), name="region"),
)


@pytest.fixture
def table(warehouse):
    catalog, root = warehouse
    tbl = make_table(catalog, root, spec=DAY_REGION_SPEC)
    tbl.append(
        rows(
            (datetime(2026, 3, 14), "eu", 1.0),
            (datetime(2026, 3, 14), "us", 2.0),
        )
    )
    return tbl


def dataset_for(table):
    # The table's location is the dataset root; metadata/ lives beneath it.
    return normalize(table.location().replace("file://", ""))


# -- decoding ------------------------------------------------------------------


def test_day_offsets_decode_from_the_epoch():
    assert _decode(0, Grain.DAY) == datetime(1970, 1, 1)
    assert _decode(20526, Grain.DAY) == datetime(2026, 3, 14)


def test_month_offsets_decode():
    assert _decode(0, Grain.MONTH) == datetime(1970, 1, 1)
    assert _decode(674, Grain.MONTH) == datetime(2026, 3, 1)


def test_year_offsets_decode():
    assert _decode(56, Grain.YEAR) == datetime(2026, 1, 1)


def test_hour_offsets_decode():
    assert _decode(1, Grain.HOUR) == datetime(1970, 1, 1, 1)


def test_undecodable_values_return_none():
    assert _decode(None, Grain.DAY) is None
    assert _decode("2026-03-14", Grain.DAY) is None


# -- discovery and spec --------------------------------------------------------


def test_recognizes_an_iceberg_table(table):
    assert IcebergCatalog().is_iceberg_table(dataset_for(table))


def test_plain_directory_is_not_an_iceberg_table(tmp_path):
    (tmp_path / "plain").mkdir()
    assert not IcebergCatalog().is_iceberg_table(normalize(str(tmp_path / "plain")))


def test_day_transform_becomes_a_day_grained_field(table):
    spec = IcebergCatalog().describe_partitioning(dataset_for(table))
    field = spec.field("dt_day")
    assert field is not None and field.kind == "time" and field.grain is Grain.DAY


def test_identity_transform_stays_a_value_field(table):
    spec = IcebergCatalog().describe_partitioning(dataset_for(table))
    field = spec.field("region")
    assert field is not None and field.kind == "value"


def test_month_transform_becomes_month_grained(warehouse):
    catalog, root = warehouse
    spec = IcePartitionSpec(
        IcePartitionField(source_id=1, field_id=1000, transform=MonthTransform(), name="dt_month")
    )
    tbl = make_table(catalog, root, spec=spec, name="db.monthly")
    tbl.append(rows((datetime(2026, 3, 14), "eu", 1.0)))

    got = IcebergCatalog().describe_partitioning(dataset_for(tbl))
    field = got.field("dt_month")
    assert field is not None and field.grain is Grain.MONTH


# -- change detection ----------------------------------------------------------


def test_first_run_reports_every_partition(table):
    changes = IcebergCatalog().changed(dataset_for(table), None)
    assert changes.complete
    assert len(changes.partitions) == 2
    assert changes.token


def test_partition_values_decode_to_real_datetimes(table):
    changes = IcebergCatalog().changed(dataset_for(table), None)
    assert KeyPredicate.of(dt_day=datetime(2026, 3, 14), region="eu") in changes.partitions


def test_token_limits_to_later_snapshots(table):
    adapter = IcebergCatalog()
    ds = dataset_for(table)
    first = adapter.changed(ds, None)

    table.append(rows((datetime(2026, 3, 15), "eu", 3.0)))

    changes = adapter.changed(ds, first.token)
    assert changes.partitions == frozenset(
        {KeyPredicate.of(dt_day=datetime(2026, 3, 15), region="eu")}
    )


def test_no_new_snapshots_reports_nothing(table):
    adapter = IcebergCatalog()
    ds = dataset_for(table)
    first = adapter.changed(ds, None)
    assert adapter.changed(ds, first.token).is_empty


def test_an_expired_token_widens_and_admits_it(table):
    """Returning an empty diff for an unknown token would be the unforgivable bug."""
    changes = IcebergCatalog().changed(dataset_for(table), "999999999999999999")
    assert not changes.complete
    assert changes.partitions


# -- live files ----------------------------------------------------------------


def test_files_for_lists_live_data(table):
    files = IcebergCatalog().files_for(dataset_for(table))
    assert files and all(f.endswith(".parquet") for f in files)


def test_files_for_can_scope_to_one_partition(table):
    adapter = IcebergCatalog()
    ds = adapter and dataset_for(table)
    scoped = adapter.files_for(ds, KeyPredicate.of(dt_day=datetime(2026, 3, 14), region="eu"))
    everything = adapter.files_for(ds)
    assert 0 < len(scoped) < len(everything)
