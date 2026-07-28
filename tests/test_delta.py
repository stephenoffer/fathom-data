"""Delta transaction log reading.

The log is constructed by hand here rather than through `deltalake`, because the
point of the adapter is that it reads the on-disk format directly. Testing against
the writer that produced it would hide format assumptions.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from fathom.adapters.delta import DeltaCatalog
from fathom.grains import Grain
from fathom.ids import normalize
from fathom.types import ANY, KeyPredicate, PartitionField, PartitionSpec

SCHEMA = {
    "type": "struct",
    "fields": [
        {"name": "dt", "type": "date", "nullable": True, "metadata": {}},
        {"name": "region", "type": "string", "nullable": True, "metadata": {}},
        {"name": "amount", "type": "double", "nullable": True, "metadata": {}},
    ],
}


def write_commit(root: Path, version: int, actions: list[dict]) -> None:
    log = root / "_delta_log"
    log.mkdir(parents=True, exist_ok=True)
    path = log / f"{version:020d}.json"
    path.write_text("\n".join(json.dumps(a) for a in actions) + "\n")


def add(dt: str, region: str, *, data_change: bool = True, name: str = "part-0") -> dict:
    return {
        "add": {
            "path": f"dt={dt}/region={region}/{name}.parquet",
            "partitionValues": {"dt": dt, "region": region},
            "size": 128,
            "modificationTime": 1_700_000_000_000,
            "dataChange": data_change,
        }
    }


def remove(dt: str, region: str, *, data_change: bool = True, name: str = "part-0") -> dict:
    return {
        "remove": {
            "path": f"dt={dt}/region={region}/{name}.parquet",
            "partitionValues": {"dt": dt, "region": region},
            "dataChange": data_change,
        }
    }


def metadata(partition_columns: list[str]) -> dict:
    return {
        "metaData": {
            "id": "t1",
            "format": {"provider": "parquet"},
            "schemaString": json.dumps(SCHEMA),
            "partitionColumns": partition_columns,
            "configuration": {},
        }
    }


@pytest.fixture
def table(tmp_path: Path) -> Path:
    root = tmp_path / "events"
    write_commit(
        root,
        0,
        [
            {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
            metadata(["dt", "region"]),
            add("2026-03-14", "eu"),
            add("2026-03-14", "us"),
        ],
    )
    write_commit(root, 1, [add("2026-03-15", "eu")])
    return root


@pytest.fixture
def catalog() -> DeltaCatalog:
    return DeltaCatalog()


def dataset_for(root: Path):
    return normalize(str(root.resolve()))


# -- discovery -----------------------------------------------------------------


def test_recognizes_a_delta_table(table, catalog):
    assert catalog.is_delta_table(dataset_for(table))


def test_plain_directory_is_not_a_delta_table(tmp_path, catalog):
    (tmp_path / "plain").mkdir()
    assert not catalog.is_delta_table(dataset_for(tmp_path / "plain"))


def test_latest_version(table, catalog):
    assert catalog.latest_version(dataset_for(table)) == 1


# -- partition spec ------------------------------------------------------------


def test_date_partition_columns_become_day_grained(table, catalog):
    spec = catalog.describe_partitioning(dataset_for(table))
    dt = spec.field("dt")
    assert dt is not None and dt.kind == "time" and dt.grain is Grain.DAY


def test_string_partition_columns_stay_value_fields(table, catalog):
    """A string column holding dates is still a value field; guessing grain is unsafe."""
    spec = catalog.describe_partitioning(dataset_for(table))
    region = spec.field("region")
    assert region is not None and region.kind == "value"


def test_unpartitioned_table_reports_no_fields(tmp_path, catalog):
    root = tmp_path / "flat"
    write_commit(root, 0, [metadata([])])
    assert len(catalog.describe_partitioning(dataset_for(root))) == 0


def test_declared_spec_overrides_inference(table, catalog):
    ds = dataset_for(table)
    declared = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))
    catalog.declare(ds, declared)
    assert catalog.describe_partitioning(ds) == declared


# -- change detection ----------------------------------------------------------


def test_first_run_reports_every_partition(table, catalog):
    changes = catalog.changed(dataset_for(table), None)
    assert changes.token == "1"
    assert changes.complete
    assert len(changes.partitions) == 3


def test_token_limits_to_later_commits(table, catalog):
    changes = catalog.changed(dataset_for(table), "0")
    assert changes.partitions == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 15), region="eu")})


def test_no_new_commits_reports_nothing(table, catalog):
    ds = dataset_for(table)
    first = catalog.changed(ds, None)
    assert catalog.changed(ds, first.token).is_empty


def test_removes_count_as_changes(table, catalog):
    """Deleting rows shifts a downstream aggregate exactly as much as inserting does."""
    write_commit(table, 2, [remove("2026-03-14", "eu")])
    changes = catalog.changed(dataset_for(table), "1")
    assert KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu") in changes.partitions


def test_metadata_only_rewrites_are_ignored(table, catalog):
    """Compaction moves bytes without changing data; rebuilding on it is pure waste."""
    write_commit(
        table,
        2,
        [
            remove("2026-03-14", "eu", data_change=False),
            add("2026-03-14", "eu", data_change=False, name="compacted"),
        ],
    )
    assert catalog.changed(dataset_for(table), "1").is_empty


def test_unparseable_partition_value_widens(tmp_path, catalog):
    root = tmp_path / "odd"
    write_commit(root, 0, [metadata(["dt", "region"]), add("not-a-date", "eu")])
    changes = catalog.changed(dataset_for(root), None)
    assert next(iter(changes.partitions)).get("dt") is ANY


def test_objects_carry_absolute_paths(table, catalog):
    changes = catalog.changed(dataset_for(table), None)
    assert changes.objects
    assert all(Path(o.path).is_absolute() for o in changes.objects)


# -- live file set -------------------------------------------------------------


def test_files_for_replays_adds_and_removes(table, catalog):
    write_commit(table, 2, [remove("2026-03-14", "us")])
    files = catalog.files_for(dataset_for(table))
    assert not any("region=us" in f for f in files)
    assert len(files) == 2


def test_files_for_can_scope_to_one_partition(table, catalog):
    """The property that makes erasure affordable."""
    files = catalog.files_for(
        dataset_for(table), KeyPredicate.of(dt=datetime(2026, 3, 15), region="eu")
    )
    assert len(files) == 1
    assert "dt=2026-03-15" in files[0]


def test_remote_namespaces_are_rejected_clearly(catalog):
    from fathom.types import DatasetId

    with pytest.raises(ValueError, match="storage adapter"):
        catalog.changed(DatasetId("s3://lake", "events"), None)
