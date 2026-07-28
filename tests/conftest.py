"""Shared fixtures: small Hive-partitioned Parquet datasets on local disk."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fathom.grains import Grain
from fathom.types import PartitionField, PartitionSpec

DAY_REGION = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))


def write_partition(
    root: Path,
    *,
    dt: date,
    region: str,
    amounts: list[float | None],
    ids: list[str] | None = None,
) -> Path:
    """Write one Hive-partitioned Parquet file and return its path."""
    directory = root / f"dt={dt.isoformat()}" / f"region={region}"
    directory.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": pa.array(ids or [f"{region}-{i}" for i in range(len(amounts))], pa.string()),
            "amount": pa.array(amounts, pa.float64()),
        }
    )
    path = directory / "part-0.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    """Two days, two regions, with statistics written into the footers."""
    root = tmp_path / "events"
    write_partition(root, dt=date(2026, 3, 14), region="eu", amounts=[1.0, 2.0, 3.0])
    write_partition(root, dt=date(2026, 3, 14), region="us", amounts=[10.0, 20.0])
    write_partition(root, dt=date(2026, 3, 15), region="eu", amounts=[4.0, None, 6.0])
    return root


@pytest.fixture
def partition_spec() -> PartitionSpec:
    return DAY_REGION
