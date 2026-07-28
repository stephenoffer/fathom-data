"""Certify the local filesystem adapter against the storage contract."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from suite import StorageAdapterContract

from conftest import write_partition
from fathom.adapters import LocalStorage
from fathom.ids import normalize
from fathom.types import DatasetId, PartitionSpec


class TestLocalStorage(StorageAdapterContract):
    def make_adapter(self, root: Path, spec: PartitionSpec) -> tuple[object, DatasetId]:
        adapter = LocalStorage()
        dataset = normalize(str(Path(root).resolve()))
        adapter.declare(dataset, spec)
        return adapter, dataset

    def add_partition(self, root: Path, *, dt: date, region: str, rows: int) -> None:
        write_partition(root, dt=dt, region=region, amounts=[1.0] * rows)
