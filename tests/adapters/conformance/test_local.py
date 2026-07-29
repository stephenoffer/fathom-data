"""Certify the built-in adapters against their contracts.

`ObjectStorage` is certified twice — once on local disk, once on `memory://` — so
the object-storage code path is exercised without a network. If both pass, an S3
adapter is the same code with different credentials.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import fsspec
import pytest
from suite import CatalogAdapterContract, StorageAdapterContract

from conftest import write_partition
from fathom.adapters import DeltaCatalog, ObjectStorage
from fathom.adapters.fs import clear_cache
from fathom.core.ids import normalize
from fathom.core.types import DatasetId, PartitionSpec


class TestObjectStorageOnDisk(StorageAdapterContract):
    def make_adapter(self, root: Path, spec: PartitionSpec) -> tuple[Any, DatasetId]:
        adapter = ObjectStorage()
        dataset = normalize(str(Path(root).resolve()))
        adapter.declare(dataset, spec)
        return adapter, dataset

    def add_partition(self, root: Path, *, dt: date, region: str, rows: int) -> None:
        write_partition(root, dt=dt, region=region, amounts=[1.0] * rows)


# -- Delta, on local disk and in object storage --------------------------------

SCHEMA = json.dumps(
    {
        "type": "struct",
        "fields": [
            {"name": "dt", "type": "date", "nullable": True, "metadata": {}},
            {"name": "region", "type": "string", "nullable": True, "metadata": {}},
        ],
    }
)


def _commit_body(actions: list[dict]) -> str:
    return "\n".join(json.dumps(a) for a in actions)


def _metadata_action() -> dict:
    return {
        "metaData": {
            "id": "t1",
            "format": {"provider": "parquet"},
            "schemaString": SCHEMA,
            "partitionColumns": ["dt", "region"],
            "configuration": {},
        }
    }


def _add_action(dt: date, region: str, name: str = "part-0") -> dict:
    return {
        "add": {
            "path": f"dt={dt.isoformat()}/region={region}/{name}.parquet",
            "partitionValues": {"dt": dt.isoformat(), "region": region},
            "size": 128,
            "modificationTime": 1_700_000_000_000,
            "dataChange": True,
        }
    }


class TestDeltaOnDisk(CatalogAdapterContract):
    def make_adapter(self, root: Path, spec: PartitionSpec) -> tuple[Any, DatasetId]:
        root = Path(root)
        log = root / "_delta_log"
        if not log.exists():
            log.mkdir(parents=True)
            (log / f"{0:020d}.json").write_text(
                _commit_body([_metadata_action(), _add_action(date(2026, 3, 14), "eu")])
            )
        return DeltaCatalog(), normalize(str(root.resolve()))

    def add_partition(self, root: Path, *, dt: date, region: str, rows: int) -> None:
        log = Path(root) / "_delta_log"
        version = len(list(log.glob("*.json")))
        (log / f"{version:020d}.json").write_text(_commit_body([_add_action(dt, region)]))


class TestDeltaOnObjectStorage(CatalogAdapterContract):
    """The same adapter over `memory://`, standing in for S3."""

    _roots: dict[str, str] = {}

    def make_adapter(self, root: Path, spec: PartitionSpec) -> tuple[Any, DatasetId]:
        clear_cache()
        fs = fsspec.filesystem("memory")
        key = f"/delta/{Path(root).name}"
        if key not in self._roots:
            fs.pipe_file(
                f"{key}/_delta_log/{0:020d}.json",
                _commit_body([_metadata_action(), _add_action(date(2026, 3, 14), "eu")]).encode(),
            )
            self._roots[key] = key
        return DeltaCatalog(), DatasetId("memory://", key.lstrip("/"))

    def add_partition(self, root: Path, *, dt: date, region: str, rows: int) -> None:
        fs = fsspec.filesystem("memory")
        key = f"/delta/{Path(root).name}"
        existing = [p for p in fs.find(f"{key}/_delta_log") if p.endswith(".json")]
        fs.pipe_file(
            f"{key}/_delta_log/{len(existing):020d}.json",
            _commit_body([_add_action(dt, region)]).encode(),
        )


@pytest.fixture(autouse=True)
def _reset_memory_filesystem():
    yield
    fsspec.filesystem("memory").store.clear()
    TestDeltaOnObjectStorage._roots.clear()
    clear_cache()
