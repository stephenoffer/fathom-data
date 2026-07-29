"""A Delta table on object storage.

Uses `memory://` in place of `s3://` so this runs with no credentials, but the code
path is identical — the filesystem layer makes protocol a configuration detail.

Swap the URI and add `storage_options` and this is a real S3 table.

    python examples/05_cloud_storage.py
"""

from __future__ import annotations

import json
from datetime import datetime

import fsspec

from fathom import KeyPredicate
from fathom.adapters import DeltaCatalog
from fathom.adapters.fs import clear_cache
from fathom.core.types import DatasetId

SCHEMA = json.dumps(
    {
        "type": "struct",
        "fields": [
            {"name": "dt", "type": "date", "nullable": True, "metadata": {}},
            {"name": "region", "type": "string", "nullable": True, "metadata": {}},
            {"name": "amount", "type": "double", "nullable": True, "metadata": {}},
        ],
    }
)


def commit(fs, root: str, version: int, actions: list[dict]) -> None:
    body = "\n".join(json.dumps(a) for a in actions)
    fs.pipe_file(f"{root}/_delta_log/{version:020d}.json", body.encode())


def add(dt: str, region: str, *, data_change: bool = True, name: str = "part-0") -> dict:
    return {
        "add": {
            "path": f"dt={dt}/region={region}/{name}.parquet",
            "partitionValues": {"dt": dt, "region": region},
            "size": 1024,
            "modificationTime": 1_700_000_000_000,
            "dataChange": data_change,
        }
    }


def main() -> None:
    clear_cache()
    fs = fsspec.filesystem("memory")
    root = "/lake/events"

    commit(
        fs,
        root,
        0,
        [
            {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
            {
                "metaData": {
                    "id": "t1",
                    "format": {"provider": "parquet"},
                    "schemaString": SCHEMA,
                    "partitionColumns": ["dt", "region"],
                    "configuration": {},
                }
            },
            add("2026-03-14", "eu"),
            add("2026-03-14", "us"),
        ],
    )

    # In a real project this is `s3://lake/events` with storage_options for creds.
    dataset = DatasetId("memory://", "lake/events")
    catalog = DeltaCatalog()

    print(f"Is a Delta table: {catalog.is_delta_table(dataset)}")

    spec = catalog.describe_partitioning(dataset)
    print("\nPartition spec, inferred from the log:")
    for field in spec.fields:
        grain = field.grain.label if field.grain else "value"
        print(f"  {field.name}: {field.kind} ({grain})")
    print("  (`dt` is a date column, so it is unambiguously daily.")
    print("   `region` is a string, so it stays a value field — guessing grain")
    print("   from content is how a rebuild silently starts covering the wrong rows.)")

    # --- first scan: everything -----------------------------------------------
    changes = catalog.changed(dataset, None)
    print(f"\nFirst scan: {len(changes.partitions)} partition(s), token={changes.token}")
    for key in sorted(changes.partitions, key=str):
        print(f"  {key}")

    # --- nothing new ----------------------------------------------------------
    print(f"\nNo new commits: {catalog.changed(dataset, changes.token).is_empty}")

    # --- one new partition ----------------------------------------------------
    commit(fs, root, 1, [add("2026-03-15", "eu")])
    incremental = catalog.changed(dataset, changes.token)
    print(f"\nAfter one commit: {[str(k) for k in incremental.partitions]}")
    assert incremental.partitions == frozenset(
        {KeyPredicate.of(dt=datetime(2026, 3, 15), region="eu")}
    )

    # --- compaction must not trigger a rebuild --------------------------------
    commit(
        fs,
        root,
        2,
        [
            {
                "remove": {
                    "path": "dt=2026-03-14/region=eu/part-0.parquet",
                    "partitionValues": {"dt": "2026-03-14", "region": "eu"},
                    "dataChange": False,
                }
            },
            add("2026-03-14", "eu", data_change=False, name="compacted"),
        ],
    )
    after_optimize = catalog.changed(dataset, incremental.token)
    print(f"\nAfter OPTIMIZE (dataChange=false): {after_optimize.is_empty}")
    print("  Compaction moves bytes without changing data. Rebuilding on it is waste.")

    # --- partition-scoped file lists, which is what makes erasure affordable ---
    scoped = catalog.files_for(dataset, KeyPredicate.of(dt=datetime(2026, 3, 15), region="eu"))
    everything = catalog.files_for(dataset)
    print(f"\nLive files: {len(everything)} total, {len(scoped)} in one partition")
    print("  Erasing a subject rewrites the second number, not the first.")


if __name__ == "__main__":
    main()
