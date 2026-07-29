"""Catalogs: table metadata and commit-level change detection.

A catalog knows its own partition spec and can name what changed between two
snapshots without listing a bucket. That makes `SNAPSHOT_DIFF` the cheapest change
source on the ladder, and it is why these are worth writing per format.

Iceberg is imported lazily: its manifests are Avro, so unlike Delta it cannot be
read without a library, and a bare install must not fail on that.
"""

from typing import Any

from .delta import DeltaCatalog

__all__ = ["DeltaCatalog", "IcebergCatalog"]


def __getattr__(name: str) -> Any:
    if name == "IcebergCatalog":
        from .iceberg import IcebergCatalog

        return IcebergCatalog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
