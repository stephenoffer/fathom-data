"""The storage adapter conformance contract.

Every `StorageAdapter` must pass this, whether it addresses local disk, S3, GCS,
ADLS, or something in-house. The contract is deliberately weak: it asserts what a
planner is entitled to rely on, not how the adapter achieves it. An adapter using
snapshot diffs and one using LIST plus mtime should both pass unchanged.

The rule that matters is the last one. An adapter may report a partition that did
not change; it may never omit one that did.

To certify a new adapter, subclass `StorageAdapterContract` and implement
`make_adapter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from fathom.adapters.base import ChangeSet, ObjectMeta
from fathom.types import ANY, Capabilities, DatasetId, KeyPredicate, PartitionSpec


class StorageAdapterContract(ABC):
    """Subclass and implement `make_adapter` to certify an adapter."""

    @abstractmethod
    def make_adapter(self, root: Path, spec: PartitionSpec) -> tuple[object, DatasetId]:
        """Return a configured adapter and the dataset id addressing `root`."""

    @abstractmethod
    def add_partition(self, root: Path, *, dt: date, region: str, rows: int) -> None:
        """Write one new partition into the dataset."""

    # -- capabilities ---------------------------------------------------------

    def test_declares_capabilities(self, lake, partition_spec):
        adapter, _ = self.make_adapter(lake, partition_spec)
        assert isinstance(adapter.capabilities, Capabilities)

    # -- listing --------------------------------------------------------------

    def test_lists_objects_with_usable_metadata(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        objects = list(adapter.list_objects(ds))
        assert objects, "adapter found no objects in a populated dataset"
        assert all(isinstance(o, ObjectMeta) for o in objects)
        assert all(o.path for o in objects)

    def test_listing_an_absent_dataset_is_empty_not_an_error(self, tmp_path, partition_spec):
        adapter, ds = self.make_adapter(tmp_path / "does-not-exist", partition_spec)
        assert list(adapter.list_objects(ds)) == []

    def test_listing_is_stable(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        first = [o.path for o in adapter.list_objects(ds)]
        second = [o.path for o in adapter.list_objects(ds)]
        assert first == second

    # -- partition attribution ------------------------------------------------

    def test_objects_map_to_partition_keys(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        keys = {adapter.key_for(ds, o.path) for o in adapter.list_objects(ds)}
        assert keys, "no partition keys derived"
        assert all(isinstance(k, KeyPredicate) for k in keys)

    def test_unresolvable_fields_widen_rather_than_erroring(self, tmp_path, partition_spec):
        """A stray file outside the layout must widen, not crash the scan."""
        root = tmp_path / "messy"
        root.mkdir()
        (root / "loose.parquet").write_bytes(b"")
        adapter, ds = self.make_adapter(root, partition_spec)
        key = adapter.key_for(ds, str(root / "loose.parquet"))
        assert key.get("dt") is ANY

    # -- change detection -----------------------------------------------------

    def test_first_call_reports_everything(self, lake, partition_spec):
        """With no token there is no baseline, so everything is potentially dirty."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        changes = adapter.changed(ds, None)
        assert isinstance(changes, ChangeSet)
        assert changes.partitions
        assert changes.token, "adapter must return a resumable token"

    def test_no_writes_means_no_new_partitions(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        first = adapter.changed(ds, None)
        second = adapter.changed(ds, first.token)
        assert second.partitions <= first.partitions

    def test_a_new_partition_is_always_reported(self, lake, partition_spec):
        """The one contract violation that matters: never omit a real change."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        baseline = adapter.changed(ds, None)

        self.add_partition(lake, dt=date(2026, 3, 20), region="apac", rows=5)

        after = adapter.changed(ds, baseline.token)
        found = {k.get("region") for k in after.partitions}
        assert "apac" in found or ANY in found, (
            "adapter failed to report a partition that was written after the token"
        )

    def test_token_round_trips(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        token = adapter.changed(ds, None).token
        assert adapter.changed(ds, token).token

    def test_incomplete_results_are_labelled(self, lake, partition_spec):
        """Adapters that cannot enumerate exhaustively must say so, not return empty."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        changes = adapter.changed(ds, None)
        assert isinstance(changes.complete, bool)
