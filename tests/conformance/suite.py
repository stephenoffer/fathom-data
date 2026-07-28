"""Adapter conformance contracts.

Every adapter must pass the contract for its surface, whether it addresses local
disk, S3, Delta, Iceberg, Snowflake, or something in-house. The contracts are
deliberately weak: they assert what a planner is entitled to rely on, not how an
adapter achieves it. An adapter using snapshot diffs and one using LIST plus mtime
should both pass unchanged.

The rule that matters, and it appears in every contract here:

    An adapter may report a partition that did not change.
    It may never omit one that did.

Everything else is an optimization. To certify a new adapter, subclass the contract
for its surface and implement the two or three hooks it asks for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

from fathom.adapters.base import ChangeSet, ObjectMeta
from fathom.types import ANY, Capabilities, DatasetId, KeyPredicate, PartitionSpec


class _ChangeDetectionContract(ABC):
    """Shared by every adapter that answers "what changed since a token"."""

    @abstractmethod
    def make_adapter(self, root: Path, spec: PartitionSpec) -> tuple[Any, DatasetId]:
        """Return a configured adapter and the dataset id addressing `root`."""

    @abstractmethod
    def add_partition(self, root: Path, *, dt: date, region: str, rows: int) -> None:
        """Write one new partition into the dataset."""

    # -- capabilities ---------------------------------------------------------

    def test_declares_capabilities(self, lake, partition_spec):
        adapter, _ = self.make_adapter(lake, partition_spec)
        assert isinstance(adapter.capabilities, Capabilities)

    def test_capability_flags_are_self_consistent(self, lake, partition_spec):
        """An adapter claiming column lineage must be able to produce lineage."""
        adapter, _ = self.make_adapter(lake, partition_spec)
        caps = adapter.capabilities
        if caps.column_lineage:
            assert hasattr(adapter, "fetch_lineage") or hasattr(adapter, "fetch_queries")

    # -- change detection -----------------------------------------------------

    def test_first_call_reports_everything(self, lake, partition_spec):
        """With no token there is no baseline, so everything is potentially dirty."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        changes = adapter.changed(ds, None)
        assert isinstance(changes, ChangeSet)
        assert changes.partitions
        assert changes.token, "an adapter must return a resumable token"

    def test_no_writes_means_no_new_partitions(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        first = adapter.changed(ds, None)
        second = adapter.changed(ds, first.token)
        assert second.partitions <= first.partitions

    def test_repeated_scans_converge(self, lake, partition_spec):
        """A quiet dataset must stop reporting, or incremental profiling is useless."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        token = adapter.changed(ds, None).token
        for _ in range(3):
            changes = adapter.changed(ds, token)
            token = changes.token or token
        assert changes.is_empty

    def test_a_new_partition_is_always_reported(self, lake, partition_spec):
        """The one contract violation that matters: never omit a real change."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        baseline = adapter.changed(ds, None)

        self.add_partition(lake, dt=date(2026, 3, 20), region="apac", rows=5)

        after = adapter.changed(ds, baseline.token)
        found = {k.get("region") for k in after.partitions}
        assert "apac" in found or ANY in found, (
            "adapter failed to report a partition written after the token"
        )

    def test_token_round_trips(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        token = adapter.changed(ds, None).token
        assert adapter.changed(ds, token).token

    def test_an_unrecognized_token_does_not_crash(self, lake, partition_spec):
        """Tokens outlive schema changes and get copied between environments."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        changes = adapter.changed(ds, "not-a-token-this-adapter-wrote")
        assert isinstance(changes, ChangeSet)

    def test_incomplete_results_are_labelled(self, lake, partition_spec):
        """Adapters that cannot enumerate exhaustively must say so, not return empty."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        assert isinstance(adapter.changed(ds, None).complete, bool)

    def test_partitions_are_key_predicates(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        changes = adapter.changed(ds, None)
        assert all(isinstance(k, KeyPredicate) for k in changes.partitions)


class StorageAdapterContract(_ChangeDetectionContract):
    """Object storage: what exists, what changed, and how to read it cheaply."""

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
        assert adapter.key_for(ds, str(root / "loose.parquet")).get("dt") is ANY

    def test_paths_are_openable_uris(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        objects = list(adapter.list_objects(ds))
        assert adapter.paths(objects) == [o.path for o in objects]


class CatalogAdapterContract(_ChangeDetectionContract):
    """Table metadata: partition specs and commit-level change detection."""

    def test_describes_partitioning(self, lake, partition_spec):
        adapter, ds = self.make_adapter(lake, partition_spec)
        spec = adapter.describe_partitioning(ds)
        assert isinstance(spec, PartitionSpec)

    def test_declared_specs_win_over_inference(self, lake, partition_spec):
        """A declaration is the user overriding us; inference must not fight it."""
        from fathom.grains import Grain
        from fathom.types import PartitionField

        adapter, ds = self.make_adapter(lake, partition_spec)
        declared = PartitionSpec.of(PartitionField.time("dt", Grain.YEAR))
        adapter.declare(ds, declared)
        assert adapter.describe_partitioning(ds) == declared

    def test_partition_keys_match_the_declared_spec(self, lake, partition_spec):
        """Keys carrying fields the spec does not declare break every comparison."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        names = set(adapter.describe_partitioning(ds).names)
        for key in adapter.changed(ds, None).partitions:
            assert {k for k, _ in key.bindings} <= names or not names

    def test_files_can_be_scoped_to_one_partition(self, lake, partition_spec):
        """What makes erasure affordable: rewriting a partition, not a table."""
        adapter, ds = self.make_adapter(lake, partition_spec)
        if not hasattr(adapter, "files_for"):
            return
        everything = adapter.files_for(ds)
        if not everything:
            return
        one = next(iter(adapter.changed(ds, None).partitions))
        assert len(adapter.files_for(ds, one)) <= len(everything)
