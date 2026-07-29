"""Footer profiling and drift detection."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from conftest import write_partition
from fathom.adapters import LocalStorage
from fathom.core.ids import normalize
from fathom.core.types import KeyPredicate
from fathom.observe.profile import Severity, drift, profile_parquet


def test_footer_profile_reads_no_data_pages(lake: Path, partition_spec):
    """Row counts, null counts, and ranges all come from metadata."""
    ds = normalize(str(lake))
    store = LocalStorage()
    store.declare(ds, partition_spec)
    got = store.profile(ds)

    assert got.source == "footer"
    assert got.row_count == 8
    assert got.file_count == 3
    assert set(got.column_names) == {"id", "amount"}

    amount = got.column("amount")
    assert amount is not None
    assert amount.null_count == 1
    assert amount.min == 1.0
    assert amount.max == 20.0


def test_profile_can_be_scoped_to_one_partition(lake: Path, partition_spec):
    """Partition scoping is what makes continuous profiling affordable."""
    ds = normalize(str(lake))
    store = LocalStorage()
    store.declare(ds, partition_spec)

    key = KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
    got = store.profile(ds, partition=key)
    assert got.row_count == 3
    assert got.file_count == 1


def test_columns_without_statistics_report_unknown_rather_than_zero(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "nostats.parquet"
    pq.write_table(pa.table({"x": pa.array([1, 2, 3])}), path, write_statistics=False)

    got = profile_parquet([path], dataset=normalize(str(tmp_path)))
    column = got.column("x")
    assert column is not None
    assert column.null_count is None  # unknown, not "no nulls"
    assert column.min is None


def test_drift_flags_a_removed_column(lake: Path, tmp_path: Path, partition_spec):
    import pyarrow as pa
    import pyarrow.parquet as pq

    before = profile_parquet(list(lake.rglob("*.parquet")), dataset=normalize(str(lake)))
    shrunk = tmp_path / "after"
    shrunk.mkdir()
    pq.write_table(pa.table({"id": pa.array(["a"] * 8)}), shrunk / "p.parquet")
    after = profile_parquet(list(shrunk.rglob("*.parquet")), dataset=normalize(str(shrunk)))

    findings = drift(before, after, min_rows=1)
    kinds = {(f.kind, f.severity) for f in findings}
    assert ("column_removed", Severity.ERROR) in kinds


def test_drift_flags_a_null_rate_jump(tmp_path: Path):
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_partition(a_root, dt=date(2026, 3, 14), region="eu", amounts=[1.0] * 2000)
    write_partition(b_root, dt=date(2026, 3, 14), region="eu", amounts=[1.0, None] * 1000)

    before = profile_parquet(list(a_root.rglob("*.parquet")), dataset=normalize(str(a_root)))
    after = profile_parquet(list(b_root.rglob("*.parquet")), dataset=normalize(str(b_root)))

    findings = drift(before, after)
    assert any(f.kind == "null_rate_shift" and f.severity is Severity.WARN for f in findings)


def test_small_partitions_downgrade_instead_of_paging_someone(tmp_path: Path):
    """A threshold without a sample-size guard is a noise generator."""
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    write_partition(a_root, dt=date(2026, 3, 14), region="eu", amounts=[1.0] * 10)
    write_partition(b_root, dt=date(2026, 3, 14), region="eu", amounts=[1.0, None] * 5)

    before = profile_parquet(list(a_root.rglob("*.parquet")), dataset=normalize(str(a_root)))
    after = profile_parquet(list(b_root.rglob("*.parquet")), dataset=normalize(str(b_root)))

    findings = drift(before, after)
    shifts = [f for f in findings if f.kind == "null_rate_shift"]
    assert shifts and all(f.severity is Severity.INFO for f in shifts)


def test_identical_profiles_produce_no_findings(lake: Path):
    files = list(lake.rglob("*.parquet"))
    a = profile_parquet(files, dataset=normalize(str(lake)))
    b = profile_parquet(files, dataset=normalize(str(lake)))
    assert drift(a, b, min_rows=1) == []
