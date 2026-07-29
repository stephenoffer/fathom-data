"""Filesystem abstraction.

The provider-specific normalization is what these tests exist for. Every cloud
spells etags and modification times differently, and getting one wrong does not
raise — it silently reports that nothing changed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import fsspec
import pytest

from fathom.adapters.fs import (
    FileInfo,
    FsspecFileSystem,
    clear_cache,
    data_files,
    filesystem_for,
    is_remote,
    join,
    split_protocol,
)
from fathom.core.errors import StorageAccessError


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


# -- URI handling --------------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://bucket/key", ("s3", "bucket/key")),
        ("gs://bucket/key", ("gs", "bucket/key")),
        ("abfss://c@acct/key", ("abfss", "c@acct/key")),
        ("/local/path", ("file", "/local/path")),
        ("relative/path", ("file", "relative/path")),
        ("S3://Bucket/Key", ("s3", "Bucket/Key")),
    ],
)
def test_protocol_splitting(uri, expected):
    assert split_protocol(uri) == expected


def test_remote_detection():
    assert is_remote("s3://b/k")
    assert not is_remote("/tmp/x")
    assert not is_remote("file:///tmp/x")


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("s3://bucket", "a", "b"), "s3://bucket/a/b"),
        (("s3://bucket/", "/a/", "/b/"), "s3://bucket/a/b"),
        (("/tmp", "a"), "/tmp/a"),
        (("s3://bucket", ""), "s3://bucket"),
    ],
)
def test_joining_does_not_mangle_the_protocol(parts, expected):
    assert join(*parts) == expected


def test_suffix_extraction():
    assert FileInfo(path="s3://b/dt=1/part-0.parquet").suffix == ".parquet"
    assert FileInfo(path="s3://b/noext").suffix == ""


# -- local ---------------------------------------------------------------------


def test_lists_and_reads_local_files(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"x" * 10)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.parquet").write_bytes(b"y" * 20)

    fs = filesystem_for(str(tmp_path))
    found = sorted(fs.ls(str(tmp_path)), key=lambda i: i.path)

    assert len(found) == 2
    assert [i.size for i in found] == [10, 20]
    assert all(i.modified is not None for i in found)
    assert fs.read_bytes(str(tmp_path / "a.parquet")) == b"x" * 10


def test_non_recursive_listing_stays_shallow(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.parquet").write_bytes(b"y")

    fs = filesystem_for(str(tmp_path))
    shallow = [i.path for i in fs.ls(str(tmp_path), recursive=False)]
    assert len(shallow) == 1


def test_listing_a_missing_prefix_is_empty_not_an_error(tmp_path):
    fs = filesystem_for(str(tmp_path))
    assert list(fs.ls(str(tmp_path / "nope"))) == []


def test_data_files_skips_sidecars_and_hidden_entries(tmp_path):
    (tmp_path / "part-0.parquet").write_bytes(b"x")
    (tmp_path / "_SUCCESS").write_bytes(b"")
    (tmp_path / ".hidden.parquet").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    fs = filesystem_for(str(tmp_path))
    found = data_files(fs, str(tmp_path))
    assert [i.path.rsplit("/", 1)[-1] for i in found] == ["part-0.parquet"]


# -- object storage (memory:// stands in for S3) --------------------------------


def test_object_storage_round_trip():
    # memory:// is process-global, so this needs a root no other test writes under.
    root = "memory://fs-round-trip"
    mem = fsspec.filesystem("memory")
    mem.pipe_file("/fs-round-trip/dt=2026-03-14/part-0.parquet", b"payload")

    fs = filesystem_for(root)
    assert fs.is_dir(root)
    found = list(fs.ls(root))
    assert len(found) == 1
    assert fs.read_bytes(found[0].path) == b"payload"


def test_a_file_is_never_a_directory(tmp_path):
    """The prefix fallback reports a file as a directory unless we check first."""
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    fs = filesystem_for(str(path))
    assert not fs.is_dir(str(path))
    assert fs.is_dir(str(tmp_path))


def test_an_object_key_is_never_a_directory():
    mem = fsspec.filesystem("memory")
    mem.pipe_file("/bucket/key.json", b"{}")
    fs = filesystem_for("memory://bucket")
    assert not fs.is_dir("memory://bucket/key.json")
    assert fs.is_dir("memory://bucket")


def test_prefix_with_objects_under_it_counts_as_a_directory():
    """Object stores have no directories; a prefix is one if anything lives under it."""
    mem = fsspec.filesystem("memory")
    mem.pipe_file("/deep/a/b/c.parquet", b"x")

    fs = filesystem_for("memory://deep")
    assert fs.is_dir("memory://deep/a")
    assert not fs.is_dir("memory://deep/nonexistent")


# -- provider normalization ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"name": "k", "ETag": '"abc123"'}, "abc123"),  # S3 quotes its ETag
        ({"name": "k", "md5Hash": "deadbeef"}, "deadbeef"),  # GCS
        ({"name": "k", "content_md5": "cafe"}, "cafe"),  # Azure
        ({"name": "k"}, None),
    ],
)
def test_etag_spellings_normalize(raw, expected):
    fs = FsspecFileSystem("memory")
    assert fs._to_info(raw).etag == expected


@pytest.mark.parametrize(
    "raw",
    [
        {"name": "k", "LastModified": datetime(2026, 3, 14, tzinfo=UTC)},
        {"name": "k", "mtime": 1773446400.0},
        {"name": "k", "last_modified": "2026-03-14T00:00:00Z"},
        {"name": "k", "updated": "2026-03-14T00:00:00+00:00"},
    ],
)
def test_modification_time_spellings_normalize(raw):
    fs = FsspecFileSystem("memory")
    got = fs._to_info(raw).modified
    assert got is not None and got.tzinfo is not None


def test_naive_timestamps_are_treated_as_utc():
    fs = FsspecFileSystem("memory")
    got = fs._to_info({"name": "k", "LastModified": datetime(2026, 3, 14)}).modified
    assert got is not None and got.tzinfo is UTC


def test_unparseable_timestamps_become_none_not_an_exception():
    fs = FsspecFileSystem("memory")
    assert fs._to_info({"name": "k", "mtime": "not a time"}).modified is None


# -- errors --------------------------------------------------------------------


def test_credential_failures_are_actionable():
    """A credential problem reported as "nothing found" sends people the wrong way."""
    with pytest.raises(StorageAccessError) as caught:
        filesystem_for("s3://fathom-test-bucket-does-not-exist").ls(
            "s3://fathom-test-bucket-does-not-exist/x"
        ).__next__()

    message = str(caught.value).lower()
    assert "cannot read s3://" in message
    # Offline CI may fail at the endpoint instead of at signing; both must be actionable.
    assert "credentials" in message or "endpoint" in message or "could not reach" in message


def test_storage_errors_name_the_uri():
    with pytest.raises(StorageAccessError) as caught:
        filesystem_for("s3://another-fake-bucket").read_bytes("s3://another-fake-bucket/key")
    assert caught.value.uri == "s3://another-fake-bucket/key"


# -- caching -------------------------------------------------------------------


def test_filesystems_are_reused_per_protocol_and_options():
    """A new client per dataset means a TLS handshake per partition."""
    assert filesystem_for("/tmp/a") is filesystem_for("/tmp/b")
    assert filesystem_for("memory://x") is not filesystem_for("/tmp/a")


def test_different_options_get_different_filesystems():
    a = filesystem_for("memory://x")
    b = filesystem_for("memory://x", skip_instance_cache=True)
    assert a is not b


def test_cache_can_be_cleared_for_credential_rotation():
    first = filesystem_for("/tmp/a")
    clear_cache()
    assert filesystem_for("/tmp/a") is not first
