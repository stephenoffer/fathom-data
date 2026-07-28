"""Filesystem abstraction over local disk and object storage.

Every adapter that touches bytes goes through this, so `s3://`, `gs://`, `abfss://`,
and a local path are the same code path. fsspec does the protocol work; this module
exists to normalize the parts fsspec deliberately leaves provider-specific.

Three of those parts matter to change detection:

- **etags** are spelled differently everywhere (`ETag` on S3, `md5Hash` on GCS,
  `content_md5` on Azure) and are the only reliable "did this object change" signal
  when timestamps collide.
- **modification times** come back as floats, strings, or datetimes depending on
  provider, and sometimes not at all.
- **directories** are a fiction on object storage. A prefix exists if anything is
  under it, which is not what `isdir` means locally.

Getting these wrong does not raise; it silently reports nothing changed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, BinaryIO, Protocol, runtime_checkable

from .errors import StorageAccessError, hint_for

__all__ = [
    "FileInfo",
    "FileSystem",
    "FsspecFileSystem",
    "filesystem_for",
    "is_remote",
    "join",
    "split_protocol",
]

_PROTOCOL = re.compile(r"^(?P<scheme>[a-z0-9+.-]+)://", re.I)

# Spellings of a content hash, best first. S3 quotes its ETag; strip that.
_ETAG_KEYS = ("ETag", "etag", "md5Hash", "content_md5", "md5", "checksum")
_MTIME_KEYS = ("LastModified", "last_modified", "mtime", "updated", "creation_time", "modified")


def split_protocol(uri: str) -> tuple[str, str]:
    """Return `(scheme, remainder)`. A bare path yields `("file", path)`."""
    match = _PROTOCOL.match(uri)
    if not match:
        return "file", uri
    scheme = match.group("scheme").lower()
    return scheme, uri[match.end() :]


def is_remote(uri: str) -> bool:
    return split_protocol(uri)[0] not in {"file", "local"}


def join(base: str, *parts: str) -> str:
    """Join URI segments without mangling the protocol separator."""
    out = base.rstrip("/")
    for part in parts:
        cleaned = str(part).strip("/")
        if cleaned:
            out = f"{out}/{cleaned}"
    return out


@dataclass(frozen=True)
class FileInfo:
    """One object, with the fields change detection actually needs."""

    path: str
    size: int = 0
    etag: str | None = None
    modified: datetime | None = None

    @property
    def suffix(self) -> str:
        name = self.path.rsplit("/", 1)[-1]
        return name[name.rfind(".") :].lower() if "." in name else ""


@runtime_checkable
class FileSystem(Protocol):
    """The surface adapters need. Deliberately small so new backends are cheap."""

    def exists(self, uri: str) -> bool: ...
    def is_dir(self, uri: str) -> bool: ...
    def ls(self, uri: str, *, recursive: bool = True) -> Iterator[FileInfo]: ...
    def info(self, uri: str) -> FileInfo: ...
    def read_bytes(self, uri: str) -> bytes: ...
    def read_text(self, uri: str) -> str: ...
    def open(self, uri: str) -> BinaryIO: ...
    def delete(self, uri: str) -> None: ...


def _coerce_time(value: Any) -> datetime | None:
    """Normalize whatever a provider calls a modification time into a datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _coerce_etag(info: Mapping[str, Any]) -> str | None:
    for key in _ETAG_KEYS:
        value = info.get(key)
        if value:
            return str(value).strip('"')
    return None


@contextmanager
def _wrapped(uri: str) -> Iterator[None]:
    """Turn provider exceptions into actionable ones.

    `FileNotFoundError` passes through: callers legitimately treat absence as empty.
    Everything else becomes a `StorageAccessError`, because a credential problem
    reported as "nothing here" sends people debugging the wrong thing entirely.
    """
    try:
        yield
    except FileNotFoundError:
        raise
    except StorageAccessError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider SDKs raise their own hierarchies
        raise StorageAccessError(uri, f"{type(exc).__name__}: {exc}", hint_for(exc)) from exc


class FsspecFileSystem:
    """A `FileSystem` backed by fsspec.

    Storage options (credentials, endpoints, region) pass straight through, so an
    S3-compatible endpoint like MinIO needs no special handling here.
    """

    def __init__(self, protocol: str = "file", **storage_options: Any) -> None:
        import fsspec

        with _wrapped(f"{protocol}://"):
            self._fs = fsspec.filesystem(protocol, **storage_options)
        self.protocol = protocol
        self.storage_options = storage_options

    def __repr__(self) -> str:
        return f"FsspecFileSystem({self.protocol!r})"

    def _strip(self, uri: str) -> str:
        """fsspec accepts full URIs, but mixing styles confuses path joins."""
        scheme, remainder = split_protocol(uri)
        return uri if scheme == self.protocol else remainder

    def exists(self, uri: str) -> bool:
        with _wrapped(uri):
            return bool(self._fs.exists(self._strip(uri)))

    def is_dir(self, uri: str) -> bool:
        target = self._strip(uri)
        with _wrapped(uri):
            try:
                # Check this first: the prefix fallback below reports a file as a
                # directory, because listing a file returns the file itself.
                if self._fs.isfile(target):
                    return False
            except FileNotFoundError:
                return False
            try:
                if self._fs.isdir(target):
                    return True
            except FileNotFoundError:
                return False
            # Object stores have no directories. A prefix is a directory if anything
            # lives under it, which `isdir` does not always report.
            try:
                return bool(next(iter(self._fs.find(target, maxdepth=1)), None))
            except FileNotFoundError:
                return False

    def _to_info(self, raw: Mapping[str, Any] | str) -> FileInfo:
        if isinstance(raw, str):
            return FileInfo(path=raw)
        path = str(raw.get("name") or raw.get("Key") or "")
        modified = None
        for key in _MTIME_KEYS:
            modified = _coerce_time(raw.get(key))
            if modified is not None:
                break
        return FileInfo(
            path=path,
            size=int(raw.get("size") or raw.get("Size") or 0),
            etag=_coerce_etag(raw),
            modified=modified,
        )

    def info(self, uri: str) -> FileInfo:
        with _wrapped(uri):
            return self._to_info(self._fs.info(self._strip(uri)))

    def ls(self, uri: str, *, recursive: bool = True) -> Iterator[FileInfo]:
        target = self._strip(uri)
        with _wrapped(uri):
            if not self._fs.exists(target):
                return
            try:
                entries = (
                    self._fs.find(target, detail=True)
                    if recursive
                    else self._fs.ls(target, detail=True)
                )
            except FileNotFoundError:
                return
        raw_values = entries.values() if isinstance(entries, dict) else entries
        for raw in raw_values:
            info = self._to_info(raw)
            if isinstance(raw, dict) and raw.get("type") == "directory":
                continue
            if info.path:
                yield info

    def read_bytes(self, uri: str) -> bytes:
        with _wrapped(uri):
            return bytes(self._fs.cat_file(self._strip(uri)))

    def read_text(self, uri: str) -> str:
        return self.read_bytes(uri).decode("utf-8")

    def open(self, uri: str) -> BinaryIO:
        with _wrapped(uri):
            handle: BinaryIO = self._fs.open(self._strip(uri), "rb")
            return handle

    def delete(self, uri: str) -> None:
        with _wrapped(uri):
            self._fs.rm_file(self._strip(uri))

    def unstrip(self, path: str) -> str:
        """Re-attach the protocol to a path fsspec returned bare."""
        if self.protocol in {"file", "local"} or _PROTOCOL.match(path):
            return path
        return f"{self.protocol}://{path.lstrip('/')}"


_CACHE: dict[tuple[str, tuple[tuple[str, Any], ...]], FsspecFileSystem] = {}


def filesystem_for(uri: str, **storage_options: Any) -> FsspecFileSystem:
    """Resolve a filesystem for a URI, reusing connections per protocol and options.

    Caching matters on object storage: constructing a new client per dataset means a
    new TLS handshake and credential resolution for every partition we touch.
    """
    protocol, _ = split_protocol(uri)
    if protocol == "local":
        protocol = "file"
    key = (protocol, tuple(sorted(storage_options.items())))
    cached = _CACHE.get(key)
    if cached is None:
        cached = FsspecFileSystem(protocol, **storage_options)
        _CACHE[key] = cached
    return cached


def clear_cache() -> None:
    """Drop cached filesystems. Mainly for tests and credential rotation."""
    _CACHE.clear()


def data_files(
    fs: FileSystem, root: str, suffixes: Iterable[str] = (".parquet", ".pq")
) -> list[FileInfo]:
    """Data files under a prefix, ignoring metadata sidecars and hidden entries."""
    wanted = {s.lower() for s in suffixes}
    out = []
    for info in fs.ls(root):
        name = info.path.rsplit("/", 1)[-1]
        if name.startswith((".", "_")):
            continue
        if info.suffix in wanted:
            out.append(info)
    return sorted(out, key=lambda i: i.path)
