"""Dataset identity normalization.

A Spark job writes ``s3a://lake/raw/events``. A Trino query reads the same bytes as
``hive.raw.events``. A notebook reads ``/dbfs/mnt/lake/raw/events``. Unless all three
collapse to one node, the dependency graph is three disconnected fragments and the
planner is useless.

Identities follow the OpenLineage dataset naming convention — a namespace locating
the system and a name locating the dataset inside it — so what we emit interoperates
with the rest of that ecosystem instead of inventing a fourth spelling.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .types import DatasetId

__all__ = [
    "AliasRegistry",
    "dataset_uri",
    "is_path_dataset",
    "normalize",
    "normalize_path",
    "normalize_table",
]

# Protocol spellings that address identical bytes.
_SCHEME_ALIASES = {
    "s3a": "s3",
    "s3n": "s3",
    "gcs": "gs",
    "abfs": "abfss",
    "wasb": "abfss",
    "wasbs": "abfss",
}

# Azure exposes the same account through two hostnames depending on protocol.
_AZURE_HOST = re.compile(r"^(?P<account>[^.]+)\.(dfs|blob)\.core\.windows\.net$", re.I)

# How each system folds unquoted identifiers, which decides whether `Orders` and
# `orders` are the same table.
_IDENTIFIER_CASE: dict[str, str | None] = {
    "snowflake": "upper",
    "databricks": "lower",
    "hive": "lower",
    "trino": "lower",
    "postgres": "lower",
    "redshift": "lower",
    "duckdb": "lower",
    "clickhouse": None,  # case-sensitive
    "bigquery": None,  # case-sensitive
}


def _clean_path(path: str) -> str:
    """Collapse duplicate separators and strip leading/trailing ones."""
    return re.sub(r"/{2,}", "/", path).strip("/")


def _resolve_mount(path: str, mounts: Mapping[str, str]) -> str | None:
    """Rewrite a DBFS-style mount path to the storage URI behind it.

    Longest prefix wins, so ``/mnt/lake/raw`` beats ``/mnt/lake``.
    """
    normalized = "/" + _clean_path(path)
    for mount in sorted(mounts, key=len, reverse=True):
        m = "/" + _clean_path(mount)
        if normalized == m or normalized.startswith(m + "/"):
            remainder = normalized[len(m) :].strip("/")
            target = mounts[mount].rstrip("/")
            return f"{target}/{remainder}" if remainder else target
    return None


def normalize_path(uri: str, *, mounts: Mapping[str, str] | None = None) -> DatasetId:
    """Normalize an object-storage or filesystem location.

    ``s3a://lake//raw/events/`` and ``s3://lake/raw/events`` both yield
    ``DatasetId("s3://lake", "raw/events")``.
    """
    raw = uri.strip()

    # DBFS is a virtual mount layer; resolve to real storage before anything else.
    if mounts:
        candidate = raw
        if candidate.startswith("dbfs:"):
            candidate = candidate[len("dbfs:") :]
        elif candidate.startswith("/dbfs/"):
            candidate = candidate[len("/dbfs") :]
        if candidate is not raw and not re.match(r"^[a-z0-9+.-]+://", candidate, re.I):
            resolved = _resolve_mount(candidate, mounts)
            if resolved is not None:
                raw = resolved

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "file").lower()
    scheme = _SCHEME_ALIASES.get(scheme, scheme)

    # Bare or file:// paths.
    if scheme == "file" or not parsed.netloc:
        path = parsed.path or raw
        return DatasetId(namespace="file", name="/" + _clean_path(path))

    if scheme == "abfss":
        # abfss://container@account.dfs.core.windows.net/path
        # wasbs://container@account.blob.core.windows.net/path
        userinfo, _, host = parsed.netloc.rpartition("@")
        container = userinfo or ""
        m = _AZURE_HOST.match(host)
        account = m.group("account").lower() if m else host.lower()
        namespace = f"abfss://{container.lower()}@{account}" if container else f"abfss://{account}"
        return DatasetId(namespace=namespace, name=_clean_path(parsed.path))

    if scheme in {"hdfs", "webhdfs"}:
        return DatasetId(namespace=f"hdfs://{parsed.netloc.lower()}", name=_clean_path(parsed.path))

    # s3, gs, r2, and anything else bucket-shaped.
    return DatasetId(namespace=f"{scheme}://{parsed.netloc.lower()}", name=_clean_path(parsed.path))


def normalize_table(
    name: str,
    *,
    system: str,
    instance: str | None = None,
    default_database: str | None = None,
    default_schema: str | None = None,
) -> DatasetId:
    """Normalize a catalog table reference to ``system://instance`` + dotted name.

    Quoted identifiers keep their case; unquoted ones fold according to the system's
    rules, so Snowflake's ``orders`` and ``ORDERS`` resolve to one dataset.
    """
    system = system.lower()
    fold = _IDENTIFIER_CASE.get(system)

    parts: list[str] = []
    for part in re.findall(r'"[^"]*"|`[^`]*`|\[[^\]]*\]|[^.]+', name.strip()):
        if part[:1] in {'"', "`", "["}:
            parts.append(part[1:-1])
        elif fold == "upper":
            parts.append(part.upper())
        elif fold == "lower":
            parts.append(part.lower())
        else:
            parts.append(part)

    def _fold(value: str) -> str:
        return value.upper() if fold == "upper" else value.lower() if fold == "lower" else value

    if len(parts) == 1:
        if default_schema:
            parts.insert(0, _fold(default_schema))
        if default_database and len(parts) < 3:
            parts.insert(0, _fold(default_database))
    elif len(parts) == 2 and default_database:
        parts.insert(0, _fold(default_database))

    namespace = f"{system}://{instance}" if instance else system
    return DatasetId(namespace=namespace, name=".".join(parts))


def normalize(
    ref: str,
    *,
    system: str | None = None,
    instance: str | None = None,
    mounts: Mapping[str, str] | None = None,
    default_database: str | None = None,
    default_schema: str | None = None,
) -> DatasetId:
    """Normalize either a storage location or a table reference.

    Dispatches on whether `ref` looks like a URI. A path always wins: `s3://a/b` is a
    location even if a `system` was supplied.
    """
    looks_like_uri = bool(re.match(r"^[a-z0-9+.-]+://", ref.strip(), re.I))
    looks_like_path = ref.startswith("/") or ref.startswith("dbfs:")
    if looks_like_uri or looks_like_path:
        return normalize_path(ref, mounts=mounts)
    if system is None:
        raise ValueError(f"{ref!r} is not a URI; pass system= to resolve it as a table")
    return normalize_table(
        ref,
        system=system,
        instance=instance,
        default_database=default_database,
        default_schema=default_schema,
    )


@dataclass
class AliasRegistry:
    """Declared equivalences between identities we cannot unify automatically.

    An external Hive table pointing at an S3 prefix is the common case: nothing in
    either reference reveals the connection, so a catalog adapter declares it.
    """

    _canonical: dict[DatasetId, DatasetId] = field(default_factory=dict)

    def alias(self, alias: DatasetId, canonical: DatasetId) -> None:
        if alias == canonical:
            return
        target = self.resolve(canonical)
        if target == alias:
            raise ValueError(f"alias cycle between {alias} and {canonical}")
        self._canonical[alias] = target

    def resolve(self, ds: DatasetId) -> DatasetId:
        seen: set[DatasetId] = set()
        cur = ds
        while cur in self._canonical:
            if cur in seen:
                raise ValueError(f"alias cycle at {cur}")
            seen.add(cur)
            cur = self._canonical[cur]
        return cur

    def __len__(self) -> int:
        return len(self._canonical)


# Namespaces that address bytes rather than a catalog entry.
_PATH_NAMESPACES = re.compile(r"^(file|s3|gs|abfss|hdfs|r2|memory)(://|$)", re.I)


def is_path_dataset(ds: DatasetId) -> bool:
    """True when this dataset lives in a filesystem or object store."""
    return bool(_PATH_NAMESPACES.match(ds.namespace))


def dataset_uri(ds: DatasetId) -> str:
    """The URI a filesystem can open.

    Inverse of `normalize_path`. Raises for catalog datasets, because a Snowflake
    table has no path and silently returning something path-shaped would send an
    adapter looking for bytes that do not exist.
    """
    if not is_path_dataset(ds):
        raise ValueError(f"{ds} is a catalog dataset, not a location; it has no URI to open")
    if ds.namespace == "file":
        return ds.name
    return f"{ds.namespace}/{ds.name}" if ds.name else ds.namespace
