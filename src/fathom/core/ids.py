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

    Args:
        uri: Any storage location — ``s3://``, ``gs://``, ``abfss://``, ``hdfs://``,
            a bare path, or a DBFS mount.
        mounts: DBFS-style mount points to their backing storage URIs, so
            ``/dbfs/mnt/lake/x`` resolves to the same identity Spark writes to.

    Example:
        >>> normalize_path("s3a://lake//raw/events/")
        DatasetId('s3://lake', 'raw/events')
        >>> normalize_path("/tmp/lake/events")
        DatasetId('file', '/tmp/lake/events')
        >>> normalize_path("/dbfs/mnt/lake/raw", mounts={"/mnt/lake": "s3://lake"})
        DatasetId('s3://lake', 'raw')
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

    Args:
        name: A table reference, dotted and optionally quoted.
        system: Which platform's folding rules apply — ``snowflake``, ``databricks``,
            ``bigquery``, ``duckdb``, and so on.
        instance: The account, workspace, or project. Two warehouses with a table of
            the same name are two datasets, and omitting this merges them.
        default_database: Prefix for references that name fewer than three parts.
        default_schema: Prefix for references that name only the table.

    Example:
        >>> normalize_table("orders", system="snowflake", instance="xy12345",
        ...                 default_database="prod", default_schema="sales")
        DatasetId('snowflake://xy12345', 'PROD.SALES.ORDERS')
        >>> normalize_table('db.schema."MixedCase"', system="snowflake")
        DatasetId('snowflake', 'DB.SCHEMA.MixedCase')
        >>> normalize_table("DB.Schema.Orders", system="duckdb")
        DatasetId('duckdb', 'db.schema.orders')
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

    Start here. This is the one function that takes whatever a user, a query log, or
    another tool wrote and produces the identity everything else is keyed on.

    Example:
        >>> normalize("s3a://lake/raw/events")
        DatasetId('s3://lake', 'raw/events')
        >>> normalize("raw.events", system="duckdb")
        DatasetId('duckdb', 'raw.events')
    """
    looks_like_uri = bool(re.match(r"^[a-z0-9+.-]+://", ref.strip(), re.I))
    looks_like_path = ref.startswith("/") or ref.startswith("dbfs:")
    if looks_like_uri or looks_like_path:
        return normalize_path(ref, mounts=mounts)
    if system is None:
        raise ValueError(
            f"{ref!r} is not a URI, so it must be a table — but no `system` was given, "
            f"and {ref!r} in Snowflake is not the same dataset as {ref!r} in DuckDB. "
            f"Pass `system=` (e.g. normalize({ref!r}, system='duckdb')), or set the "
            f"top-level `system:` key in fathom.yml so the project supplies it"
        )
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

    Until the two are aliased they are two nodes, and a plan seeded at one reaches
    nothing that reads the other — a graph that looks complete and propagates
    nothing. `fathom doctor` reports pairs that look like they should be aliased.

    Example:
        >>> registry = AliasRegistry()
        >>> hive = DatasetId("hive", "raw.events")
        >>> bytes_ = DatasetId("s3://lake", "raw/events")
        >>> registry.alias(hive, bytes_)
        >>> registry.resolve(hive)
        DatasetId('s3://lake', 'raw/events')
        >>> hive in registry, len(registry)
        (True, 1)
    """

    _canonical: dict[DatasetId, DatasetId] = field(default_factory=dict)

    def alias(self, alias: DatasetId, canonical: DatasetId) -> None:
        """Declare that two identities are the same dataset.

        Aliasing something to itself is a no-op rather than an error, so a caller
        looping over declarations does not have to filter them.

        Args:
            alias: The identity to redirect.
            canonical: The identity to redirect it to. Resolved first, so chains
                collapse and every alias points at the final target.

        Raises:
            ValueError: The declaration would form a cycle, leaving no canonical
                identity for either side.
        """
        if alias == canonical:
            return
        target = self.resolve(canonical)
        if target == alias:
            raise ValueError(
                f"aliasing {alias} to {canonical} would form a cycle: {canonical} "
                f"already resolves back to {alias}. One of the two has to be the "
                f"canonical identity — alias the other to it, not both ways"
            )
        self._canonical[alias] = target

    def resolve(self, ds: DatasetId) -> DatasetId:
        """Follow aliases to the canonical identity, refusing to loop.

        Safe to call on anything: an identity nobody aliased returns unchanged.
        """
        seen: set[DatasetId] = set()
        cur = ds
        while cur in self._canonical:
            if cur in seen:
                raise ValueError(
                    f"alias cycle at {cur}; the declarations form a loop with no "
                    f"canonical identity. Aliases involved: "
                    f"{', '.join(sorted(str(d) for d in seen))}"
                )
            seen.add(cur)
            cur = self._canonical[cur]
        return cur

    def __contains__(self, ds: object) -> bool:
        """True when this identity has been declared an alias of another."""
        return ds in self._canonical

    def __len__(self) -> int:
        return len(self._canonical)

    def items(self) -> list[tuple[DatasetId, DatasetId]]:
        """Every declared alias and what it resolves to, sorted for stable output."""
        return sorted(self._canonical.items(), key=lambda kv: (str(kv[0]), str(kv[1])))


# Namespaces that address bytes rather than a catalog entry.
_PATH_NAMESPACES = re.compile(r"^(file|s3|gs|abfss|hdfs|r2|memory)(://|$)", re.I)


def is_path_dataset(ds: DatasetId) -> bool:
    """True when this dataset lives in a filesystem or object store.

    The test for "can I open bytes for this?" — which decides whether profiling
    reads Parquet footers or asks a warehouse to compute statistics for us.

    Example:
        >>> is_path_dataset(DatasetId("s3://lake", "raw/events"))
        True
        >>> is_path_dataset(DatasetId("snowflake://xy12345", "db.schema.orders"))
        False
    """
    return bool(_PATH_NAMESPACES.match(ds.namespace))


def dataset_uri(ds: DatasetId) -> str:
    """The URI a filesystem can open.

    Inverse of `normalize_path`. Raises for catalog datasets, because a Snowflake
    table has no path and silently returning something path-shaped would send an
    adapter looking for bytes that do not exist.

    Example:
        >>> dataset_uri(DatasetId("s3://lake", "raw/events"))
        's3://lake/raw/events'
        >>> dataset_uri(DatasetId("file", "/tmp/lake/events"))
        '/tmp/lake/events'
    """
    if not is_path_dataset(ds):
        raise ValueError(
            f"{ds} is a catalog dataset in {ds.namespace!r}, not a location, so there "
            f"are no bytes to open. Reach it through the engine adapter for that "
            f"system instead — or, if it is an external table over object storage, "
            f"declare the alias so both spellings resolve to one dataset"
        )
    if ds.namespace == "file":
        return ds.name
    return f"{ds.namespace}/{ds.name}" if ds.name else ds.namespace
