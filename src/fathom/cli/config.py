"""Project configuration: `fathom.yml`.

The point of this file is that partition specs have to live *somewhere* durable.
They cannot be fully inferred — Snowflake has no partitions to read, Delta records
column names but not grain, and every wrong guess silently changes what a rebuild
covers. Passing them as `--spec` flags on every invocation is how they drift.

Design rules, each earned:

- **Fail loudly on anything unrecognized.** A typo in a key name that is silently
  ignored produces a config that looks right and plans wrong. Unknown keys raise.
- **Never hold secrets.** `storage_options` and adapter options take `${ENV_VAR}`
  references, resolved at load. A config file that must contain a warehouse password
  is a config file nobody commits, which defeats the purpose.
- **Paths resolve relative to the config file**, not the working directory, so the
  same project works from any cwd.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..core.errors import ConfigError
from ..core.grains import Grain
from ..core.ids import normalize
from ..core.paths import PathTemplate
from ..core.types import UNPARTITIONED, DatasetId, PartitionField, PartitionSpec

__all__ = [
    "CONFIG_NAMES",
    "ContractConfig",
    "DatasetConfig",
    "LineageConfig",
    "PolicyConfig",
    "ProjectConfig",
    "PublicationConfig",
    "find_config",
    "load_config",
    "parse_config",
]

CONFIG_NAMES = ("fathom.yml", "fathom.yaml", ".fathom.yml")
DEFAULT_STORE = ".fathom/fathom.db"

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

_TOP_LEVEL = {
    "version",
    "store",
    "system",
    "instance",
    "storage_options",
    "adapters",
    "datasets",
    "lineage",
    "policies",
    "publications",
    "contracts",
}
_DATASET_KEYS = {
    "name",
    "adapter",
    "partition",
    "template",
    "watermark",
    "location",
    "sql",
    "model",
    "role",
}
_LINEAGE_KEYS = {"type", "paths", "dialect", "manifest", "events", "adapter", "since"}
_POLICY_KEYS = {"dataset", "forbid", "require", "reason"}
_PUBLICATION_KEYS = {"name", "kind", "instance", "inputs"}
_CONTRACT_KEYS = {"dataset", "producer", "consumers", "columns", "max_staleness", "note"}

# Sink kinds a publication may declare. Kept as a literal set rather than derived from
# `graph.sinks.SinkKind` so that `cli.config` stays parseable without importing the
# graph layer, and so a renamed enum member fails a config test rather than silently
# accepting a value nothing handles.
_SINK_KINDS = frozenset({"dashboard", "report", "filing", "export", "endpoint", "notebook"})


def _expand(value: Any) -> Any:
    """Resolve `${VAR}` and `${VAR:-default}` references, recursively."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, fallback = match.group(1), match.group(2)
            found = os.environ.get(name)
            if found is None:
                if fallback is None:
                    raise ConfigError(
                        f"{name} is referenced in the config but not set in the "
                        f"environment; export it, or write ${{{name}:-default}}"
                    )
                return fallback
            return found

        return _ENV_REF.sub(replace, value)
    if isinstance(value, Mapping):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_DURATION_UNITS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def _parse_duration(value: Any, where: str) -> timedelta | None:
    """Parse `6h`, `30m`, `2d`. A bare number is rejected rather than assumed.

    Every config format that accepts a bare number for a duration ends up with two
    readers disagreeing about the unit, and the disagreement is silent.
    """
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value
    match = _DURATION.match(str(value))
    if not match:
        raise ConfigError(
            f"{where}: {value!r} is not a duration. Use a number and a unit — "
            "30m, 6h, 2d, 1w — because a bare number is ambiguous"
        )
    amount, unit = float(match.group(1)), match.group(2).lower()
    return timedelta(**{_DURATION_UNITS[unit]: amount})


def _check_keys(blob: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(blob) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {where}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(allowed))}"
        )


def _parse_partition(entries: Any, where: str) -> PartitionSpec:
    """Accept `[dt]`, `[{field: dt, grain: day}]`, or a bare string."""
    if entries is None:
        return UNPARTITIONED
    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, Sequence):
        raise ConfigError(f"{where}: partition must be a list")

    fields: list[PartitionField] = []
    for entry in entries:
        if isinstance(entry, str):
            fields.append(PartitionField.value(entry))
            continue
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where}: partition entries must be strings or mappings")
        name = entry.get("field") or entry.get("name")
        if not name:
            raise ConfigError(f"{where}: partition entry is missing `field`")
        grain = entry.get("grain") or entry.get("granularity")
        if grain:
            try:
                fields.append(PartitionField.time(str(name), Grain.parse(str(grain))))
            except ValueError as exc:
                raise ConfigError(f"{where}: {exc}") from exc
        else:
            fields.append(PartitionField.value(str(name)))
    try:
        return PartitionSpec.of(*fields)
    except ValueError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


@dataclass(frozen=True)
class DatasetConfig:
    """One declared dataset: a source, a model, or both."""

    dataset: DatasetId
    raw_name: str
    adapter: str = ""
    spec: PartitionSpec = UNPARTITIONED
    template: PathTemplate | None = None
    watermark: str | None = None
    location: str | None = None
    sql: str | None = None
    sql_path: Path | None = None
    role: str = ""

    @property
    def is_model(self) -> bool:
        return self.sql is not None or self.sql_path is not None


@dataclass(frozen=True)
class LineageConfig:
    """Where lineage comes from."""

    kind: str  # sql | dbt | openlineage | adapter
    paths: tuple[Path, ...] = ()
    dialect: str = ""
    manifest: Path | None = None
    events: str | None = None
    adapter: str = ""


@dataclass(frozen=True)
class PolicyConfig:
    dataset: DatasetId
    forbid: frozenset[str] = frozenset()
    require: frozenset[str] = frozenset()
    reason: str = ""


@dataclass(frozen=True)
class ContractConfig:
    """One team's promise to another about a dataset.

    `max_staleness` is stored as a `timedelta` so the CLI and the library agree on
    what "6h" means; parsing it here rather than at each call site is what stops two
    readers disagreeing about whether a bare number was seconds or hours.
    """

    dataset: DatasetId
    producer: str
    consumers: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    max_staleness: timedelta | None = None
    note: str = ""


@dataclass(frozen=True)
class PublicationConfig:
    """One published artefact and the datasets it draws on.

    Declared rather than discovered: no BI tool exposes its queries uniformly, and a
    guessed dashboard dependency is worse than an absent one, because a restatement
    notice built on a guess is a notice that names the wrong people.
    """

    name: str
    kind: str
    instance: str = "local"
    inputs: tuple[DatasetId, ...] = ()


@dataclass
class ProjectConfig:
    """A parsed and validated `fathom.yml`."""

    root: Path = field(default_factory=Path.cwd)
    path: Path | None = None
    version: int = 1
    store: Path = field(default_factory=lambda: Path(DEFAULT_STORE))
    system: str = "duckdb"
    instance: str | None = None
    storage_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    adapters: dict[str, dict[str, Any]] = field(default_factory=dict)
    datasets: list[DatasetConfig] = field(default_factory=list)
    lineage: list[LineageConfig] = field(default_factory=list)
    policies: list[PolicyConfig] = field(default_factory=list)
    publications: list[PublicationConfig] = field(default_factory=list)
    contracts: list[ContractConfig] = field(default_factory=list)

    # -- lookups ---------------------------------------------------------------

    @property
    def specs(self) -> dict[DatasetId, PartitionSpec]:
        return {d.dataset: d.spec for d in self.datasets if d.spec.fields}

    @property
    def models(self) -> list[DatasetConfig]:
        return [d for d in self.datasets if d.is_model]

    @property
    def sources(self) -> list[DatasetConfig]:
        return [d for d in self.datasets if not d.is_model]

    def dataset(self, name: str | DatasetId) -> DatasetConfig | None:
        """Look up a declared dataset by identity or by the name in the config."""
        resolved = name if isinstance(name, DatasetId) else self.resolve(name)
        return next((d for d in self.datasets if d.dataset == resolved), None)

    def resolve(self, name: str) -> DatasetId:
        """Resolve a name the way the config would, path or table.

        A URI is passed through untouched. Routing it via `Path` would collapse the
        double slash — `Path("file:///a/b")` stringifies to `file:/a/b` — and the
        result no longer parses as a URI at all.
        """
        if "://" in name:
            return normalize(name)
        if name.startswith((".", "/", "~")):
            return normalize(str((self.root / name).expanduser().resolve()))
        return normalize(name, system=self.system, instance=self.instance)

    def options_for(self, protocol: str) -> dict[str, Any]:
        return dict(self.storage_options.get(protocol, {}))

    def sql_for(self, entry: DatasetConfig) -> str | None:
        if entry.sql is not None:
            return entry.sql
        if entry.sql_path is not None:
            return entry.sql_path.read_text()
        return None


def _resolve_path(root: Path, value: str) -> Path:
    """Paths are relative to the config file, so a project works from any cwd."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate)


def parse_config(blob: Mapping[str, Any], *, root: Path, path: Path | None = None) -> ProjectConfig:
    """Validate a config mapping. Unknown keys are errors, not warnings."""
    if not isinstance(blob, Mapping):
        raise ConfigError("the config file must contain a mapping at the top level")

    blob = _expand(dict(blob))
    _check_keys(blob, _TOP_LEVEL, "the config file")

    version = int(blob.get("version", 1))
    if version != 1:
        raise ConfigError(f"config version {version} is not supported by this release")

    config = ProjectConfig(
        root=root,
        path=path,
        version=version,
        store=_resolve_path(root, str(blob.get("store", DEFAULT_STORE))),
        system=str(blob.get("system", "duckdb")),
        instance=blob.get("instance"),
        storage_options={k: dict(v) for k, v in (blob.get("storage_options") or {}).items()},
        adapters={k: dict(v) for k, v in (blob.get("adapters") or {}).items()},
    )

    for index, entry in enumerate(blob.get("datasets") or []):
        where = f"datasets[{index}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where}: each dataset must be a mapping")
        _check_keys(entry, _DATASET_KEYS, where)
        name = entry.get("name")
        if not name:
            raise ConfigError(f"{where}: missing `name`")

        template = entry.get("template")
        sql_path = entry.get("model")
        config.datasets.append(
            DatasetConfig(
                dataset=config.resolve(str(name)),
                raw_name=str(name),
                adapter=str(entry.get("adapter") or ""),
                spec=_parse_partition(entry.get("partition"), where),
                template=PathTemplate(str(template)) if template else None,
                watermark=entry.get("watermark"),
                location=entry.get("location"),
                sql=entry.get("sql"),
                sql_path=_resolve_path(root, str(sql_path)) if sql_path else None,
                role=str(entry.get("role") or ""),
            )
        )

    seen: set[DatasetId] = set()
    for entry in config.datasets:
        if entry.dataset in seen:
            raise ConfigError(f"{entry.raw_name} is declared twice; a dataset may appear only once")
        seen.add(entry.dataset)

    for index, entry in enumerate(blob.get("lineage") or []):
        where = f"lineage[{index}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where}: each lineage source must be a mapping")
        _check_keys(entry, _LINEAGE_KEYS, where)
        kind = str(entry.get("type") or "")
        if kind not in {"sql", "dbt", "openlineage", "adapter"}:
            raise ConfigError(
                f"{where}: type must be one of sql, dbt, openlineage, adapter (got {kind!r})"
            )
        raw_paths = entry.get("paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        manifest = entry.get("manifest")
        config.lineage.append(
            LineageConfig(
                kind=kind,
                paths=tuple(_resolve_path(root, str(p)) for p in raw_paths),
                dialect=str(entry.get("dialect") or config.system),
                manifest=_resolve_path(root, str(manifest)) if manifest else None,
                events=entry.get("events"),
                adapter=str(entry.get("adapter") or ""),
            )
        )

    for index, entry in enumerate(blob.get("policies") or []):
        where = f"policies[{index}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where}: each policy must be a mapping")
        _check_keys(entry, _POLICY_KEYS, where)
        target = entry.get("dataset")
        if not target:
            raise ConfigError(f"{where}: missing `dataset`")
        forbid = entry.get("forbid") or []
        require = entry.get("require") or []
        config.policies.append(
            PolicyConfig(
                dataset=config.resolve(str(target)),
                forbid=frozenset(forbid if isinstance(forbid, list) else [forbid]),
                require=frozenset(require if isinstance(require, list) else [require]),
                reason=str(entry.get("reason") or f"declared in {path or 'config'}"),
            )
        )

    for index, entry in enumerate(blob.get("publications") or []):
        where = f"publications[{index}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where}: each publication must be a mapping")
        _check_keys(entry, _PUBLICATION_KEYS, where)
        name = entry.get("name")
        if not name:
            raise ConfigError(f"{where}: missing `name`")
        kind = str(entry.get("kind") or "dashboard").lower()
        if kind not in _SINK_KINDS:
            raise ConfigError(
                f"{where}: unknown kind {kind!r}. Valid kinds: {', '.join(sorted(_SINK_KINDS))}"
            )
        inputs = entry.get("inputs") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            raise ConfigError(
                f"{where}: a publication with no `inputs` records nothing; either name "
                "the datasets it draws on or remove it"
            )
        config.publications.append(
            PublicationConfig(
                name=str(name),
                kind=kind,
                instance=str(entry.get("instance") or "local"),
                inputs=tuple(config.resolve(str(i)) for i in inputs),
            )
        )

    for index, entry in enumerate(blob.get("contracts") or []):
        where = f"contracts[{index}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where}: each contract must be a mapping")
        _check_keys(entry, _CONTRACT_KEYS, where)
        target = entry.get("dataset")
        if not target:
            raise ConfigError(f"{where}: missing `dataset`")
        producer = entry.get("producer")
        if not producer:
            raise ConfigError(
                f"{where}: missing `producer`. A contract with no owner names nobody "
                "when it is breached, which is the one thing a contract adds"
            )
        consumers = entry.get("consumers") or []
        columns = entry.get("columns") or []
        config.contracts.append(
            ContractConfig(
                dataset=config.resolve(str(target)),
                producer=str(producer),
                consumers=tuple(
                    str(c) for c in (consumers if isinstance(consumers, list) else [consumers])
                ),
                columns=tuple(
                    str(c) for c in (columns if isinstance(columns, list) else [columns])
                ),
                max_staleness=_parse_duration(entry.get("max_staleness"), where),
                note=str(entry.get("note") or ""),
            )
        )

    return config


def find_config(start: Path | None = None) -> Path | None:
    """Search upward for a config file, the way git finds its root."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in CONFIG_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load a config file, searching upward when none is given."""
    resolved = Path(path).resolve() if path else find_config()
    if resolved is None:
        raise ConfigError(
            f"no config file found. Create one of {', '.join(CONFIG_NAMES)} in this "
            "directory or a parent, or pass --config"
        )
    if not resolved.is_file():
        raise ConfigError(f"{resolved} does not exist")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is a hard dependency
        raise ConfigError("reading a config file needs PyYAML: pip install pyyaml") from exc

    try:
        blob = yaml.safe_load(resolved.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved} is not valid YAML: {exc}") from exc

    return parse_config(blob, root=resolved.parent, path=resolved)
