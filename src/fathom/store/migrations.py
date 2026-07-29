"""Upgrading a store, and getting the data back out of one.

`sqlite.py` checks the schema version and refuses anything newer, which is correct
and leaves nowhere to go: there is no migration path, no backup, and no export. A
store that cannot be upgraded is a store people copy and abandon.

The design is deliberately unfashionable. Migrations are **forward-only** and
numbered, with no `down` step, because a down migration that drops a column throws
away data an operator cannot get back — and the one time it runs is during an
incident, when nobody is thinking clearly. Rolling back means restoring the backup
that `backup` took first, which is a slower and much safer operation.

Three properties the runner guarantees:

- **A backup happens before anything runs**, unless explicitly waived.
- **Each migration is one transaction**, so a failure leaves the version where it
  was rather than halfway.
- **Applied migrations are recorded with their checksum**, so editing a migration
  that has already run is detected instead of silently diverging two deployments.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..core.errors import FathomError

__all__ = [
    "AppliedMigration",
    "Migration",
    "MigrationError",
    "MigrationPlan",
    "applied_migrations",
    "backup",
    "checksum_of",
    "compact",
    "current_version",
    "export_store",
    "import_store",
    "migrate",
    "pending",
    "plan_migration",
    "restore",
    "verify_applied",
]

_LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migration (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied TEXT NOT NULL
)
"""


class MigrationError(FathomError):
    """A migration could not be applied, or an applied one no longer matches."""


@dataclass(frozen=True)
class Migration:
    """One forward step.

    `statements` are executed in order inside a single transaction. `apply` is for
    the rare migration that needs to read data to decide what to write; if it is
    set, `statements` run first.
    """

    version: int
    name: str
    statements: tuple[str, ...] = ()
    apply: Callable[[sqlite3.Connection], None] | None = None
    description: str = ""

    def checksum(self) -> str:
        """Commits to the statements, so an edited migration is detectable.

        A callable's body is not hashed — Python gives no stable way to do that — so
        a migration using `apply` should carry a version bump rather than an edit.
        """
        payload = json.dumps(
            {"version": self.version, "name": self.name, "statements": list(self.statements)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def checksum_of(migration: Migration) -> str:
    return migration.checksum()


@dataclass(frozen=True)
class AppliedMigration:
    """A migration this store has already run."""

    version: int
    name: str
    checksum: str
    applied: datetime


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(_LEDGER)
    return connection


def current_version(path: str | Path) -> int:
    """The highest migration this store has applied. Zero for a fresh one."""
    connection = _connect(path)
    try:
        row = connection.execute("SELECT MAX(version) AS v FROM schema_migration").fetchone()
        return int(row["v"] or 0)
    finally:
        connection.close()


def applied_migrations(path: str | Path) -> list[AppliedMigration]:
    connection = _connect(path)
    try:
        return [
            AppliedMigration(
                version=int(r["version"]),
                name=str(r["name"]),
                checksum=str(r["checksum"]),
                applied=datetime.fromisoformat(str(r["applied"])),
            )
            for r in connection.execute("SELECT * FROM schema_migration ORDER BY version")
        ]
    finally:
        connection.close()


def pending(path: str | Path, migrations: Sequence[Migration]) -> list[Migration]:
    """Migrations not yet applied, in order."""
    done = {m.version for m in applied_migrations(path)}
    return sorted((m for m in migrations if m.version not in done), key=lambda m: m.version)


def verify_applied(path: str | Path, migrations: Sequence[Migration]) -> list[str]:
    """Detect migrations whose definition changed after they ran.

    Two deployments that ran "the same" migration at different revisions have
    silently diverged, and every later migration assumes a schema one of them does
    not have.
    """
    by_version = {m.version: m for m in migrations}
    problems: list[str] = []
    for record in applied_migrations(path):
        migration = by_version.get(record.version)
        if migration is None:
            problems.append(
                f"version {record.version} ({record.name}) was applied but is no longer "
                "defined; this store ran a migration this build does not know about"
            )
            continue
        if migration.checksum() != record.checksum:
            problems.append(
                f"version {record.version} ({record.name}) was edited after it ran; "
                f"applied {record.checksum}, defined {migration.checksum()}"
            )
    return problems


@dataclass(frozen=True)
class MigrationPlan:
    """What `migrate` would do."""

    path: str
    current: int
    target: int
    steps: tuple[Migration, ...]
    problems: tuple[str, ...] = ()

    @property
    def is_safe(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if self.problems:
            lines = [f"{self.path}: BLOCKED at version {self.current}"]
            lines.extend(f"  {p}" for p in self.problems)
            return "\n".join(lines)
        if not self.steps:
            return f"{self.path}: already at version {self.current}"
        lines = [f"{self.path}: {self.current} -> {self.target}"]
        lines.extend(f"  {m.version} {m.name}" for m in self.steps)
        return "\n".join(lines)


def plan_migration(path: str | Path, migrations: Sequence[Migration]) -> MigrationPlan:
    """What would run, and whether it is safe to."""
    steps = pending(path, migrations)
    current = current_version(path)
    return MigrationPlan(
        path=str(path),
        current=current,
        target=steps[-1].version if steps else current,
        steps=tuple(steps),
        problems=tuple(verify_applied(path, migrations)),
    )


def backup(path: str | Path, destination: str | Path | None = None) -> Path:
    """Copy a store aside before touching it.

    Uses SQLite's backup API rather than a file copy, so a store being written to
    concurrently produces a consistent snapshot rather than a torn one.
    """
    source = Path(path)
    if not source.exists():
        raise MigrationError(f"{source} does not exist; nothing to back up")

    target = (
        Path(destination)
        if destination
        else source.with_suffix(source.suffix + f".{datetime.now(UTC):%Y%m%dT%H%M%S}.bak")
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    live = sqlite3.connect(str(source))
    saved = sqlite3.connect(str(target))
    try:
        live.backup(saved)
    finally:
        saved.close()
        live.close()
    return target


def restore(backup_path: str | Path, path: str | Path) -> Path:
    """Put a backup back. The rollback path, since migrations are forward-only."""
    source, target = Path(backup_path), Path(path)
    if not source.exists():
        raise MigrationError(f"{source} does not exist; nothing to restore")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def migrate(
    path: str | Path,
    migrations: Sequence[Migration],
    *,
    take_backup: bool = True,
    dry_run: bool = False,
) -> MigrationPlan:
    """Apply pending migrations.

    Each step is one transaction, so a failure leaves the version where it was
    rather than halfway. A backup is taken first unless explicitly waived — the one
    time somebody waives it is the one time they need it.
    """
    plan = plan_migration(path, migrations)
    if not plan.is_safe:
        raise MigrationError(plan.summary())
    if dry_run or not plan.steps:
        return plan

    if take_backup:
        backup(path)

    connection = _connect(path)
    try:
        for migration in plan.steps:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in migration.statements:
                    connection.execute(statement)
                if migration.apply is not None:
                    migration.apply(connection)
                connection.execute(
                    "INSERT INTO schema_migration (version, name, checksum, applied) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        migration.version,
                        migration.name,
                        migration.checksum(),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            except Exception as exc:
                connection.execute("ROLLBACK")
                raise MigrationError(
                    f"migration {migration.version} ({migration.name}) failed and was "
                    f"rolled back; the store is still at version "
                    f"{current_version(path)}: {exc}"
                ) from exc
            connection.execute("COMMIT")
    finally:
        connection.close()
    return plan


def export_store(path: str | Path, destination: str | Path) -> Path:
    """Dump every table to newline-delimited JSON.

    Portable in a way a SQLite file is not: it survives a schema change, another
    engine, and being read by something that is not this library.
    """
    connection = _connect(path)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        tables = [
            str(r["name"])
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        with target.open("w", encoding="utf-8") as handle:
            for table in tables:
                for row in connection.execute(f"SELECT * FROM {table}"):  # noqa: S608
                    handle.write(json.dumps({"table": table, "row": dict(row)}, default=str) + "\n")
        return target
    finally:
        connection.close()


def import_store(source: str | Path, path: str | Path, *, replace: bool = False) -> int:
    """Load an export back. Returns the number of rows written.

    Tables must already exist — importing does not create schema, because guessing
    column types from JSON is how a store ends up with everything as TEXT.
    """
    connection = _connect(path)
    written = 0
    try:
        known = {
            str(r["name"])
            for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if replace:
            for table in known - {"schema_migration"}:
                connection.execute(f"DELETE FROM {table}")  # noqa: S608

        connection.execute("BEGIN IMMEDIATE")
        try:
            with Path(source).open(encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    blob = json.loads(stripped)
                    table, row = blob["table"], blob["row"]
                    if table not in known:
                        raise MigrationError(
                            f"export contains table {table!r}, which does not exist in "
                            "the target store; migrate it to the right schema first"
                        )
                    columns = ", ".join(row)
                    placeholders = ", ".join("?" * len(row))
                    connection.execute(
                        f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})",  # noqa: S608
                        tuple(row.values()),
                    )
                    written += 1
        except Exception:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        return written
    finally:
        connection.close()


def compact(path: str | Path) -> dict[str, int]:
    """Reclaim space, reporting how much.

    Profile history grows without bound; a store that has been running a year is
    mostly superseded rows.
    """
    before = Path(path).stat().st_size
    connection = sqlite3.connect(str(path), isolation_level=None)
    try:
        connection.execute("VACUUM")
        connection.execute("ANALYZE")
    finally:
        connection.close()
    after = Path(path).stat().st_size
    return {"before": before, "after": after, "reclaimed": max(0, before - after)}
