"""Store migrations, backup, and export.

Migrations are forward-only, so the tests centre on the two things that makes
load-bearing: a failure must leave the version where it was, and an edited
migration that has already run must be detected rather than silently diverging two
deployments.
"""

from __future__ import annotations

import sqlite3

import pytest

from fathom.store.migrations import (
    Migration,
    MigrationError,
    applied_migrations,
    backup,
    checksum_of,
    compact,
    current_version,
    export_store,
    import_store,
    migrate,
    pending,
    plan_migration,
    restore,
    verify_applied,
)

FIRST = Migration(1, "create", ("CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)",))
SECOND = Migration(2, "add column", ("ALTER TABLE t ADD COLUMN c TEXT",))
BROKEN = Migration(3, "broken", ("THIS IS NOT SQL",))


@pytest.fixture
def store(tmp_path):
    return tmp_path / "fathom.db"


# -- planning ------------------------------------------------------------------


def test_a_fresh_store_is_at_version_zero(store):
    assert current_version(store) == 0


def test_planning_lists_what_would_run(store):
    plan = plan_migration(store, [FIRST, SECOND])
    assert [m.version for m in plan.steps] == [1, 2]
    assert plan.target == 2
    assert plan.is_safe


def test_a_dry_run_changes_nothing(store):
    migrate(store, [FIRST, SECOND], dry_run=True)
    assert current_version(store) == 0


def test_pending_shrinks_as_migrations_apply(store):
    migrate(store, [FIRST], take_backup=False)
    assert [m.version for m in pending(store, [FIRST, SECOND])] == [2]


def test_migrating_an_up_to_date_store_is_a_no_op(store):
    migrate(store, [FIRST], take_backup=False)
    plan = migrate(store, [FIRST], take_backup=False)
    assert plan.steps == ()
    assert "already at version" in plan.summary()


# -- applying ------------------------------------------------------------------


def test_migrations_apply_in_order(store):
    migrate(store, [SECOND, FIRST], take_backup=False)
    assert current_version(store) == 2
    assert [m.version for m in applied_migrations(store)] == [1, 2]


def test_a_failure_leaves_the_version_where_it_was(store):
    """Each migration is one transaction, so a failure is not halfway."""
    migrate(store, [FIRST, SECOND], take_backup=False)
    with pytest.raises(MigrationError, match="rolled back"):
        migrate(store, [FIRST, SECOND, BROKEN], take_backup=False)
    assert current_version(store) == 2


def test_a_failed_migration_is_not_recorded_as_applied(store):
    with pytest.raises(MigrationError):
        migrate(store, [FIRST, BROKEN], take_backup=False)
    assert 3 not in {m.version for m in applied_migrations(store)}


def test_a_callable_migration_runs(store):
    def seed(connection: sqlite3.Connection) -> None:
        connection.execute("INSERT INTO t (a, b) VALUES (1, 'seeded')")

    migrate(store, [FIRST, Migration(2, "seed", (), apply=seed)], take_backup=False)
    connection = sqlite3.connect(str(store))
    assert connection.execute("SELECT b FROM t").fetchone()[0] == "seeded"
    connection.close()


# -- checksums -----------------------------------------------------------------


def test_an_edited_migration_that_already_ran_is_detected(store):
    """Two deployments that ran "the same" migration at different revisions have
    silently diverged, and every later migration assumes a schema one lacks."""
    migrate(store, [FIRST], take_backup=False)
    edited = Migration(1, "create", ("CREATE TABLE t (a INTEGER, different TEXT)",))

    problems = verify_applied(store, [edited])
    assert problems
    assert "edited after it ran" in problems[0]


def test_an_edited_migration_blocks_further_migration(store):
    migrate(store, [FIRST], take_backup=False)
    edited = Migration(1, "create", ("CREATE TABLE t (a INTEGER, different TEXT)",))
    with pytest.raises(MigrationError, match="edited"):
        migrate(store, [edited, SECOND], take_backup=False)


def test_a_migration_this_build_does_not_know_about_is_detected(store):
    """The store ran ahead of the code, which is a downgrade nobody meant to do."""
    migrate(store, [FIRST, SECOND], take_backup=False)
    problems = verify_applied(store, [FIRST])
    assert any("no longer defined" in p for p in problems)


def test_the_checksum_covers_the_statements(store):
    assert checksum_of(FIRST) != checksum_of(
        Migration(1, "create", ("CREATE TABLE t (a INTEGER, extra TEXT)",))
    )


def test_the_checksum_is_stable_across_calls():
    assert checksum_of(FIRST) == checksum_of(FIRST)


# -- backup and restore --------------------------------------------------------


def test_migrating_takes_a_backup_by_default(store, tmp_path):
    migrate(store, [FIRST], take_backup=False)  # create the file first
    before = {p.name for p in tmp_path.iterdir()}
    migrate(store, [FIRST, SECOND])
    after = {p.name for p in tmp_path.iterdir()}
    assert any(name.endswith(".bak") for name in after - before)


def test_restore_puts_a_backup_back(store):
    migrate(store, [FIRST], take_backup=False)
    saved = backup(store)

    migrate(store, [FIRST, SECOND], take_backup=False)
    assert current_version(store) == 2

    restore(saved, store)
    assert current_version(store) == 1


def test_backing_up_a_missing_store_says_so(tmp_path):
    with pytest.raises(MigrationError, match="does not exist"):
        backup(tmp_path / "nope.db")


def test_restoring_from_a_missing_backup_says_so(store):
    with pytest.raises(MigrationError, match="does not exist"):
        restore(store.parent / "nope.bak", store)


# -- export and import ---------------------------------------------------------


def test_export_and_import_round_trip(store, tmp_path):
    migrate(store, [FIRST], take_backup=False)
    connection = sqlite3.connect(str(store), isolation_level=None)
    connection.execute("INSERT INTO t (a, b) VALUES (1, 'x')")
    connection.execute("INSERT INTO t (a, b) VALUES (2, 'y')")
    connection.close()

    dump = export_store(store, tmp_path / "out.jsonl")
    assert dump.exists()

    target = tmp_path / "restored.db"
    migrate(target, [FIRST], take_backup=False)
    written = import_store(dump, target, replace=True)
    assert written >= 2

    connection = sqlite3.connect(str(target))
    assert connection.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
    connection.close()


def test_importing_into_a_store_without_the_table_says_so(store, tmp_path):
    migrate(store, [FIRST], take_backup=False)
    connection = sqlite3.connect(str(store), isolation_level=None)
    connection.execute("INSERT INTO t (a, b) VALUES (1, 'x')")
    connection.close()
    dump = export_store(store, tmp_path / "out.jsonl")

    empty = tmp_path / "empty.db"
    with pytest.raises(MigrationError, match="does not exist in the target"):
        import_store(dump, empty)


def test_compact_reports_what_it_reclaimed(store):
    migrate(store, [FIRST], take_backup=False)
    result = compact(store)
    assert result["reclaimed"] >= 0
    assert result["before"] > 0
