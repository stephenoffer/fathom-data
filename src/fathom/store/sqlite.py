"""Persistence for the two durable artifacts.

SQLite, deliberately. The graph is thousands of edges, not billions, and profile
history is one row per column per partition per run. A server-backed store would be
one more thing to operate before the tool does anything useful. Postgres can come
later behind the same interface if a team outgrows a file.

Everything here is idempotent. Ingest runs get interrupted, replayed, and run twice
concurrently; none of that should corrupt the graph.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..core.codec import (
    dataset_from_json,
    dataset_to_json,
    key_from_json,
    key_to_json,
    mapping_from_json,
    mapping_to_json,
    spec_from_json,
    spec_to_json,
    stat_from_json,
    stat_to_json,
)
from ..core.types import UNPARTITIONED, DatasetId, KeyPredicate
from ..core.util.clock import as_utc, now
from ..graph.history import Revision
from ..graph.model import Edge, Graph
from ..graph.plan.lifetime import RunRecord
from ..observe.completeness import Arrival
from ..observe.profile import ColumnProfile, Profile
from ..observe.seasonal import Observation
from ..observe.shadow import ShadowObservation
from ..observe.usage import ReadEvent, UsageStats, summarize

__all__ = ["INTERNAL_NAMESPACE", "Store"]

# Namespace reserved for the store's own bookkeeping rows (resume cursors).
# Never a real dataset, and filtered out of every graph we hand back.
INTERNAL_NAMESPACE = "fathom"

# How long a writer waits for another writer's transaction before giving up.
BUSY_TIMEOUT_MS = 30_000

SCHEMA_VERSION = 3

# Statements that move an existing store *to* each version, applied in order.
# Version 1 is the initial schema, which `_SCHEMA` creates outright. Version 2 adds
# only new tables, which `CREATE TABLE IF NOT EXISTS` already brings into an older
# store — so neither needs statements here. The table exists so that the first
# version to add or change a *column* has somewhere to say so, rather than leaving
# every existing store silently missing it.
_MIGRATIONS: dict[int, tuple[str, ...]] = {}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY,
    ref TEXT NOT NULL UNIQUE,
    spec TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edge (
    src INTEGER NOT NULL REFERENCES dataset(id),
    dst INTEGER NOT NULL REFERENCES dataset(id),
    evidence TEXT NOT NULL,
    mapping TEXT NOT NULL,
    columns TEXT NOT NULL,
    observed TEXT NOT NULL,
    PRIMARY KEY (src, dst, evidence)
);

CREATE TABLE IF NOT EXISTS token (
    dataset INTEGER NOT NULL REFERENCES dataset(id),
    adapter TEXT NOT NULL,
    value TEXT NOT NULL,
    updated TEXT NOT NULL,
    PRIMARY KEY (dataset, adapter)
);

CREATE TABLE IF NOT EXISTS profile (
    id INTEGER PRIMARY KEY,
    dataset INTEGER NOT NULL REFERENCES dataset(id),
    partition TEXT NOT NULL,
    captured TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS profile_lookup ON profile (dataset, partition, captured DESC);

CREATE TABLE IF NOT EXISTS profile_column (
    profile INTEGER NOT NULL REFERENCES profile(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    dtype TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    null_count INTEGER,
    min_value TEXT,
    max_value TEXT,
    distinct_estimate INTEGER,
    byte_size INTEGER,
    PRIMARY KEY (profile, name)
);

CREATE TABLE IF NOT EXISTS label (
    dataset INTEGER NOT NULL REFERENCES dataset(id),
    column_name TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    origin TEXT NOT NULL,
    confirmed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (dataset, column_name, label)
);

CREATE TABLE IF NOT EXISTS shadow (
    id INTEGER PRIMARY KEY,
    dataset INTEGER NOT NULL REFERENCES dataset(id),
    observed TEXT NOT NULL,
    planned INTEGER NOT NULL,
    actual INTEGER NOT NULL,
    missed INTEGER NOT NULL,
    total INTEGER NOT NULL
);

-- Three append-only event streams, added in schema 2. Each feeds a module that is
-- useless without history: completeness needs to know when a partition landed, usage
-- needs reads over a window, and lifetime cost needs the runs that actually happened.
-- They are separate tables rather than one polymorphic `event` table because their
-- columns genuinely differ and a shared table would be half nulls.

CREATE TABLE IF NOT EXISTS arrival (
    id INTEGER PRIMARY KEY,
    dataset INTEGER NOT NULL REFERENCES dataset(id),
    partition TEXT NOT NULL,
    observed TEXT NOT NULL,
    digest TEXT NOT NULL DEFAULT '',
    row_count INTEGER
);
CREATE INDEX IF NOT EXISTS arrival_lookup ON arrival (dataset, partition, observed);

CREATE TABLE IF NOT EXISTS read_event (
    id INTEGER PRIMARY KEY,
    dataset INTEGER NOT NULL REFERENCES dataset(id),
    principal TEXT NOT NULL,
    at TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'query',
    query_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS read_lookup ON read_event (dataset, at DESC);

CREATE TABLE IF NOT EXISTS run_record (
    id INTEGER PRIMARY KEY,
    dataset INTEGER NOT NULL REFERENCES dataset(id),
    at TEXT NOT NULL,
    partitions INTEGER NOT NULL DEFAULT 0,
    bytes_scanned INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    seconds REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS run_lookup ON run_record (dataset, at DESC);

-- The graph's own revision history, added in schema 3. A history that does not
-- survive the process is not a history, and every question it answers ("who narrowed
-- this edge, and when") is asked weeks after the change.
--
-- `revision` holds the authored metadata; `revision_change` holds one row per edge
-- the revision touched. Storing the changes rather than the whole `GraphDiff` is a
-- deliberate narrowing: it answers which, when, and by whom, and it does not
-- reconstruct the mappings. `graph_digest` is what lets a reader verify a graph they
-- still hold is the one a revision described.

CREATE TABLE IF NOT EXISTS revision (
    digest TEXT PRIMARY KEY,
    parent TEXT,
    at TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    datasets INTEGER NOT NULL DEFAULT 0,
    edges INTEGER NOT NULL DEFAULT 0,
    safe INTEGER NOT NULL DEFAULT 1,
    ordinal INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS revision_order ON revision (ordinal);

CREATE TABLE IF NOT EXISTS revision_change (
    revision TEXT NOT NULL REFERENCES revision(digest) ON DELETE CASCADE,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '',
    verb TEXT NOT NULL,
    PRIMARY KEY (revision, src, dst, evidence, verb)
);
CREATE INDEX IF NOT EXISTS revision_change_edge ON revision_change (src, dst);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _partition_moment(key: KeyPredicate, field_name: str) -> datetime | None:
    """The datetime a partition key sits at, or None if it carries none."""
    if field_name:
        value = key.get(field_name)
        return value if isinstance(value, datetime) else None
    for _, value in key.bindings:
        if isinstance(value, datetime):
            return value
    return None


def _changes_of(revision: Revision) -> list[tuple[str, str, str, str]]:
    """Flatten a revision's diff into (src, dst, evidence, verb) rows."""
    out: list[tuple[str, str, str, str]] = []
    diff = revision.diff
    for edge in diff.added_edges:
        out.append((str(edge.src), str(edge.dst), edge.evidence, "added"))
    for edge in diff.removed_edges:
        out.append((str(edge.src), str(edge.dst), edge.evidence, "removed"))
    for change in diff.changed_edges:
        verb = "narrowed" if change.narrowed else "widened" if change.widened else "changed"
        out.append((str(change.src), str(change.dst), change.evidence, verb))
    return out


class Store:
    """SQLite-backed persistence. Safe to open concurrently; writes are serialized."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # `save_graph` writes an entire ingest inside one transaction, which on a
        # large warehouse takes far longer than the 5s default. Without a longer
        # wait, a second concurrent ingest fails outright with "database is locked"
        # — and this class promises that running twice concurrently is safe.
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (str(SCHEMA_VERSION),),
        )
        self._check_version()
        self._migrate()

    def _migrate(self) -> None:
        """Bring an older store up to `SCHEMA_VERSION`, in order and in one transaction.

        `_SCHEMA` is all `CREATE TABLE IF NOT EXISTS`, which creates a store that does
        not exist and does exactly nothing to one that does. Without this, the first
        release that adds a column would leave every existing store silently missing
        it — the failure would surface as an `OperationalError` in the middle of
        somebody's ingest, against a database holding the only copy of their graph.

        Each entry lists the statements that move the schema *to* that version. The
        mechanism is here before it is needed on purpose: a migration path added after
        the first breaking change is a migration path that arrives one release late.
        """
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        found = int(row["value"]) if row else SCHEMA_VERSION
        if found >= SCHEMA_VERSION:
            return
        with self._tx() as conn:
            for target in range(found + 1, SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS.get(target, ()):
                    conn.execute(statement)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def _check_version(self) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        found = int(row["value"]) if row else SCHEMA_VERSION
        if found > SCHEMA_VERSION:
            raise RuntimeError(
                f"store at {self.path} was written by a newer fathom "
                f"(schema {found} > {SCHEMA_VERSION}); upgrade rather than downgrade"
            )

    def close(self) -> None:
        """Close the connection. The store is not usable afterwards."""
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # -- datasets --------------------------------------------------------------

    def _dataset_id(self, conn: sqlite3.Connection, ds: DatasetId, spec_json: str | None) -> int:
        ref = dataset_to_json(ds)
        row = conn.execute("SELECT id, spec FROM dataset WHERE ref = ?", (ref,)).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO dataset (ref, spec) VALUES (?, ?)",
                (ref, spec_json or spec_to_json(UNPARTITIONED)),
            )
            return int(cur.lastrowid or 0)
        # Only ever widen a known spec with a more specific one; never silently
        # replace a real spec with the unpartitioned default.
        if spec_json and spec_json != row["spec"] and spec_json != spec_to_json(UNPARTITIONED):
            conn.execute("UPDATE dataset SET spec = ? WHERE id = ?", (spec_json, row["id"]))
        return int(row["id"])

    # -- graph -----------------------------------------------------------------

    def save_graph(self, graph: Graph, *, replace_evidence: Iterable[str] = ()) -> None:
        """Merge a graph into the store. Re-running an ingest is a no-op.

        Merging alone never removes anything, so a dependency that goes away stays in
        the graph forever: a model edited to stop reading a table keeps being
        invalidated by it, and `erase` keeps naming a dataset the subject's data no
        longer reaches. Both are silent, and both get worse every release.

        `replace_evidence` names evidence prefixes the caller has just regenerated in
        full — `("sql:", "dbt:")` after re-reading every model file and the whole
        manifest. Edges carrying those prefixes are deleted and rewritten inside the
        same transaction, so what the store holds for that source is exactly what the
        source now says. Incremental sources, which report only what changed since a
        resume token, must not be listed: for those, merge is the correct behaviour.
        """
        prefixes = tuple(replace_evidence)
        with self._tx() as conn:
            for prefix in prefixes:
                conn.execute(
                    "DELETE FROM edge WHERE evidence = ? OR evidence LIKE ? ESCAPE '\\'",
                    (
                        prefix.rstrip(":"),
                        prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",
                    ),
                )
            for ds in graph.datasets:
                self._dataset_id(conn, ds, spec_to_json(graph.spec(ds)))
            for edge in graph.edges:
                src = self._dataset_id(conn, edge.src, None)
                dst = self._dataset_id(conn, edge.dst, None)
                conn.execute(
                    "INSERT INTO edge (src, dst, evidence, mapping, columns, observed) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(src, dst, evidence) DO UPDATE SET "
                    "mapping=excluded.mapping, columns=excluded.columns, "
                    "observed=excluded.observed",
                    (
                        src,
                        dst,
                        edge.evidence,
                        mapping_to_json(edge.mapping),
                        json.dumps([list(pair) for pair in edge.columns], separators=(",", ":")),
                        _now(),
                    ),
                )

    def load_graph(self) -> Graph:
        """Rebuild the whole graph in memory.

        Bookkeeping rows in the reserved internal namespace are filtered out, so a
        resume cursor never surfaces as an isolated dataset in somebody's lineage.
        """
        graph = Graph()
        by_id: dict[int, DatasetId] = {}
        for row in self._conn.execute("SELECT id, ref, spec FROM dataset"):
            ds = dataset_from_json(row["ref"])
            by_id[int(row["id"])] = ds
            if ds.namespace == INTERNAL_NAMESPACE:
                # Resume cursors are keyed by a dataset row for storage convenience.
                # They are not datasets, and left in they show up as isolated nodes in
                # every graph the user renders, counts, or plans an erasure over.
                continue
            graph.add_dataset(ds, spec_from_json(row["spec"]))
        for row in self._conn.execute(
            "SELECT src, dst, evidence, mapping, columns FROM edge ORDER BY src, dst, evidence"
        ):
            columns = tuple((str(a), str(b)) for a, b in json.loads(row["columns"]))
            graph.add_edge(
                Edge(
                    src=by_id[int(row["src"])],
                    dst=by_id[int(row["dst"])],
                    mapping=mapping_from_json(row["mapping"]),
                    columns=columns,
                    evidence=row["evidence"],
                )
            )
        return graph

    # -- resume tokens ---------------------------------------------------------

    def get_token(self, dataset: DatasetId, adapter: str) -> str | None:
        """The stored resume cursor for one dataset and adapter, or None."""
        row = self._conn.execute(
            "SELECT t.value FROM token t JOIN dataset d ON d.id = t.dataset "
            "WHERE d.ref = ? AND t.adapter = ?",
            (dataset_to_json(dataset), adapter),
        ).fetchone()
        return str(row["value"]) if row else None

    def set_token(self, dataset: DatasetId, adapter: str, value: str) -> None:
        """Record a resume cursor.

        Advancing it past a source's own reporting lag permanently skips rows that had
        not landed yet, which is why adapters state that lag rather than leaving it to
        be discovered in production.
        """
        with self._tx() as conn:
            ds_id = self._dataset_id(conn, dataset, None)
            conn.execute(
                "INSERT INTO token (dataset, adapter, value, updated) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(dataset, adapter) DO UPDATE SET "
                "value=excluded.value, updated=excluded.updated",
                (ds_id, adapter, value, _now()),
            )

    # -- profiles --------------------------------------------------------------

    def save_profile(self, profile: Profile, *, captured: datetime | None = None) -> int:
        """Persist one profile and its columns, returning the new row id."""
        stamp = (captured or datetime.now(UTC)).isoformat()
        with self._tx() as conn:
            ds_id = self._dataset_id(conn, profile.dataset, None)
            cur = conn.execute(
                "INSERT INTO profile (dataset, partition, captured, row_count, file_count, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ds_id,
                    key_to_json(profile.partition),
                    stamp,
                    profile.row_count,
                    profile.file_count,
                    profile.source,
                ),
            )
            profile_id = int(cur.lastrowid or 0)
            conn.executemany(
                "INSERT INTO profile_column (profile, name, dtype, row_count, null_count, "
                "min_value, max_value, distinct_estimate, byte_size) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        profile_id,
                        c.name,
                        c.dtype,
                        c.row_count,
                        c.null_count,
                        stat_to_json(c.min),
                        stat_to_json(c.max),
                        c.distinct_estimate,
                        c.byte_size,
                    )
                    for c in profile.columns
                ],
            )
            return profile_id

    def _hydrate(self, row: sqlite3.Row) -> Profile:
        columns = tuple(
            ColumnProfile(
                name=c["name"],
                dtype=c["dtype"],
                row_count=int(c["row_count"]),
                null_count=None if c["null_count"] is None else int(c["null_count"]),
                min=stat_from_json(c["min_value"]),
                max=stat_from_json(c["max_value"]),
                distinct_estimate=(
                    None if c["distinct_estimate"] is None else int(c["distinct_estimate"])
                ),
                byte_size=None if c["byte_size"] is None else int(c["byte_size"]),
            )
            for c in self._conn.execute(
                "SELECT * FROM profile_column WHERE profile = ? ORDER BY name", (row["id"],)
            )
        )
        ref = self._conn.execute(
            "SELECT ref FROM dataset WHERE id = ?", (row["dataset"],)
        ).fetchone()
        return Profile(
            dataset=dataset_from_json(ref["ref"]),
            partition=key_from_json(row["partition"]),
            row_count=int(row["row_count"]),
            columns=columns,
            file_count=int(row["file_count"]),
            source=row["source"],
        )

    def latest_profile(
        self, dataset: DatasetId, partition: KeyPredicate | None = None
    ) -> Profile | None:
        """The most recent profile for one dataset partition, or None."""
        row = self._conn.execute(
            "SELECT p.* FROM profile p JOIN dataset d ON d.id = p.dataset "
            "WHERE d.ref = ? AND p.partition = ? ORDER BY p.captured DESC, p.id DESC LIMIT 1",
            (dataset_to_json(dataset), key_to_json(partition or KeyPredicate())),
        ).fetchone()
        return self._hydrate(row) if row else None

    def profile_history(
        self, dataset: DatasetId, partition: KeyPredicate | None = None, *, limit: int = 20
    ) -> list[Profile]:
        """Recent profiles for one partition, newest first."""
        rows = self._conn.execute(
            "SELECT p.* FROM profile p JOIN dataset d ON d.id = p.dataset "
            "WHERE d.ref = ? AND p.partition = ? ORDER BY p.captured DESC, p.id DESC LIMIT ?",
            (dataset_to_json(dataset), key_to_json(partition or KeyPredicate()), limit),
        ).fetchall()
        return [self._hydrate(r) for r in rows]

    def seasonal_observations(
        self, dataset: DatasetId, *, field_name: str = "", limit: int = 500
    ) -> list[Observation]:
        """Profile history as observations a seasonal baseline can be learned from.

        Each observation is bucketed by the **partition's own moment**, not by when
        profiling ran, so a Monday partition backfilled on a Saturday lands in the
        Monday band. `field_name` names the partition field carrying that moment;
        empty means the first datetime binding found, which is right whenever there is
        only one time field and explicit whenever there is more.

        Profiles whose partition carries no datetime are skipped rather than dated by
        their capture time — a whole-dataset profile has no position in a weekly cycle,
        and giving it one would put every such profile in the same bucket.
        """
        rows = self._conn.execute(
            "SELECT p.* FROM profile p JOIN dataset d ON d.id = p.dataset "
            "WHERE d.ref = ? ORDER BY p.captured ASC, p.id ASC LIMIT ?",
            (dataset_to_json(dataset), limit),
        ).fetchall()

        out: list[Observation] = []
        for row in rows:
            profile = self._hydrate(row)
            when = _partition_moment(profile.partition, field_name)
            if when is not None:
                out.append(Observation(when, profile))
        return out

    def last_profiled(self, dataset: DatasetId) -> datetime | None:
        """When this dataset was last profiled.

        Used as a build-time proxy by freshness checks. It is a proxy and not the
        truth: a dataset profiled at noon may have been built at dawn. Freshness
        reports say so rather than implying the two are the same.
        """
        row = self._conn.execute(
            "SELECT MAX(p.captured) AS captured FROM profile p "
            "JOIN dataset d ON d.id = p.dataset WHERE d.ref = ?",
            (dataset_to_json(dataset),),
        ).fetchone()
        if row is None or row["captured"] is None:
            return None
        return datetime.fromisoformat(str(row["captured"]))

    def profiled_partitions(self, dataset: DatasetId) -> list[KeyPredicate]:
        """Every partition of a dataset that has ever been profiled."""
        rows = self._conn.execute(
            "SELECT DISTINCT p.partition FROM profile p JOIN dataset d ON d.id = p.dataset "
            "WHERE d.ref = ?",
            (dataset_to_json(dataset),),
        ).fetchall()
        return [key_from_json(r["partition"]) for r in rows]

    # -- labels ----------------------------------------------------------------

    def set_label(
        self,
        dataset: DatasetId,
        column: str,
        label: str,
        *,
        confidence: float,
        origin: str,
        confirmed: bool = False,
    ) -> None:
        """Record a label, keeping any human confirmation that already exists.

        Confirmation is sticky by design: re-running inference must never quietly undo
        a decision somebody made.
        """
        with self._tx() as conn:
            ds_id = self._dataset_id(conn, dataset, None)
            conn.execute(
                "INSERT INTO label (dataset, column_name, label, confidence, origin, confirmed) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(dataset, column_name, label) DO UPDATE SET "
                "confidence=excluded.confidence, origin=excluded.origin, "
                # A human confirmation is sticky: re-running inference must not undo it.
                "confirmed=MAX(label.confirmed, excluded.confirmed)",
                (ds_id, column, label, confidence, origin, int(confirmed)),
            )

    def labels_for(self, dataset: DatasetId) -> dict[str, list[tuple[str, float, bool]]]:
        """Labels stored for one dataset, grouped by column name."""
        out: dict[str, list[tuple[str, float, bool]]] = {}
        for row in self._conn.execute(
            "SELECT l.column_name, l.label, l.confidence, l.confirmed FROM label l "
            "JOIN dataset d ON d.id = l.dataset WHERE d.ref = ? ORDER BY l.column_name, l.label",
            (dataset_to_json(dataset),),
        ):
            out.setdefault(row["column_name"], []).append(
                (row["label"], float(row["confidence"]), bool(row["confirmed"]))
            )
        return out

    def all_labels(self) -> dict[DatasetId, dict[str, list[tuple[str, float, bool]]]]:
        """Every stored label, grouped by dataset and then column."""
        out: dict[DatasetId, dict[str, list[tuple[str, float, bool]]]] = {}
        for row in self._conn.execute(
            "SELECT d.ref, l.column_name, l.label, l.confidence, l.confirmed FROM label l "
            "JOIN dataset d ON d.id = l.dataset ORDER BY d.ref, l.column_name"
        ):
            ds = dataset_from_json(row["ref"])
            out.setdefault(ds, {}).setdefault(row["column_name"], []).append(
                (row["label"], float(row["confidence"]), bool(row["confirmed"]))
            )
        return out

    # -- shadow mode -----------------------------------------------------------

    def record_shadow(self, observation: ShadowObservation) -> None:
        """Persist one graded plan-versus-truth comparison."""
        with self._tx() as conn:
            ds_id = self._dataset_id(conn, observation.dataset, None)
            conn.execute(
                "INSERT INTO shadow (dataset, observed, planned, actual, missed, total) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ds_id,
                    observation.observed.isoformat(),
                    observation.planned,
                    observation.actual,
                    observation.missed,
                    observation.total,
                ),
            )

    def shadow_history(self, *, limit: int = 100) -> list[ShadowObservation]:
        """Recent shadow observations, newest first."""
        rows = self._conn.execute(
            "SELECT s.*, d.ref FROM shadow s JOIN dataset d ON d.id = s.dataset "
            "ORDER BY s.observed DESC, s.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            ShadowObservation(
                dataset=dataset_from_json(r["ref"]),
                observed=datetime.fromisoformat(r["observed"]),
                planned=int(r["planned"]),
                actual=int(r["actual"]),
                missed=int(r["missed"]),
                total=int(r["total"]),
            )
            for r in rows
        ]

    def shadow_summary(self) -> dict[str, Any]:
        """Accumulated totals, including the miss count the approach lives on."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS runs, SUM(planned) AS planned, SUM(actual) AS actual, "
            "SUM(missed) AS missed, SUM(total) AS total FROM shadow"
        ).fetchone()
        runs = int(row["runs"] or 0)
        planned = int(row["planned"] or 0)
        total = int(row["total"] or 0)
        return {
            "runs": runs,
            "planned": planned,
            "actual": int(row["actual"] or 0),
            "missed": int(row["missed"] or 0),
            "total": total,
            "savings": 0.0 if total == 0 else 1.0 - planned / total,
        }

    # -- convenience -----------------------------------------------------------

    # -- event streams ---------------------------------------------------------
    #
    # Three append-only logs, added in schema 2. Every one of them exists because a
    # module above is answering a question about *history* and had no way to obtain
    # it: which partitions have landed and when, who read a dataset, and what each
    # build actually cost. Nothing here interprets the events — that stays in
    # `observe.completeness`, `observe.usage`, and `graph.plan.lifetime`, so the
    # store keeps storing and retrieving rather than growing a second analysis layer.

    def record_arrival(self, arrival: Arrival) -> None:
        """Append one observation that a partition was written."""
        with self._tx() as conn:
            ds_id = self._dataset_id(conn, arrival.dataset, None)
            conn.execute(
                "INSERT INTO arrival (dataset, partition, observed, digest, row_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    ds_id,
                    key_to_json(arrival.key),
                    as_utc(arrival.observed).isoformat(),
                    arrival.digest,
                    arrival.row_count,
                ),
            )

    def record_arrivals(self, arrivals: Iterable[Arrival]) -> int:
        """Append many at once. Returns how many were written."""
        count = 0
        with self._tx() as conn:
            for arrival in arrivals:
                ds_id = self._dataset_id(conn, arrival.dataset, None)
                conn.execute(
                    "INSERT INTO arrival (dataset, partition, observed, digest, row_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        ds_id,
                        key_to_json(arrival.key),
                        as_utc(arrival.observed).isoformat(),
                        arrival.digest,
                        arrival.row_count,
                    ),
                )
                count += 1
        return count

    def arrivals(
        self,
        dataset: DatasetId | None = None,
        *,
        since: datetime | None = None,
        limit: int = 10_000,
    ) -> list[Arrival]:
        """Recorded arrivals, oldest first so duplicate groups read in order."""
        sql = "SELECT a.*, d.ref FROM arrival a JOIN dataset d ON d.id = a.dataset"
        clauses: list[str] = []
        params: list[Any] = []
        if dataset is not None:
            clauses.append("d.ref = ?")
            params.append(dataset_to_json(dataset))
        if since is not None:
            clauses.append("a.observed >= ?")
            params.append(as_utc(since).isoformat())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY a.observed ASC, a.id ASC LIMIT ?"
        params.append(limit)
        return [
            Arrival(
                dataset=dataset_from_json(r["ref"]),
                key=key_from_json(r["partition"]),
                observed=datetime.fromisoformat(r["observed"]),
                digest=r["digest"],
                row_count=r["row_count"],
            )
            for r in self._conn.execute(sql, params).fetchall()
        ]

    def present_partitions(self, dataset: DatasetId) -> list[KeyPredicate]:
        """Distinct partitions ever observed arriving.

        The `present` side of a completeness check, from arrivals rather than from a
        listing — so it still answers after a partition has been deleted, which a
        listing cannot.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT a.partition FROM arrival a JOIN dataset d ON d.id = a.dataset "
            "WHERE d.ref = ?",
            (dataset_to_json(dataset),),
        ).fetchall()
        return [key_from_json(r["partition"]) for r in rows]

    def record_read(self, event: ReadEvent) -> None:
        """Append one observed read."""
        with self._tx() as conn:
            ds_id = self._dataset_id(conn, event.dataset, None)
            conn.execute(
                "INSERT INTO read_event (dataset, principal, at, kind, query_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    ds_id,
                    event.principal,
                    as_utc(event.at).isoformat(),
                    event.kind,
                    event.query_id,
                ),
            )

    def record_reads(self, events: Iterable[ReadEvent]) -> int:
        """Append many reads. A query log replay is thousands of rows, not one."""
        count = 0
        with self._tx() as conn:
            for event in events:
                ds_id = self._dataset_id(conn, event.dataset, None)
                conn.execute(
                    "INSERT INTO read_event (dataset, principal, at, kind, query_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        ds_id,
                        event.principal,
                        as_utc(event.at).isoformat(),
                        event.kind,
                        event.query_id,
                    ),
                )
                count += 1
        return count

    def reads(
        self,
        dataset: DatasetId | None = None,
        *,
        since: datetime | None = None,
        limit: int = 100_000,
    ) -> list[ReadEvent]:
        """Recorded reads, newest first."""
        sql = "SELECT r.*, d.ref FROM read_event r JOIN dataset d ON d.id = r.dataset"
        clauses: list[str] = []
        params: list[Any] = []
        if dataset is not None:
            clauses.append("d.ref = ?")
            params.append(dataset_to_json(dataset))
        if since is not None:
            clauses.append("r.at >= ?")
            params.append(as_utc(since).isoformat())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY r.at DESC, r.id DESC LIMIT ?"
        params.append(limit)
        return [
            ReadEvent(
                dataset=dataset_from_json(r["ref"]),
                principal=r["principal"],
                at=datetime.fromisoformat(r["at"]),
                kind=r["kind"],
                query_id=r["query_id"],
            )
            for r in self._conn.execute(sql, params).fetchall()
        ]

    def usage(self, *, window: timedelta | None = None) -> dict[DatasetId, UsageStats]:
        """Aggregated reads over `window`, ready for `observe.usage`.

        The window is passed to `summarize` rather than inferred, and is also what
        bounds the query — so the returned stats carry the period they were computed
        over and a caller cannot report "unused" without it.
        """
        since = now() - window if window is not None else None
        return summarize(self.reads(since=since), window=window)

    def record_run(self, run: RunRecord) -> None:
        """Append one build that actually happened."""
        with self._tx() as conn:
            ds_id = self._dataset_id(conn, run.dataset, None)
            conn.execute(
                "INSERT INTO run_record (dataset, at, partitions, bytes_scanned, tokens, seconds) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ds_id,
                    as_utc(run.at).isoformat(),
                    run.partitions,
                    run.bytes_scanned,
                    run.tokens,
                    run.seconds,
                ),
            )

    def runs(
        self,
        dataset: DatasetId | None = None,
        *,
        since: datetime | None = None,
        limit: int = 100_000,
    ) -> list[RunRecord]:
        """Recorded runs, oldest first, ready for `lifetime.accumulate`."""
        sql = "SELECT r.*, d.ref FROM run_record r JOIN dataset d ON d.id = r.dataset"
        clauses: list[str] = []
        params: list[Any] = []
        if dataset is not None:
            clauses.append("d.ref = ?")
            params.append(dataset_to_json(dataset))
        if since is not None:
            clauses.append("r.at >= ?")
            params.append(as_utc(since).isoformat())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY r.at ASC, r.id ASC LIMIT ?"
        params.append(limit)
        return [
            RunRecord(
                dataset=dataset_from_json(r["ref"]),
                at=datetime.fromisoformat(r["at"]),
                partitions=int(r["partitions"]),
                bytes_scanned=int(r["bytes_scanned"]),
                tokens=int(r["tokens"]),
                seconds=float(r["seconds"]),
            )
            for r in self._conn.execute(sql, params).fetchall()
        ]

    # -- graph revisions -------------------------------------------------------

    def record_revision(self, revision: Revision) -> None:
        """Persist one revision and the edges it touched.

        Idempotent on digest: re-recording the same revision is a no-op, so a replayed
        ingest does not fork the chain.

        What is stored is narrower than a `Revision` in memory — the authored metadata
        plus one row per touched edge, not the `GraphDiff` itself. That answers which
        edge, when, and by whom, which is every question an incident asks of a history.
        Reconstructing the mappings would need the graphs, and those are not kept.
        """
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT 1 FROM revision WHERE digest = ?", (revision.digest,)
            ).fetchone()
            if existing:
                return
            ordinal = int(
                conn.execute("SELECT COALESCE(MAX(ordinal), 0) FROM revision").fetchone()[0]
            )
            conn.execute(
                "INSERT INTO revision "
                "(digest, parent, at, author, note, datasets, edges, safe, ordinal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    revision.digest,
                    revision.parent,
                    as_utc(revision.at).isoformat(),
                    revision.author,
                    revision.note,
                    revision.datasets,
                    revision.edges,
                    1 if revision.is_safe else 0,
                    ordinal + 1,
                ),
            )
            for src, dst, evidence, verb in _changes_of(revision):
                conn.execute(
                    "INSERT OR IGNORE INTO revision_change (revision, src, dst, evidence, verb) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (revision.digest, src, dst, evidence, verb),
                )

    def revisions(self, *, limit: int = 500) -> list[dict[str, Any]]:
        """Recorded revisions, oldest first.

        Plain mappings rather than `Revision` objects, because a stored revision has no
        `GraphDiff` and returning one with an empty diff would read as "nothing
        changed" instead of "the diff was not kept".
        """
        rows = self._conn.execute(
            "SELECT * FROM revision ORDER BY ordinal ASC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {
                "digest": r["digest"],
                "parent": r["parent"],
                "at": datetime.fromisoformat(r["at"]),
                "author": r["author"],
                "note": r["note"],
                "datasets": int(r["datasets"]),
                "edges": int(r["edges"]),
                "is_safe": bool(r["safe"]),
            }
            for r in rows
        ]

    def head_revision(self) -> dict[str, Any] | None:
        """The most recently recorded revision, or `None` for an empty history."""
        found = self.revisions()
        return found[-1] if found else None

    def edge_changes(
        self, src: DatasetId, dst: DatasetId, *, verb: str | None = None
    ) -> list[dict[str, Any]]:
        """Every recorded revision that touched one edge, oldest first.

        Pass ``verb="narrowed"`` for the incident question: six days of downstream data
        stopped being invalidated, so when did that window shrink and who shrank it.
        """
        sql = (
            "SELECT c.verb, r.* FROM revision_change c JOIN revision r ON r.digest = c.revision "
            "WHERE c.src = ? AND c.dst = ?"
        )
        params: list[Any] = [str(src), str(dst)]
        if verb is not None:
            sql += " AND c.verb = ?"
            params.append(verb)
        sql += " ORDER BY r.ordinal ASC"
        return [
            {
                "verb": r["verb"],
                "digest": r["digest"],
                "at": datetime.fromisoformat(r["at"]),
                "author": r["author"],
                "note": r["note"],
            }
            for r in self._conn.execute(sql, params).fetchall()
        ]

    def unsafe_revisions(self) -> list[dict[str, Any]]:
        """Revisions that narrowed a mapping or removed an edge, oldest first."""
        return [r for r in self.revisions() if not r["is_safe"] and r["parent"] is not None]

    def datasets(self) -> Iterable[DatasetId]:
        """Every dataset the store knows about, internal bookkeeping excluded."""
        for row in self._conn.execute("SELECT ref FROM dataset ORDER BY ref"):
            ds = dataset_from_json(row["ref"])
            if ds.namespace == INTERNAL_NAMESPACE:
                continue  # bookkeeping row, not a dataset
            yield ds
