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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .codec import (
    dataset_from_json,
    dataset_to_json,
    key_from_json,
    key_to_json,
    mapping_from_json,
    mapping_to_json,
    spec_from_json,
    spec_to_json,
)
from .graph import Edge, Graph
from .profile import ColumnProfile, Profile
from .types import UNPARTITIONED, DatasetId, KeyPredicate

__all__ = ["Store", "ShadowObservation"]

SCHEMA_VERSION = 1

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
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ShadowObservation:
    """One comparison of a plan against ground truth.

    `missed` is the number the whole approach lives or dies on: partitions the
    planner called clean that a full rebuild proved dirty. It must stay at zero.
    """

    dataset: DatasetId
    observed: datetime
    planned: int
    actual: int
    missed: int
    total: int

    @property
    def savings(self) -> float:
        return 0.0 if self.total == 0 else 1.0 - (self.planned / self.total)


class Store:
    """SQLite-backed persistence. Safe to open concurrently; writes are serialized."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (str(SCHEMA_VERSION),),
        )
        self._check_version()

    def _check_version(self) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        found = int(row["value"]) if row else SCHEMA_VERSION
        if found > SCHEMA_VERSION:
            raise RuntimeError(
                f"store at {self.path} was written by a newer fathom "
                f"(schema {found} > {SCHEMA_VERSION}); upgrade rather than downgrade"
            )

    def close(self) -> None:
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

    def save_graph(self, graph: Graph) -> None:
        """Merge a graph into the store. Re-running an ingest is a no-op."""
        with self._tx() as conn:
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
        graph = Graph()
        by_id: dict[int, DatasetId] = {}
        for row in self._conn.execute("SELECT id, ref, spec FROM dataset"):
            ds = dataset_from_json(row["ref"])
            by_id[int(row["id"])] = ds
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
        row = self._conn.execute(
            "SELECT t.value FROM token t JOIN dataset d ON d.id = t.dataset "
            "WHERE d.ref = ? AND t.adapter = ?",
            (dataset_to_json(dataset), adapter),
        ).fetchone()
        return str(row["value"]) if row else None

    def set_token(self, dataset: DatasetId, adapter: str, value: str) -> None:
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
                        None if c.min is None else str(c.min),
                        None if c.max is None else str(c.max),
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
                min=c["min_value"],
                max=c["max_value"],
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
        row = self._conn.execute(
            "SELECT p.* FROM profile p JOIN dataset d ON d.id = p.dataset "
            "WHERE d.ref = ? AND p.partition = ? ORDER BY p.captured DESC, p.id DESC LIMIT 1",
            (dataset_to_json(dataset), key_to_json(partition or KeyPredicate())),
        ).fetchone()
        return self._hydrate(row) if row else None

    def profile_history(
        self, dataset: DatasetId, partition: KeyPredicate | None = None, *, limit: int = 20
    ) -> list[Profile]:
        rows = self._conn.execute(
            "SELECT p.* FROM profile p JOIN dataset d ON d.id = p.dataset "
            "WHERE d.ref = ? AND p.partition = ? ORDER BY p.captured DESC, p.id DESC LIMIT ?",
            (dataset_to_json(dataset), key_to_json(partition or KeyPredicate()), limit),
        ).fetchall()
        return [self._hydrate(r) for r in rows]

    def profiled_partitions(self, dataset: DatasetId) -> list[KeyPredicate]:
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

    def datasets(self) -> Iterable[DatasetId]:
        for row in self._conn.execute("SELECT ref FROM dataset ORDER BY ref"):
            yield dataset_from_json(row["ref"])
