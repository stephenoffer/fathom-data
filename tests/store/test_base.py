"""The persistence protocol, and that the one implementation actually satisfies it.

A protocol nothing asserts against is a docstring. These tests are what make
`Persistence` a contract: they fail when `Store` drops a method, when a signature
drifts, and — the case that actually happens — when someone adds a capability to
`Store` and forgets that a future Postgres backend will have to provide it too.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from fathom.core.ids import normalize_table
from fathom.core.types import KeyPredicate
from fathom.observe.profile import ColumnProfile, Profile
from fathom.store import Persistence, Store

RAW = normalize_table("raw.events", system="duckdb")


@pytest.fixture
def store():
    with Store() as opened:
        yield opened


def protocol_methods() -> list[str]:
    return sorted(
        name
        for name in dir(Persistence)
        if not name.startswith("_") and callable(getattr(Persistence, name))
    )


def test_the_protocol_is_not_empty():
    """A guard on the guard: an empty protocol would make every assertion below vacuous."""
    assert len(protocol_methods()) >= 10


def test_sqlite_satisfies_the_protocol_structurally(store):
    assert isinstance(store, Persistence)


@pytest.mark.parametrize("name", protocol_methods())
def test_sqlite_implements_every_protocol_method(name: str, store):
    assert hasattr(store, name), f"Store is missing {name}()"
    assert callable(getattr(store, name))


@pytest.mark.parametrize("name", protocol_methods())
def test_signatures_match_the_protocol(name: str):
    """A backend that takes different arguments is not a drop-in, whatever it claims.

    Extra parameters are allowed and must be optional: a backend may offer more than
    the protocol promises, but a caller holding only the protocol has to be able to
    call it. Missing parameters are always a break.
    """
    expected = inspect.signature(getattr(Persistence, name))
    actual = inspect.signature(getattr(Store, name))

    missing = [p for p in expected.parameters if p not in actual.parameters]
    assert not missing, f"{name}() is missing {missing}, which the protocol declares"

    required_extras = [
        name_
        for name_, param in actual.parameters.items()
        if name_ not in expected.parameters and param.default is inspect.Parameter.empty
    ]
    assert not required_extras, (
        f"{name}() requires {required_extras}, which the protocol does not declare; "
        "either give them defaults or add them to the protocol"
    )


def test_the_protocol_stores_and_retrieves_but_does_not_query():
    """Traversal runs in memory against a Graph, so no backend reimplements it in SQL."""
    forbidden = {"ancestors", "descendants", "invalidate", "reachable", "paths"}
    assert not forbidden & set(protocol_methods())


def test_writes_are_idempotent(store):
    """Ingest runs get interrupted, replayed, and run twice concurrently."""
    from fathom.core.partitions import PartitionMapping
    from fathom.graph import Edge, Graph

    graph = Graph()
    graph.add_edge(Edge(RAW, normalize_table("gold.x", system="duckdb"), PartitionMapping()))

    store.save_graph(graph)
    store.save_graph(graph)
    assert len(store.load_graph().edges) == 1


def test_a_confirmed_label_outlives_later_inference(store):
    """The one behaviour the protocol's docstring promises and a backend could get wrong."""
    store.set_label(RAW, "email", "pii", confidence=1.0, origin="human", confirmed=True)
    store.set_label(RAW, "email", "pii", confidence=0.4, origin="inferred", confirmed=False)

    labels = store.labels_for(RAW)["email"]
    assert [entry for entry in labels if entry[0] == "pii"][0][2] is True


def test_profile_round_trip_preserves_the_partition_key(store):
    partition = KeyPredicate.of(dt=datetime(2026, 3, 14, tzinfo=UTC))
    store.save_profile(
        Profile(
            dataset=RAW,
            partition=partition,
            row_count=42,
            columns=(ColumnProfile("amount", "double", row_count=42),),
        )
    )
    found = store.latest_profile(RAW, partition)
    assert found is not None
    assert found.partition == partition
    assert found.row_count == 42


def test_datasets_lists_what_was_written(store):
    store.set_token(RAW, "delta", "v1")
    assert RAW in list(store.datasets())
    assert store.get_token(RAW, "delta") == "v1"
    assert store.get_token(RAW, "iceberg") is None
