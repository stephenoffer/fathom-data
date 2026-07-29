"""What a persistence backend has to offer.

SQLite is the only implementation and the right default — the graph is thousands of
edges, not billions, and a server to operate before the tool does anything useful is
how a tool goes unadopted. This protocol exists so that stops being an assumption
baked into every call site.

The surface is deliberately narrow, and narrow in a specific direction: **it stores
and retrieves, it does not query.** Traversal lives in `graph.query` and runs against
an in-memory `Graph`, so a backend never has to reimplement reachability in SQL.
`load_graph` returning the whole graph is not a limitation to route around later; it
is the design, and it is what keeps the planner identical regardless of where the
edges were parked.

Everything is idempotent. Ingest runs get interrupted, replayed, and run twice
concurrently; none of that may corrupt the graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from ..core.types import DatasetId, KeyPredicate
from ..graph.model import Graph
from ..observe.profile import Profile

__all__ = ["Persistence"]


@runtime_checkable
class Persistence(Protocol):
    """The storage surface the rest of the library depends on."""

    # -- graph -----------------------------------------------------------------

    def save_graph(self, graph: Graph, *, replace_evidence: Iterable[str] = ()) -> None:
        """Merge a graph in. Re-running an ingest must be a no-op.

        `replace_evidence` names evidence prefixes the caller has just regenerated in
        full, whose stored edges must be deleted and rewritten in the same
        transaction. Without it a merge-only store keeps every dependency that ever
        existed, so a model edited to stop reading a table goes on being invalidated
        by it — silently, and worse every release.
        """
        ...

    def load_graph(self) -> Graph:
        """The whole graph, ready to traverse in memory."""
        ...

    # -- resume tokens ---------------------------------------------------------

    def get_token(self, dataset: DatasetId, adapter: str) -> str | None:
        """The stored resume cursor for one dataset and adapter, if any."""
        ...

    def set_token(self, dataset: DatasetId, adapter: str, value: str) -> None:
        """Record a resume cursor. Advancing it skips anything not yet landed."""
        ...

    # -- profiles --------------------------------------------------------------

    def save_profile(self, profile: Profile, *, captured: datetime | None = None) -> int:
        """Persist one profile and return its row id."""
        ...

    def latest_profile(
        self, dataset: DatasetId, partition: KeyPredicate | None = None
    ) -> Profile | None:
        """The most recent profile for a dataset partition, or None."""
        ...

    def profile_history(
        self, dataset: DatasetId, partition: KeyPredicate | None = None, *, limit: int = 20
    ) -> list[Profile]:
        """Recent profiles for one partition, newest first."""
        ...

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
        """Record a label. A human confirmation must outlive any later inference."""
        ...

    def labels_for(self, dataset: DatasetId) -> dict[str, list[tuple[str, float, bool]]]:
        """Stored labels for one dataset, keyed by column name."""
        ...

    # -- shadow mode -----------------------------------------------------------

    def shadow_summary(self) -> dict[str, Any]:
        """Accumulated savings and, more importantly, the accumulated miss count."""
        ...

    # -- inventory -------------------------------------------------------------

    def datasets(self) -> Iterable[DatasetId]:
        """Every dataset the store knows about."""
        ...

    def close(self) -> None:
        """Release the underlying connection."""
        ...
