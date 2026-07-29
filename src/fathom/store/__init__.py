"""Persistence for the two durable artifacts.

    base     the protocol: what any backend must offer
    sqlite   the one implementation, and the right default

SQLite deliberately. The graph is thousands of edges, not billions, and a server to
operate before the tool does anything useful is how a tool goes unadopted. The
protocol exists so Postgres can arrive later without anything upstream noticing.
"""

from ..observe.shadow import ShadowObservation
from .base import Persistence
from .sqlite import Store

__all__ = ["Persistence", "ShadowObservation", "Store"]
