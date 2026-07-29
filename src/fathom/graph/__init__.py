"""The dependency graph, and everything that reads it structurally.

    model       Edge, Graph, InvalidationPlan — the artifact and the planner
    query       traversal: ancestors, descendants, paths, cycles, subgraphs
    selectors   dbt-style selection strings resolved against a graph
    diff        what changed between two versions, and whether that is safe
    history     the chain of those changes over time — who narrowed this edge, and when
    sinks       the last hop: dashboards, reports, and filings a number reached
    metrics     coverage — how much of the graph is precise enough to plan on
    plan/       what a plan costs, and how it runs

`query` is where new traversals go. Adding one to `Graph` itself is almost always
wrong: the planner needs five methods and a traversal added to the class becomes a
method every future backend has to implement.
"""

from . import diff, history, metrics, plan, query, selectors, sinks
from .model import Edge, Graph, InvalidationPlan, link

__all__ = [
    "Edge",
    "Graph",
    "InvalidationPlan",
    "diff",
    "history",
    "link",
    "metrics",
    "plan",
    "query",
    "selectors",
    "sinks",
]
