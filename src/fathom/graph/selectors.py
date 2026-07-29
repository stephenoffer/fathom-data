"""Selecting parts of the graph with a string.

Every real use of lineage starts by naming a slice of it: *this model and everything
downstream*, *everything feeding these two dashboards*, *the gold layer only*. Typing
that as Python is fine once and unbearable in a CI config, so it gets a syntax.

The syntax is dbt's, because tens of thousands of people already know it and a
second spelling for the same idea is a tax:

    ``model``            just that dataset
    ``+model``           the dataset and everything upstream
    ``model+``           the dataset and everything downstream
    ``+model+``          both directions
    ``2+model``          upstream, at most two hops
    ``model+3``          downstream, at most three hops
    ``@model``           the dataset, its descendants, and everything those need
    ``ns:duckdb``        every dataset in a namespace
    ``name:gold.*``      glob against the dataset name
    ``tag:pii``          datasets carrying a label (needs a `labels` argument)
    ``*gold*``           bare glob against the full identity

Space separates unions, a comma separates intersections, so
``+gold.revenue ns:snowflake`` is a union and ``ns:snowflake,name:gold.*`` is the
Snowflake gold tables. Exclusion is a separate argument rather than a prefix,
because a `-` inside a dataset name is common and ambiguity here silently selects
the wrong thing.

`@model` deserves its explanation: it is what you select to *rebuild* a model. The
model's descendants must be rebuilt too, and rebuilding them needs their own inputs,
which may sit outside the descendant set entirely.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from ..core.types import ColumnRef, DatasetId
from .model import Graph
from .query import ancestors, descendants

# Dataset to the label names it carries. `govern.policy.tag_index` builds one from a
# `LabelSet`; taking the plain mapping keeps selection independent of the policy
# machinery, which a selector has no reason to know about.
Tags: TypeAlias = Mapping[DatasetId, Iterable[str]]

__all__ = [
    "Selection",
    "SelectorError",
    "Term",
    "difference",
    "expand",
    "explain",
    "select_columns",
    "selector_for",
    "intersect",
    "matches",
    "parse",
    "parse_term",
    "resolve",
    "select_datasets",
    "select_edges",
    "select_subgraph",
    "union",
    "validate",
]

_TERM = re.compile(
    r"^(?:(?P<up>\d*)\+)?"
    r"(?P<at>@)?"
    r"(?P<body>.*?)"
    r"(?:\+(?P<down>\d*))?$"
)

# How far `+` reaches when no number is given. Deeper than any warehouse worth
# selecting across, so an unqualified `+` reads as "all the way".
UNLIMITED = 64


class SelectorError(ValueError):
    """A selector could not be parsed, or named nothing.

    Naming nothing is an error rather than an empty result on purpose: a CI job
    guarding `+gold.revenue` should fail loudly when the model is renamed, not
    quietly start checking zero datasets.
    """


@dataclass(frozen=True)
class Term:
    """One atom of a selector, already parsed."""

    body: str
    upstream: int = 0
    downstream: int = 0
    build_scope: bool = False  # the `@` form
    kind: str = "glob"  # glob | namespace | name | tag

    def __str__(self) -> str:
        up = (
            ""
            if not self.upstream
            else ("+" if self.upstream >= UNLIMITED else f"{self.upstream}+")
        )
        down = (
            ""
            if not self.downstream
            else ("+" if self.downstream >= UNLIMITED else f"+{self.downstream}")
        )
        prefix = "@" if self.build_scope else ""
        label = self.body if self.kind == "glob" else f"{self.kind}:{self.body}"
        return f"{up}{prefix}{label}{down}"


def parse_term(text: str) -> Term:
    """Parse one selector atom. Raises `SelectorError` on anything unrecognizable."""
    raw = text.strip()
    if not raw:
        raise SelectorError("empty selector term")
    match = _TERM.match(raw)
    if match is None:  # pragma: no cover - the pattern matches any string
        raise SelectorError(f"cannot parse selector term {text!r}")

    up_raw, down_raw = match.group("up"), match.group("down")
    body = match.group("body").strip()
    if not body:
        raise SelectorError(f"selector term {text!r} names nothing")

    kind = "glob"
    for prefix in ("ns", "namespace", "name", "tag", "label"):
        if body.startswith(f"{prefix}:"):
            body = body[len(prefix) + 1 :]
            kind = {"namespace": "namespace", "ns": "namespace", "label": "tag"}.get(prefix, prefix)
            break

    def depth(value: str | None) -> int:
        if value is None:
            return 0
        return UNLIMITED if value == "" else int(value)

    return Term(
        body=body,
        upstream=depth(up_raw),
        downstream=depth(down_raw),
        build_scope=match.group("at") is not None,
        kind=kind,
    )


@dataclass(frozen=True)
class Selection:
    """A parsed selector: a union of intersections of terms."""

    groups: tuple[tuple[Term, ...], ...] = ()

    def __str__(self) -> str:
        return " ".join(",".join(str(t) for t in group) for group in self.groups)

    @property
    def is_empty(self) -> bool:
        """True when this selection names nothing."""
        return not self.groups


def parse(selector: str) -> Selection:
    """Parse a whole selector string into unions of intersections."""
    groups: list[tuple[Term, ...]] = []
    for chunk in selector.split():
        terms = tuple(parse_term(part) for part in chunk.split(",") if part.strip())
        if terms:
            groups.append(terms)
    if not groups:
        raise SelectorError(f"selector {selector!r} names nothing")
    return Selection(groups=tuple(groups))


def matches(term: Term, ds: DatasetId, *, labels: Tags | None = None) -> bool:
    """True when one dataset satisfies a term's matcher, ignoring graph traversal."""
    if term.kind == "namespace":
        return fnmatch.fnmatch(ds.namespace, term.body)
    if term.kind == "name":
        return fnmatch.fnmatch(ds.name, term.body)
    if term.kind == "tag":
        return bool(labels) and term.body in set(labels.get(ds, ()) if labels else ())
    pattern = term.body if any(c in term.body for c in "*?[") else f"*{term.body}*"
    return fnmatch.fnmatch(str(ds), pattern) or str(ds) == term.body or ds.name == term.body


def expand(graph: Graph, term: Term, seeds: Iterable[DatasetId]) -> set[DatasetId]:
    """Apply a term's graph traversal to the datasets its matcher selected."""
    out: set[DatasetId] = set(seeds)
    for seed in list(out):
        if term.upstream:
            out.update(ancestors(graph, seed, max_depth=term.upstream))
        if term.downstream:
            out.update(descendants(graph, seed, max_depth=term.downstream))
        if term.build_scope:
            # Everything downstream, plus whatever those need to be rebuilt. Without
            # the second step the selection describes a build that cannot run.
            downstream = descendants(graph, seed)
            out.update(downstream)
            for node in downstream:
                out.update(ancestors(graph, node))
    return out


def _resolve_term(graph: Graph, term: Term, *, labels: Tags | None = None) -> set[DatasetId]:
    seeds = {ds for ds in graph.datasets if matches(term, ds, labels=labels)}
    return expand(graph, term, seeds)


def resolve(
    graph: Graph,
    selector: str | Selection,
    *,
    exclude: str | Selection | None = None,
    labels: Tags | None = None,
    allow_empty: bool = False,
) -> list[DatasetId]:
    """Resolve a selector against a graph.

    Raises rather than returning an empty list unless `allow_empty` is set, so a
    renamed model breaks the job that selects it instead of silently narrowing it.
    """
    parsed = parse(selector) if isinstance(selector, str) else selector

    selected: set[DatasetId] = set()
    for group in parsed.groups:
        matched: set[DatasetId] | None = None
        for term in group:
            found = _resolve_term(graph, term, labels=labels)
            matched = found if matched is None else (matched & found)
        selected |= matched or set()

    if exclude is not None:
        excluded = parse(exclude) if isinstance(exclude, str) else exclude
        for group in excluded.groups:
            matched = None
            for term in group:
                found = _resolve_term(graph, term, labels=labels)
                matched = found if matched is None else (matched & found)
            selected -= matched or set()

    if not selected and not allow_empty:
        raise SelectorError(f"selector {parsed} matched no datasets in this graph")
    return sorted(selected, key=str)


def select_datasets(
    graph: Graph,
    selector: str,
    *,
    exclude: str | None = None,
    labels: Tags | None = None,
    allow_empty: bool = True,
) -> list[DatasetId]:
    """Resolve a selector, empty-tolerant by default. The convenience entry point."""
    return resolve(graph, selector, exclude=exclude, labels=labels, allow_empty=allow_empty)


def select_subgraph(graph: Graph, selector: str, *, exclude: str | None = None) -> Graph:
    """The induced subgraph over a selection, ready to plan or render."""
    from .query import subgraph

    return subgraph(graph, resolve(graph, selector, exclude=exclude, allow_empty=True))


def select_edges(graph: Graph, selector: str, *, exclude: str | None = None) -> list[str]:
    """Rendered edges with both ends inside a selection."""
    keep = set(resolve(graph, selector, exclude=exclude, allow_empty=True))
    return sorted(str(e) for e in graph.edges if e.src in keep and e.dst in keep)


def select_columns(graph: Graph, selector: str, *, labels: Tags | None = None) -> list[ColumnRef]:
    """Every known column of every dataset a selector picks."""
    from .query import columns_of

    out: list[ColumnRef] = []
    for ds in resolve(graph, selector, labels=labels, allow_empty=True):
        out.extend(ColumnRef(ds, name) for name in columns_of(graph, ds))
    return sorted(out, key=str)


def validate(selector: str) -> list[str]:
    """Parse without a graph, returning the problems found. Empty means valid."""
    try:
        parse(selector)
    except SelectorError as exc:
        return [str(exc)]
    return []


def union(*groups: Iterable[DatasetId]) -> list[DatasetId]:
    """Sorted union of several dataset collections."""
    out: set[DatasetId] = set()
    for group in groups:
        out |= set(group)
    return sorted(out, key=str)


def intersect(*groups: Iterable[DatasetId]) -> list[DatasetId]:
    """Sorted intersection. An empty argument list yields an empty result."""
    sets = [set(group) for group in groups]
    if not sets:
        return []
    out = sets[0]
    for other in sets[1:]:
        out &= other
    return sorted(out, key=str)


def difference(base: Iterable[DatasetId], remove: Iterable[DatasetId]) -> list[DatasetId]:
    """Sorted set difference."""
    return sorted(set(base) - set(remove), key=str)


def explain(graph: Graph, selector: str, *, labels: Tags | None = None) -> str:
    """Describe what a selector picks and why, term by term.

    Written for the moment a CI job selects three datasets when someone expected
    thirty, which is otherwise a bisect through selector syntax.
    """
    parsed = parse(selector)
    lines = [f"selector `{parsed}`"]
    total: set[DatasetId] = set()
    for group in parsed.groups:
        group_result: set[DatasetId] | None = None
        for term in group:
            seeds = {ds for ds in graph.datasets if matches(term, ds, labels=labels)}
            expanded = expand(graph, term, seeds)
            lines.append(
                f"  term `{term}`: {len(seeds)} direct match(es) → {len(expanded)} after traversal"
            )
            group_result = expanded if group_result is None else (group_result & expanded)
        resolved = group_result or set()
        lines.append(f"  group `{','.join(str(t) for t in group)}` → {len(resolved)} dataset(s)")
        total |= resolved
    lines.append(f"total: {len(total)} dataset(s)")
    return "\n".join(lines)


def selector_for(datasets: Iterable[DatasetId]) -> str:
    """The literal selector naming exactly these datasets.

    The inverse direction: a UI lets someone pick datasets, this writes the string
    that reproduces the pick in a config file.
    """
    return " ".join(sorted(str(ds) for ds in datasets))
