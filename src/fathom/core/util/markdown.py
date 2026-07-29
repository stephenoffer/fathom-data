"""Markdown assembly.

A dozen functions across `report`, `ai`, and `govern` build Markdown tables by
joining pipes by hand. Each one had its own opinion about the separator row, its own
truncation limit, and its own way of rendering an absent value — so two tables in the
same generated report did not look like they came from the same tool.

Nothing here is clever. It exists so the rendering rules are stated once:

- an absent value is an em dash, never an empty cell, so a short row is visibly
  short rather than looking like a rendering bug
- pipes inside cell content are escaped, because dataset names contain them often
  enough to break a table in production and never in a test
- truncation is a row that says how many were dropped, never a silent cut
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "ABSENT",
    "bullets",
    "cell",
    "code",
    "heading",
    "note",
    "table",
]

# What an unpopulated cell renders as. A blank cell reads as a bug in the generator.
ABSENT = "—"


def cell(value: Any) -> str:
    """Render one cell: absent values become an em dash, pipes are escaped."""
    if value is None or value == "":
        return ABSENT
    return str(value).replace("|", "\\|").replace("\n", " ")


def code(value: Any) -> str:
    """A cell in backticks, for identifiers. Absent values stay unquoted."""
    return ABSENT if value is None or value == "" else f"`{cell(value)}`"


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    limit: int | None = None,
    empty: str = "_(none)_",
) -> str:
    """A Markdown table. Over `limit` rows, the overflow is stated rather than cut."""
    body = [list(row) for row in rows]
    if not body:
        return empty

    shown = body if limit is None else body[:limit]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in shown)
    if limit is not None and len(body) > limit:
        dropped = len(body) - limit
        lines.append("| " + " | ".join([f"_+{dropped} more_"] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def bullets(items: Iterable[Any], *, empty: str = "_(none)_") -> str:
    """A bullet list, or `empty` when there is nothing to list."""
    rendered = [f"- {cell(item)}" for item in items]
    return "\n".join(rendered) if rendered else empty


def heading(text: str, level: int = 2) -> str:
    """A heading with a blank line after it, ready to join into a document."""
    return f"{'#' * max(1, level)} {text}\n"


def note(text: str) -> str:
    """A block quote. Used for the warnings that must not be skimmed past."""
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
