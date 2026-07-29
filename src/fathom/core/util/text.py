"""Text measurement, for the places where text costs money.

Two modules estimate token counts — `ai.vectors` to price a re-embed, `ai.prompts`
to size a context window — and both were carrying their own characters-per-token
constant. Two constants that are meant to be the same number will eventually not be,
and the failure is a cost estimate that disagrees with itself between two screens of
the same report.

The estimate is deliberately crude. Being within twenty per cent decides "worth
doing" versus "not", and an exact tokenizer would pull a model dependency into a
library whose whole point is that it reads metadata rather than data.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence

__all__ = [
    "CHARS_PER_TOKEN",
    "did_you_mean",
    "join_truncated",
    "normalize",
    "options",
    "token_estimate",
    "truncate",
]

# Characters per token for English prose, across the common BPE vocabularies. Code
# and non-Latin scripts run lower; both directions are within the tolerance this is
# used at.
CHARS_PER_TOKEN = 4.0


def normalize(text: str) -> str:
    """Collapse runs of whitespace. What content addressing hashes."""
    return " ".join(text.split())


def token_estimate(text: str, *, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Approximate tokens in a string. Never returns zero for non-empty text."""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def truncate(items: Sequence[str], limit: int) -> list[str]:
    """The first `limit` items, plus a line saying how many were dropped."""
    if len(items) <= limit:
        return list(items)
    return [*items[:limit], f"… and {len(items) - limit} more"]


def join_truncated(items: Iterable[str], limit: int = 3, *, separator: str = ", ") -> str:
    """Join a few items and count the rest. For one-line summaries."""
    values = sorted(items)
    head = separator.join(values[:limit])
    return head + (f"{separator}+{len(values) - limit} more" if len(values) > limit else "")


def did_you_mean(word: str, candidates: Iterable[str], *, cutoff: float = 0.6) -> str:
    """A ``. Did you mean 'x'?`` fragment, or an empty string when nothing is close.

    Every message that rejects a name the user typed should offer the nearest one we
    do know. A typo and an unsupported feature produce the same error otherwise, and
    the user goes reading documentation to discover they wrote ``daly``.

    Returns a fragment ready to concatenate onto a message, so call sites read as
    ``f"unknown grain {s!r}{did_you_mean(s, names)}"``.

    Example:
        >>> did_you_mean("daly", ["hour", "day", "month"])
        ". Did you mean 'day'?"
        >>> did_you_mean("fortnight", ["hour", "day", "month"])
        ''
    """
    matches = difflib.get_close_matches(word.lower(), [c.lower() for c in candidates], 1, cutoff)
    return f". Did you mean {matches[0]!r}?" if matches else ""


def options(candidates: Iterable[str], *, limit: int = 12) -> str:
    """The valid values, sorted, for the tail of an error message.

    Listing what *is* accepted turns a rejection into an answer. Long lists are cut
    off rather than filling a terminal, because past a dozen names the user wants
    the documentation, not the enumeration.

    Example:
        >>> options(["day", "hour", "month"])
        "one of: 'day', 'hour', 'month'"
    """
    values = sorted(set(candidates))
    shown = ", ".join(repr(v) for v in values[:limit])
    if len(values) > limit:
        shown += f", … ({len(values) - limit} more)"
    return f"one of: {shown}"
