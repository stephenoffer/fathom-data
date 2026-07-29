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

from collections.abc import Iterable, Sequence

__all__ = ["CHARS_PER_TOKEN", "join_truncated", "normalize", "token_estimate", "truncate"]

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
