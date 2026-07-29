"""Content addressing.

Six modules here hash something to answer "is this the same as last time" — a
training run's inputs, a prompt template, a chunk of text, an erasure proof. They
were each doing it slightly differently, which is the one thing content addressing
must not do: two digests of the same content computed by two call sites have to
agree, or the comparison they exist for silently stops working.

So the rules live in one place:

- **JSON is canonical.** Sorted keys, no whitespace. Dict ordering must not change a
  digest, and it will otherwise, because Python preserves insertion order.
- **Text is whitespace-normalized before hashing.** Reflowing a paragraph is not a
  content change; changing a word is. This matters most for prompts, where a
  formatter run would otherwise read as a new version.
- **Truncation is explicit.** `short()` says how many characters it kept, rather
  than each caller slicing a different number and the digests looking comparable
  when they are not.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["SHORT", "canonical_json", "of_json", "of_text", "short"]

# Characters kept by `short`. 16 hex characters is 64 bits — collision-free for any
# realistic number of datasets, and short enough to read in a table.
SHORT = 16


def canonical_json(payload: Any) -> str:
    """Serialize deterministically: sorted keys, no incidental whitespace.

    `default=str` so a datetime or a `DatasetId` in the payload becomes its string
    form rather than raising. Callers that need a specific rendering should convert
    before calling rather than relying on that fallback.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def of_text(text: str, *, normalize: bool = True) -> str:
    """Hash a string. Runs of whitespace collapse first unless told otherwise."""
    body = " ".join(text.split()) if normalize else text
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def of_json(payload: Any) -> str:
    """Hash a structure through its canonical JSON form."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def short(digest: str, length: int = SHORT) -> str:
    """The first `length` characters of a digest, for display and for keys."""
    return digest[:length]
