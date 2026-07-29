"""Label changes between two runs of inference and propagation.

The one line of this diff that belongs in a compliance review is `new_pii`: columns
that did not carry a personal-data label last week and do now. Everything else is
churn from a confidence threshold moving, and burying the first in the second is how
a review stops being read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.types import ColumnRef
from .policy import Label, LabelSet

__all__ = ["LabelDiff", "diff_labels"]


@dataclass
class LabelDiff:
    """Label changes between two runs of inference and propagation."""

    added: dict[ColumnRef, set[Label]] = field(default_factory=dict)
    removed: dict[ColumnRef, set[Label]] = field(default_factory=dict)
    reconfidenced: list[tuple[ColumnRef, Label, Label]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when nothing changed."""
        return not (self.added or self.removed or self.reconfidenced)

    @property
    def new_pii(self) -> list[ColumnRef]:
        """Columns that newly carry a personal-data label.

        The one line of this diff that belongs in a compliance review.
        """
        return sorted(
            (ref for ref, labels in self.added.items() if any(x.name == "pii" for x in labels)),
            key=str,
        )

    def summary(self) -> str:
        """The diff as text."""
        if self.is_empty:
            return "labels: no change"
        head = (
            f"labels: +{sum(len(v) for v in self.added.values())} "
            f"-{sum(len(v) for v in self.removed.values())}, "
            f"{len(self.reconfidenced)} confidence change(s)"
        )
        if self.new_pii:
            head += f"  [{len(self.new_pii)} column(s) newly labelled pii]"
        return head


def diff_labels(before: LabelSet, after: LabelSet) -> LabelDiff:
    """Compare two label sets by name, treating a confidence move as its own event."""
    out = LabelDiff()
    for ref in sorted(set(before) | set(after), key=str):
        b = {label.name: label for label in before.get(ref, set())}
        a = {label.name: label for label in after.get(ref, set())}
        added = {a[name] for name in set(a) - set(b)}
        removed = {b[name] for name in set(b) - set(a)}
        if added:
            out.added[ref] = added
        if removed:
            out.removed[ref] = removed
        for name in sorted(set(a) & set(b)):
            if a[name].confidence != b[name].confidence or a[name].confirmed != b[name].confirmed:
                out.reconfidenced.append((ref, b[name], a[name]))
    return out
