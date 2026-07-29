"""A vocabulary is a schema, and changing one is a schema change for every text asset.

Swapping a tokenizer changes every token count, every context budget, every cached
embedding, and the meaning of every `max_length` in the codebase. Nothing warns you,
because the code still runs and the numbers still look like numbers.

The three things this makes checkable:

**A vocabulary diff is a lineage event.** `diff` reports what was added, removed, and
re-indexed. Re-indexing is the dangerous one: a token that keeps its string and
changes its id silently invalidates every cached embedding and every id-space
artefact, while an added token merely makes old checkpoints incomplete.

**Token counts move even when the text does not.** `reestimate` prices a corpus
against a new vocabulary, and `budget_check` fails when a re-tokenized document no
longer fits a fixed context window — which is the failure that would otherwise show up
as silent truncation at the far end of a RAG pipeline.

**Fertility is the number that predicts the bill.** Tokens per word, per language.
A vocabulary that is fine for English and fertile for Turkish costs more and performs
worse on Turkish, and one aggregate number hides it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ...core.types import DatasetId
from ..assets import corpus as corpus_asset
from ..assets import tokenizer as tokenizer_asset

__all__ = [
    "BudgetVerdict",
    "CorpusEstimate",
    "Fertility",
    "Tokenizer",
    "VocabularyDiff",
    "budget_check",
    "compatible",
    "diff",
    "fertility",
    "fertility_by_language",
    "is_superset",
    "reestimate",
    "special_token_conflicts",
    "tokenizer_edges",
    "vocabulary_overlap",
    "worst_language",
]


@dataclass(frozen=True)
class Tokenizer:
    """A vocabulary with an identity.

    `vocabulary` maps token string to id. Both directions matter: a token that keeps
    its string and changes its id is the case that breaks caches, and you cannot see
    it from a set of strings alone.
    """

    name: str
    vocabulary: Mapping[str, int] = field(default_factory=dict)
    special_tokens: tuple[str, ...] = ()
    registry: str = "local"
    algorithm: str = ""  # bpe, unigram, wordpiece
    version: str = ""

    @property
    def dataset(self) -> DatasetId:
        return tokenizer_asset(self.name, registry=self.registry)

    @property
    def size(self) -> int:
        return len(self.vocabulary)

    def id_of(self, token: str) -> int | None:
        return self.vocabulary.get(token)


@dataclass(frozen=True)
class VocabularyDiff:
    """What changed between two vocabularies.

    `reindexed` is separated from `added` and `removed` because the three have
    different blast radii, and collapsing them into "the vocab changed" loses the
    only distinction that tells you what to rebuild.
    """

    before: str
    after: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    reindexed: tuple[tuple[str, int, int], ...]  # token, old id, new id
    special_changed: tuple[str, ...] = ()

    @property
    def unchanged(self) -> bool:
        return not (self.added or self.removed or self.reindexed or self.special_changed)

    @property
    def invalidates_embeddings(self) -> bool:
        """Any id movement invalidates every cached embedding and id-space artefact.

        Additions alone do not: old ids still mean what they meant, and the new
        tokens simply have no cached vector yet.
        """
        return bool(self.reindexed or self.removed)

    def summary(self) -> str:
        if self.unchanged:
            return f"{self.before} -> {self.after}: identical"
        lines = [
            f"{self.before} -> {self.after}: "
            f"+{len(self.added)} -{len(self.removed)} ~{len(self.reindexed)} reindexed"
        ]
        if self.special_changed:
            lines.append(
                f"  special tokens changed: {', '.join(self.special_changed)} — every "
                "prompt template built on them is now wrong"
            )
        if self.invalidates_embeddings:
            lines.append(
                "  ids moved, so every cached embedding and id-space artefact "
                "downstream is stale and must be recomputed"
            )
        elif self.added:
            lines.append(
                "  ids are stable; existing embeddings remain valid and the new "
                f"{len(self.added)} token(s) have no vector yet"
            )
        return "\n".join(lines)


def diff(before: Tokenizer, after: Tokenizer) -> VocabularyDiff:
    """Compare two vocabularies, separating additions from id movement."""
    old, new = before.vocabulary, after.vocabulary
    added = tuple(sorted(set(new) - set(old)))
    removed = tuple(sorted(set(old) - set(new)))
    reindexed = tuple(
        (token, old[token], new[token])
        for token in sorted(set(old) & set(new))
        if old[token] != new[token]
    )
    special = tuple(sorted(set(before.special_tokens) ^ set(after.special_tokens)))
    return VocabularyDiff(before.name, after.name, added, removed, reindexed, special)


def vocabulary_overlap(a: Tokenizer, b: Tokenizer) -> float:
    """Jaccard overlap of the two token sets. 1.0 when the strings agree exactly.

    Deliberately ignores ids: this answers "do these cover the same text", which is a
    different question from "can I reuse my cache".
    """
    left, right = set(a.vocabulary), set(b.vocabulary)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def is_superset(candidate: Tokenizer, base: Tokenizer) -> bool:
    """True when `candidate` adds tokens to `base` without moving any existing id.

    The one kind of vocabulary change that is safe to roll forward: old ids still
    decode to what they decoded to before.
    """
    return all(
        candidate.vocabulary.get(token) == identifier
        for token, identifier in base.vocabulary.items()
    )


def compatible(a: Tokenizer, b: Tokenizer) -> tuple[bool, str]:
    """Whether artefacts built with `a` can be read by `b`."""
    change = diff(a, b)
    if change.unchanged:
        return True, ""
    if change.invalidates_embeddings:
        return False, (
            f"{len(change.reindexed)} token(s) changed id and {len(change.removed)} were "
            "removed; every cached embedding, vector index, and stored token id built "
            "with the old vocabulary decodes to different text now"
        )
    if change.special_changed:
        return False, (
            f"special tokens changed ({', '.join(change.special_changed)}); prompt "
            "templates and chat formats built on them produce different sequences"
        )
    return True, ""


def special_token_conflicts(tokenizer: Tokenizer) -> list[str]:
    """Special tokens that are absent from the vocabulary they claim to belong to.

    A declared special token with no id is a silent fallback to byte-level splitting,
    which turns one control token into five ordinary ones and quietly breaks framing.
    """
    return [token for token in tokenizer.special_tokens if token not in tokenizer.vocabulary]


def tokenizer_edges(
    tokenizer: Tokenizer, consumers: Iterable[DatasetId]
) -> list[tuple[DatasetId, DatasetId]]:
    """The vocabulary feeds everything tokenized with it.

    These edges are what make a vocabulary bump an invalidation rather than a
    surprise: the planner already knows how to rebuild what a changed input feeds.
    """
    source = tokenizer.dataset
    return [(source, target) for target in consumers]


# -- counting ------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusEstimate:
    """What a corpus costs under a given vocabulary."""

    corpus: str
    tokenizer: str
    documents: int
    tokens: int
    words: int = 0

    @property
    def tokens_per_document(self) -> float:
        return self.tokens / self.documents if self.documents else 0.0

    @property
    def dataset(self) -> DatasetId:
        return corpus_asset(self.corpus)


def reestimate(before: CorpusEstimate, ratio: float, *, tokenizer: str = "") -> CorpusEstimate:
    """Re-price a corpus under a new vocabulary.

    `ratio` is new tokens per old token, which is what a fertility comparison on a
    sample gives you. Re-tokenizing the whole corpus to find out it grew 12% is the
    expensive way to learn something a sample answers.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return CorpusEstimate(
        corpus=before.corpus,
        tokenizer=tokenizer or before.tokenizer,
        documents=before.documents,
        tokens=round(before.tokens * ratio),
        words=before.words,
    )


@dataclass(frozen=True)
class BudgetVerdict:
    """Whether a document still fits after a vocabulary change."""

    fits: bool
    tokens: int
    limit: int
    overflow: int = 0

    @property
    def headroom(self) -> int:
        return max(0, self.limit - self.tokens)

    def summary(self) -> str:
        if self.fits:
            return f"{self.tokens}/{self.limit} tokens, {self.headroom} to spare"
        return (
            f"{self.tokens}/{self.limit} tokens — {self.overflow} over. A fixed context "
            "window does not grow with the vocabulary, so this truncates silently at "
            "the far end of the pipeline rather than failing here."
        )


def budget_check(tokens: int, limit: int, *, reserve: int = 0) -> BudgetVerdict:
    """Whether a token count fits a context window, minus whatever is reserved.

    `reserve` is for the completion, the system prompt, and the retrieved context —
    the parts people forget until a long input silently evicts them.
    """
    if limit <= 0:
        raise ValueError(f"context limit must be positive, got {limit}")
    usable = limit - reserve
    return BudgetVerdict(
        fits=tokens <= usable,
        tokens=tokens,
        limit=usable,
        overflow=max(0, tokens - usable),
    )


# -- fertility -----------------------------------------------------------------


@dataclass(frozen=True)
class Fertility:
    """Tokens per word for one language.

    The number that predicts both the bill and the quality gap. A vocabulary at 1.3
    for English and 3.1 for Turkish costs Turkish users more than twice as much per
    sentence and gives the model less room to work in.
    """

    language: str
    tokens: int
    words: int

    @property
    def ratio(self) -> float:
        return self.tokens / self.words if self.words else 0.0


def fertility(tokens: int, words: int, *, language: str = "und") -> Fertility:
    return Fertility(language=language, tokens=tokens, words=words)


def fertility_by_language(samples: Iterable[Fertility]) -> dict[str, float]:
    """Merge per-language samples into one ratio each, worst first.

    Aggregated across languages rather than pooled, because pooling weights by corpus
    composition and a corpus that is 90% English reports an English number.
    """
    totals: dict[str, list[int]] = {}
    for sample in samples:
        entry = totals.setdefault(sample.language, [0, 0])
        entry[0] += sample.tokens
        entry[1] += sample.words
    ratios = {lang: (t / w if w else 0.0) for lang, (t, w) in totals.items()}
    return dict(sorted(ratios.items(), key=lambda kv: -kv[1]))


def worst_language(samples: Sequence[Fertility]) -> tuple[str, float] | None:
    """The language this vocabulary serves worst, and by how much.

    Returns `None` on an empty sample rather than inventing a language, because "we
    measured nothing" and "every language is equally served" are different claims.
    """
    ratios = fertility_by_language(samples)
    if not ratios:
        return None
    language = next(iter(ratios))
    return language, ratios[language]
