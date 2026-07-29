"""Vocabularies as schemas.

The distinction the whole module turns on: an added token leaves every existing id
meaning what it meant, and a moved id silently invalidates every cached embedding.
Collapsing both into "the vocab changed" loses the only fact that says what to
rebuild.
"""

from __future__ import annotations

import pytest

from fathom.ai.assets import corpus
from fathom.ai.train.tokenizers import (
    CorpusEstimate,
    Tokenizer,
    budget_check,
    compatible,
    diff,
    fertility,
    fertility_by_language,
    is_superset,
    reestimate,
    special_token_conflicts,
    tokenizer_edges,
    vocabulary_overlap,
    worst_language,
)

BASE = Tokenizer("v1", {"a": 0, "b": 1, "c": 2}, special_tokens=("a",))
ADDED = Tokenizer("v2", {"a": 0, "b": 1, "c": 2, "d": 3}, special_tokens=("a",))
MOVED = Tokenizer("v3", {"a": 0, "b": 2, "c": 1}, special_tokens=("a",))
DROPPED = Tokenizer("v4", {"a": 0, "b": 1}, special_tokens=("a",))


# -- identity ------------------------------------------------------------------


def test_a_tokenizer_has_a_dataset_identity():
    assert "v1" in str(BASE.dataset)


def test_size_and_lookup():
    assert BASE.size == 3
    assert BASE.id_of("b") == 1
    assert BASE.id_of("missing") is None


def test_the_registry_is_part_of_the_identity():
    assert Tokenizer("v1", registry="hf").dataset != BASE.dataset


# -- diffing -------------------------------------------------------------------


def test_an_identical_vocabulary_diffs_to_nothing():
    change = diff(BASE, BASE)
    assert change.unchanged
    assert "identical" in change.summary()


def test_an_addition_is_reported_without_touching_existing_ids():
    change = diff(BASE, ADDED)
    assert change.added == ("d",)
    assert change.reindexed == ()
    assert not change.invalidates_embeddings


def test_a_moved_id_invalidates_embeddings():
    """A token that keeps its string and changes its id is the case that breaks
    caches, and it is invisible from a set of strings."""
    change = diff(BASE, MOVED)
    assert change.added == ()
    assert {t for t, _, _ in change.reindexed} == {"b", "c"}
    assert change.invalidates_embeddings


def test_a_removal_invalidates_embeddings():
    assert diff(BASE, DROPPED).invalidates_embeddings


def test_the_summary_distinguishes_the_two_cases():
    assert "remain valid" in diff(BASE, ADDED).summary()
    assert "stale" in diff(BASE, MOVED).summary()


def test_changed_special_tokens_are_reported():
    other = Tokenizer("v5", BASE.vocabulary, special_tokens=("a", "b"))
    change = diff(BASE, other)
    assert change.special_changed == ("b",)
    assert "prompt template" in change.summary()


# -- compatibility -------------------------------------------------------------


def test_adding_tokens_stays_compatible():
    ok, why = compatible(BASE, ADDED)
    assert ok
    assert why == ""


def test_moving_ids_is_incompatible_and_says_why():
    ok, why = compatible(BASE, MOVED)
    assert not ok
    assert "decodes to different text" in why


def test_changing_special_tokens_is_incompatible():
    other = Tokenizer("v5", BASE.vocabulary, special_tokens=("z",))
    ok, why = compatible(BASE, other)
    assert not ok
    assert "special tokens changed" in why


def test_a_superset_keeps_every_existing_id():
    assert is_superset(ADDED, BASE)
    assert not is_superset(MOVED, BASE)


def test_a_dropped_token_is_not_a_superset():
    assert not is_superset(DROPPED, BASE)


def test_overlap_ignores_ids():
    """This answers "do these cover the same text", not "can I reuse my cache"."""
    assert vocabulary_overlap(BASE, MOVED) == 1.0


def test_overlap_of_disjoint_vocabularies_is_zero():
    assert vocabulary_overlap(BASE, Tokenizer("x", {"z": 0})) == 0.0


def test_two_empty_vocabularies_overlap_completely():
    assert vocabulary_overlap(Tokenizer("a"), Tokenizer("b")) == 1.0


def test_a_declared_special_token_missing_from_the_vocabulary_is_caught():
    """It silently falls back to byte-level splitting, turning one control token into
    five ordinary ones."""
    assert special_token_conflicts(Tokenizer("t", {"a": 0}, special_tokens=("a", "<eos>"))) == [
        "<eos>"
    ]


def test_a_consistent_tokenizer_has_no_conflicts():
    assert special_token_conflicts(BASE) == []


# -- lineage -------------------------------------------------------------------


def test_the_vocabulary_feeds_everything_tokenized_with_it():
    edges = tokenizer_edges(BASE, [corpus("web"), corpus("code")])
    assert len(edges) == 2
    assert all(source == BASE.dataset for source, _ in edges)


def test_no_consumers_means_no_edges():
    assert tokenizer_edges(BASE, []) == []


# -- counting ------------------------------------------------------------------


def test_reestimating_scales_the_token_count():
    """Re-tokenizing a whole corpus to learn it grew 12% is the expensive way to find
    out something a sample answers."""
    before = CorpusEstimate("web", "v1", documents=100, tokens=1000)
    after = reestimate(before, 1.12, tokenizer="v2")
    assert after.tokens == 1120
    assert after.tokenizer == "v2"
    assert after.documents == 100


def test_a_non_positive_ratio_raises():
    with pytest.raises(ValueError, match="must be positive"):
        reestimate(CorpusEstimate("web", "v1", 1, 1), 0)


def test_tokens_per_document():
    assert CorpusEstimate("web", "v1", documents=4, tokens=100).tokens_per_document == 25.0


def test_tokens_per_document_of_an_empty_corpus_is_zero():
    assert CorpusEstimate("web", "v1", documents=0, tokens=0).tokens_per_document == 0.0


def test_an_estimate_carries_a_dataset_identity():
    assert "web" in str(CorpusEstimate("web", "v1", 1, 1).dataset)


# -- budgets -------------------------------------------------------------------


def test_a_document_that_fits_reports_headroom():
    verdict = budget_check(1000, 4096)
    assert verdict.fits
    assert verdict.headroom == 3096


def test_an_overflow_is_reported_with_the_amount():
    """A fixed context window does not grow with the vocabulary, so this truncates
    silently at the far end of the pipeline."""
    verdict = budget_check(5000, 4096)
    assert not verdict.fits
    assert verdict.overflow == 904
    assert "truncates silently" in verdict.summary()


def test_the_reserve_comes_off_the_limit():
    """The completion and system prompt are the parts people forget until a long
    input silently evicts them."""
    assert not budget_check(4000, 4096, reserve=500).fits


def test_a_non_positive_limit_raises():
    with pytest.raises(ValueError, match="must be positive"):
        budget_check(10, 0)


def test_exactly_filling_the_window_fits():
    assert budget_check(4096, 4096).fits


# -- fertility -----------------------------------------------------------------


def test_fertility_is_tokens_per_word():
    assert fertility(130, 100, language="en").ratio == 1.3


def test_fertility_with_no_words_is_zero_rather_than_a_crash():
    assert fertility(10, 0).ratio == 0.0


def test_languages_are_ranked_worst_first():
    """A vocabulary at 1.3 for English and 3.1 for Turkish costs Turkish users twice
    as much per sentence."""
    ratios = fertility_by_language(
        [fertility(130, 100, language="en"), fertility(310, 100, language="tr")]
    )
    assert list(ratios) == ["tr", "en"]


def test_samples_of_one_language_are_merged():
    ratios = fertility_by_language(
        [fertility(10, 10, language="en"), fertility(30, 10, language="en")]
    )
    assert ratios["en"] == 2.0


def test_the_worst_language_is_named_with_its_ratio():
    worst = worst_language([fertility(130, 100, language="en"), fertility(310, 100, language="tr")])
    assert worst == ("tr", 3.1)


def test_measuring_nothing_returns_none_rather_than_inventing_a_language():
    """ "We measured nothing" and "every language is equally served" are different
    claims."""
    assert worst_language([]) is None
