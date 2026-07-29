"""Contamination and near-duplicate detection by content.

The behaviour under test is mostly about refusing to overclaim: an estimate stays an
estimate until text is compared, an empty comparison is not a clean result, and
signatures that are not comparable raise rather than returning a plausible number.
"""

from __future__ import annotations

import pytest

from fathom.ai.quality import (
    ContaminationVerdict,
    Sketch,
    cluster_duplicates,
    containment,
    estimate_jaccard,
    exact_overlap,
    find_duplicates,
    jaccard,
    longest_common_substring,
    ngram_overlap,
    ngrams,
    normalize_text,
    report,
    shingles,
    sketch_of,
    suspicious_pairs,
    verify,
)

QUESTION = "What is the capital of France? The capital of France is Paris, on the Seine."
LEAKED = (
    "Trivia dump. what is the capital of france the capital of France is Paris on the "
    "Seine!! Plus some other text that came with it."
)
CLEAN = "The mitochondrion generates most of the chemical energy a cell needs to run."


# -- normalization -------------------------------------------------------------


def test_normalization_survives_reformatting():
    """An eval question recased and stripped of punctuation is still that question,
    and a comparison that misses it reports clean, which is worse than not looking."""
    assert normalize_text("What  is\tthis?!") == "what is this"


def test_normalization_can_preserve_case_and_punctuation():
    assert normalize_text("What is this?", fold_case=False, strip_punctuation=False) == (
        "What is this?"
    )


def test_ngrams_of_short_text_degrade_to_the_whole_text():
    assert ngrams("two words", 13) == ["two words"]


def test_ngrams_of_empty_text_are_empty():
    assert ngrams("", 5) == []


def test_shingles_are_a_set_of_ngrams():
    assert shingles("a b c d e f", size=3) == set(ngrams("a b c d e f", 3))


# -- similarity ----------------------------------------------------------------


def test_containment_is_directional_and_jaccard_is_not():
    """Jaccard punishes size differences, so a short eval fully inside an enormous
    corpus scores near zero on it while being completely contained."""
    small, large = {"a", "b"}, {"a", "b", *(f"x{i}" for i in range(100))}
    assert containment(small, large) == 1.0
    assert jaccard(small, large) < 0.05


def test_containment_of_nothing_is_zero():
    assert containment([], ["a"]) == 0.0


def test_jaccard_of_two_empty_sets_is_zero_not_one():
    assert jaccard([], []) == 0.0


def test_ngram_overlap_finds_a_reformatted_leak():
    assert ngram_overlap(QUESTION, LEAKED, n=5) > 0.9


def test_ngram_overlap_of_unrelated_text_is_zero():
    assert ngram_overlap(QUESTION, CLEAN, n=5) == 0.0


def test_exact_overlap_returns_the_evidence():
    shared = exact_overlap(QUESTION, LEAKED, n=5)
    assert shared
    assert all(len(s.split()) == 5 for s in shared)


def test_longest_common_substring_needs_a_minimum_run():
    """A short accidental match is not evidence of anything."""
    assert longest_common_substring("abc", "abd", minimum=40) == ""
    assert longest_common_substring(QUESTION, LEAKED, minimum=20)


# -- sketches ------------------------------------------------------------------


def test_identical_text_sketches_identically():
    assert estimate_jaccard(sketch_of("a", QUESTION), sketch_of("b", QUESTION)) == 1.0


def test_unrelated_text_sketches_apart():
    assert estimate_jaccard(sketch_of("a", QUESTION), sketch_of("b", CLEAN)) < 0.1


def test_signatures_of_different_lengths_refuse_to_compare():
    """Comparing them anyway produces a number that looks fine and means nothing."""
    left = sketch_of("a", QUESTION, permutations=64)
    right = sketch_of("b", QUESTION, permutations=128)
    with pytest.raises(ValueError, match="different lengths"):
        estimate_jaccard(left, right)


def test_signatures_of_different_shingle_sizes_refuse_to_compare():
    left = sketch_of("a", QUESTION, shingle_size=3)
    right = sketch_of("b", QUESTION, shingle_size=5)
    with pytest.raises(ValueError, match="different shingle sizes"):
        estimate_jaccard(left, right)


def test_sketches_round_trip_through_serialisation():
    """They are serialisable so they can be computed where the data lives."""
    original = sketch_of("a", QUESTION)
    assert Sketch.from_dict(original.to_dict()) == original


def test_a_truncated_sketch_is_rejected_on_load():
    """One read back with an empty signature would match everything at a similarity
    nobody asked for."""
    with pytest.raises(ValueError, match="non-empty signature"):
        Sketch.from_dict({"identifier": "a", "signature": []})


def test_empty_text_still_produces_a_signature():
    assert len(sketch_of("a", "").signature) == 128


# -- duplicate detection -------------------------------------------------------


def test_duplicates_are_found_and_clustered_transitively():
    text = "the quick brown fox jumps over the lazy dog " * 3
    sketches = [sketch_of(f"d{i}", text) for i in range(3)]
    sketches.append(sketch_of("other", "entirely unrelated content about oceans and tides"))

    matches = find_duplicates(sketches, threshold=0.8)
    clusters = cluster_duplicates(matches)

    assert len(clusters) == 1
    assert clusters[0].members == ("d0", "d1", "d2")
    assert "other" not in clusters[0].members


def test_a_cluster_names_what_deduplication_would_remove():
    text = "repeated content here for the purposes of this test " * 3
    matches = find_duplicates([sketch_of(f"d{i}", text) for i in range(3)], threshold=0.8)
    cluster = cluster_duplicates(matches)[0]
    assert len(cluster.redundant) == 2
    assert cluster.representative not in cluster.redundant


def test_banding_below_its_sensitivity_raises_rather_than_missing_silently():
    """Pairs under the cutover would be missed, not reported, so the caller is told."""
    sketches = [sketch_of(f"d{i}", f"text number {i} " * 5) for i in range(3)]
    with pytest.raises(ValueError, match="below this banding's sensitivity"):
        find_duplicates(sketches, threshold=0.1)


def test_banding_must_divide_the_signature():
    sketches = [sketch_of("a", QUESTION)]
    with pytest.raises(ValueError, match="does not divide"):
        find_duplicates(sketches, bands=7)


def test_finding_duplicates_among_nothing_is_empty():
    assert find_duplicates([]) == []


def test_suspicious_pairs_are_directional():
    """The question is whether the eval leaked into training, not whether the two
    sets resemble each other."""
    found = suspicious_pairs([sketch_of("q", QUESTION)], [sketch_of("d", LEAKED)], threshold=0.3)
    assert found
    assert found[0].left == "q"
    assert found[0].right == "d"


# -- verdicts ------------------------------------------------------------------


def test_without_text_the_verdict_stops_at_suspect():
    """An estimate is a reason to check, not a finding."""
    result = report("trivia", [sketch_of("q", QUESTION)], [sketch_of("d", LEAKED)], threshold=0.3)
    assert result.verdict is ContaminationVerdict.SUSPECT
    assert "estimate, not a finding" in result.summary()


def test_with_text_a_verified_match_becomes_contaminated():
    texts = {"q": QUESTION, "d": LEAKED}
    result = report(
        "trivia", [sketch_of("q", QUESTION)], [sketch_of("d", LEAKED)], threshold=0.3, texts=texts
    )
    assert result.verdict is ContaminationVerdict.CONTAMINATED
    assert result.verified_matches
    assert result.verified_matches[0].evidence


def test_unrelated_content_is_clean():
    result = report("trivia", [sketch_of("q", QUESTION)], [sketch_of("d", CLEAN)], threshold=0.3)
    assert result.verdict is ContaminationVerdict.CLEAN


def test_comparing_nothing_is_unknown_not_clean():
    """The distinction that stops an unchecked benchmark being reported as verified."""
    assert report("trivia", [], []).verdict is ContaminationVerdict.UNKNOWN
    assert "not a clean result" in report("trivia", [], []).summary()


def test_verification_downgrades_a_false_positive():
    """Two texts that sketch alike but share no literal run are not contamination."""
    matches = suspicious_pairs(
        [sketch_of("q", QUESTION)], [sketch_of("d", QUESTION)], threshold=0.3
    )
    verified = verify(matches, {"q": QUESTION, "d": "totally different words entirely"})
    assert not verified[0].verified


def test_verification_passes_through_matches_it_cannot_check():
    matches = suspicious_pairs([sketch_of("q", QUESTION)], [sketch_of("d", LEAKED)], threshold=0.3)
    assert verify(matches, {}) == matches
