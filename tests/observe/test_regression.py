"""Did the rewrite change the numbers, and did we compare enough to say?

Two properties carry this module. A changed partition is reported as *changed* and
never as *wrong*, because a refactor that fixes a bug is supposed to change output.
And a clean result is refused below a coverage floor, because comparing three
partitions out of four hundred proves nothing about the other three hundred and
ninety-seven.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from fathom.core.types import DatasetId, KeyPredicate
from fathom.observe import regression
from fathom.observe.profile import ColumnProfile, Profile, Severity

GOLD = DatasetId("duckdb", "gold.monthly")


def key(day: int) -> KeyPredicate:
    return KeyPredicate.of(dt=datetime(2026, 3, day))


def prints(**assignments: str) -> dict[KeyPredicate, str]:
    return {key(int(d.lstrip("d"))): digest for d, digest in assignments.items()}


def profile(rows: int, *, nulls: int = 0, low: float = 0.0, high: float = 10.0) -> Profile:
    return Profile(
        dataset=GOLD,
        row_count=rows,
        columns=(
            ColumnProfile("amount", "double", row_count=rows, null_count=nulls, min=low, max=high),
        ),
    )


# -- fingerprints --------------------------------------------------------------


def test_a_differing_digest_is_a_change():
    before, after = prints(d1="a", d2="b"), prints(d1="a", d2="c")
    assert regression.compare_fingerprints(before, after) == [key(2)]


def test_identical_digests_are_no_change():
    assert regression.compare_fingerprints(prints(d1="a"), prints(d1="a")) == []


def test_a_partition_on_one_side_only_is_not_a_content_change():
    """An unbuilt slice must not read as a table that shrank."""
    before, after = prints(d1="a", d2="b"), prints(d1="a")
    assert regression.compare_fingerprints(before, after) == []


def test_only_in_separates_gone_from_new():
    gone, new = regression.only_in(prints(d1="a", d2="b"), prints(d1="a", d3="c"))
    assert gone == [key(2)]
    assert new == [key(3)]


# -- explaining a difference ---------------------------------------------------


def test_a_row_count_move_is_an_error():
    (finding,) = regression.explain(profile(100), profile(120))
    assert finding.kind == "regression_row_count"
    assert finding.severity is Severity.ERROR
    assert finding.before == 100 and finding.after == 120


def test_a_null_rate_move_is_reported():
    findings = regression.explain(profile(100, nulls=0), profile(100, nulls=20))
    assert any(f.kind == "regression_null_rate" for f in findings)


def test_a_range_move_is_only_a_warning():
    findings = regression.explain(profile(100, high=10), profile(100, high=12))
    (finding,) = [f for f in findings if f.kind == "regression_max"]
    assert finding.severity is Severity.WARN


def test_a_breaking_schema_change_is_reported():
    before = Profile(dataset=GOLD, row_count=10, columns=(ColumnProfile("a", "string"),))
    after = Profile(dataset=GOLD, row_count=10, columns=())
    assert any(f.kind == "regression_schema" for f in regression.explain(before, after))


def test_tolerance_absorbs_a_summation_order_change():
    """A refactor that reorders a float sum and nothing else should stay quiet."""
    findings = regression.explain(
        profile(100, high=10.0), profile(100, high=10.001), tolerance=0.01
    )
    assert findings == []


def test_a_move_from_zero_is_reported_despite_tolerance():
    """Zero to anything has no meaningful ratio, so tolerance cannot absorb it."""
    findings = regression.explain(profile(0), profile(50), tolerance=0.99)
    assert any(f.kind == "regression_row_count" for f in findings)


def test_a_boolean_column_is_not_treated_as_a_numeric_range():
    before = Profile(
        dataset=GOLD, row_count=10, columns=(ColumnProfile("flag", "bool", min=False, max=False),)
    )
    after = Profile(
        dataset=GOLD, row_count=10, columns=(ColumnProfile("flag", "bool", min=False, max=True),)
    )
    assert [f for f in regression.explain(before, after) if "regression_m" in f.kind] == []


def test_identical_profiles_explain_nothing():
    assert regression.explain(profile(100), profile(100)) == []


# -- the report ----------------------------------------------------------------


def test_a_clean_comparison_over_enough_partitions_is_clean():
    same = prints(d1="a", d2="b", d3="c", d4="d")
    result = regression.compare_outputs(GOLD, same, same)
    assert result.is_clean
    assert result.is_conclusive
    assert "no change detected" in result.summary()


def test_a_changed_partition_is_reported():
    before, after = prints(d1="a", d2="b"), prints(d1="a", d2="different")
    result = regression.compare_outputs(GOLD, before, after)
    assert len(result.changed) == 1
    assert result.changed[0].partition == key(2)
    assert not result.is_clean


def test_a_change_is_never_called_wrong():
    """A rewrite that fixes a bug is supposed to change the output."""
    before, after = prints(d1="a", d2="b"), prints(d1="a", d2="c")
    summary = regression.compare_outputs(GOLD, before, after).summary()
    assert "changed" in summary
    assert "not this tool's call" in summary
    assert "wrong" not in summary.replace("not a wrong one", "")


def test_profiles_explain_how_it_changed():
    before, after = prints(d1="a"), prints(d1="b")
    result = regression.compare_outputs(
        GOLD,
        before,
        after,
        profiles_before={key(1): profile(100)},
        profiles_after={key(1): profile(150)},
        expected=1,
    )
    (changed,) = result.changed
    assert changed.is_explained
    assert "row count moved from 100 to 150" in str(changed)


def test_without_profiles_the_change_says_it_cannot_explain_itself():
    result = regression.compare_outputs(GOLD, prints(d1="a"), prints(d1="b"), expected=1)
    (changed,) = result.changed
    assert not changed.is_explained
    assert "no profiles supplied" in str(changed)


def test_unexplained_collects_the_least_actionable_findings():
    result = regression.compare_outputs(GOLD, prints(d1="a"), prints(d1="b"), expected=1)
    assert len(regression.unexplained(result)) == 1


# -- coverage, which is the whole game -----------------------------------------


def test_comparing_almost_nothing_is_not_a_pass():
    """Three partitions out of four hundred proves nothing about the rest."""
    same = prints(d1="a")
    result = regression.compare_outputs(GOLD, same, same, expected=400)
    assert not result.is_conclusive
    assert not result.is_clean
    assert "NOT CONCLUSIVE" in result.summary()


def test_coverage_is_the_compared_fraction():
    same = prints(d1="a", d2="b")
    assert regression.compare_outputs(GOLD, same, same, expected=4).coverage == pytest.approx(0.5)


def test_coverage_never_exceeds_one():
    same = prints(d1="a", d2="b", d3="c")
    assert regression.compare_outputs(GOLD, same, same, expected=1).coverage == 1.0


def test_expected_defaults_to_what_either_side_knew():
    before, after = prints(d1="a", d2="b"), prints(d1="a", d3="c")
    result = regression.compare_outputs(GOLD, before, after)
    assert result.expected == 3
    assert result.compared == 1


def test_comparing_nothing_is_inconclusive():
    result = regression.compare_outputs(GOLD, {}, {})
    assert not result.is_conclusive
    assert result.coverage == 0.0


# -- accepted changes ----------------------------------------------------------


def test_an_intended_change_is_counted_not_hidden():
    """A review that cannot see how many were waved through is not a review."""
    before, after = prints(d1="a", d2="b"), prints(d1="x", d2="b")
    result = regression.compare_outputs(GOLD, before, after, intended=[key(1)])
    assert result.changed == []
    assert result.accepted == [key(1)]
    assert "1 change(s) accepted as intended" in result.summary()


def test_an_unaccepted_change_still_reports():
    before, after = prints(d1="a", d2="b"), prints(d1="x", d2="y")
    result = regression.compare_outputs(GOLD, before, after, intended=[key(1)])
    assert [c.partition for c in result.changed] == [key(2)]


# -- the merge gate ------------------------------------------------------------


def test_is_regression_blocks_on_a_content_change():
    result = regression.compare_outputs(GOLD, prints(d1="a"), prints(d1="b"))
    assert regression.is_regression(result)


def test_is_regression_blocks_on_a_lost_partition():
    result = regression.compare_outputs(GOLD, prints(d1="a", d2="b"), prints(d1="a"))
    assert regression.is_regression(result)


def test_is_regression_is_not_the_negation_of_is_clean():
    """An inconclusive report is neither a regression nor a pass."""
    same = prints(d1="a")
    result = regression.compare_outputs(GOLD, same, same, expected=400)
    assert not result.is_clean
    assert not regression.is_regression(result)


def test_an_accepted_change_does_not_block():
    before, after = prints(d1="a", d2="b", d3="c", d4="d"), prints(d1="x", d2="b", d3="c", d4="d")
    result = regression.compare_outputs(GOLD, before, after, intended=[key(1)])
    assert not regression.is_regression(result)


# -- rollups -------------------------------------------------------------------


def test_changed_partitions_feed_back_into_a_plan():
    result = regression.compare_outputs(GOLD, prints(d1="a", d2="b"), prints(d1="x", d2="y"))
    assert set(regression.changed_partitions(result)) == {key(1), key(2)}


def test_worst_puts_the_most_explained_first():
    before, after = prints(d1="a", d2="b"), prints(d1="x", d2="y")
    result = regression.compare_outputs(
        GOLD,
        before,
        after,
        profiles_before={key(1): profile(100), key(2): profile(100)},
        profiles_after={key(1): profile(100, nulls=50, high=99), key(2): profile(101)},
        expected=2,
    )
    assert regression.worst(result)[0].partition == key(1)


def test_summarize_ranks_datasets_and_flags_the_inconclusive():
    same = prints(d1="a")
    thin = regression.compare_outputs(GOLD, same, same, expected=400)
    other = DatasetId("duckdb", "gold.other")
    changed = regression.compare_outputs(other, prints(d1="a"), prints(d1="b"), expected=1)

    text = regression.summarize([thin, changed])
    assert "1 changed" in text
    assert "INCONCLUSIVE" in text
    assert "compared too little to conclude anything" in text


def test_summarize_of_nothing_says_so():
    assert regression.summarize([]) == "nothing compared"
