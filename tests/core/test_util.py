"""The shared helpers.

Small enough to look untestable, and exactly the kind of thing that silently
diverges when six modules each keep their own copy. These tests pin the properties
the callers depend on.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.util import clock, digest, markdown, text

# -- digest --------------------------------------------------------------------


def test_json_digest_ignores_key_order():
    """Python preserves insertion order; a content address must not."""
    assert digest.of_json({"a": 1, "b": 2}) == digest.of_json({"b": 2, "a": 1})


def test_json_digest_tracks_content():
    assert digest.of_json({"a": 1}) != digest.of_json({"a": 2})


def test_text_digest_normalizes_whitespace_by_default():
    assert digest.of_text("one  two\n three") == digest.of_text("one two three")
    assert digest.of_text("a b", normalize=False) != digest.of_text("a  b", normalize=False)


def test_canonical_json_falls_back_to_str_for_unserializable_values():
    stamp = datetime(2026, 3, 14, tzinfo=UTC)
    assert "2026-03-14" in digest.canonical_json({"when": stamp})


def test_short_keeps_the_declared_length():
    full = digest.of_text("x")
    assert len(digest.short(full)) == digest.SHORT
    assert digest.short(full, 8) == full[:8]


# -- clock ---------------------------------------------------------------------


def test_naive_timestamps_are_utc_not_local():
    """Assuming local time would make retention wrong by hours, only in production."""
    assert clock.as_utc(datetime(2026, 3, 14)).tzinfo is UTC
    aware = datetime(2026, 3, 14, tzinfo=UTC)
    assert clock.as_utc(aware) is aware


def test_age_against_a_supplied_reference():
    stamp = datetime(2026, 3, 14, tzinfo=UTC)
    reference = datetime(2026, 3, 20, tzinfo=UTC)
    assert clock.age(stamp, reference=reference) == timedelta(days=6)


def test_absent_timestamps_are_unknown_not_zero():
    assert clock.age(None) is None
    # Unknown must not read as fresh: an unbuilt dataset is past every budget.
    assert clock.is_older_than(None, timedelta(days=365))


def test_is_older_than_compares_against_the_budget():
    stamp = datetime(2026, 3, 14, tzinfo=UTC)
    reference = datetime(2026, 3, 20, tzinfo=UTC)
    assert clock.is_older_than(stamp, timedelta(days=1), reference=reference)
    assert not clock.is_older_than(stamp, timedelta(days=30), reference=reference)


# -- markdown ------------------------------------------------------------------


def test_table_renders_headers_and_rows():
    out = markdown.table(["A", "B"], [[1, 2], [3, 4]])
    assert out.splitlines()[0] == "| A | B |"
    assert out.splitlines()[1] == "|---|---|"
    assert "| 1 | 2 |" in out


def test_absent_values_are_an_em_dash_not_a_blank():
    assert markdown.table(["A"], [[None]]).endswith("| — |")
    assert markdown.cell("") == markdown.ABSENT
    assert markdown.code(None) == markdown.ABSENT


def test_pipes_in_content_do_not_break_the_table():
    out = markdown.table(["A"], [["a|b"]])
    assert "a\\|b" in out
    assert out.count("|") == out.splitlines()[-1].count("|") + 4


def test_truncation_is_stated_rather_than_silent():
    out = markdown.table(["A"], [[i] for i in range(10)], limit=3)
    assert "_+7 more_" in out


def test_an_empty_table_says_so():
    assert markdown.table(["A"], []) == "_(none)_"
    assert markdown.bullets([]) == "_(none)_"


def test_note_quotes_every_line():
    assert markdown.note("one\ntwo") == "> one\n> two"


# -- text ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", 0), ("x", 1), ("x" * 400, 100)],
)
def test_token_estimate(value, expected):
    assert text.token_estimate(value) == expected


def test_normalize_collapses_runs():
    assert text.normalize("  a \n b  ") == "a b"


def test_truncate_counts_what_it_dropped():
    assert text.truncate(["a", "b", "c"], 2) == ["a", "b", "… and 1 more"]
    assert text.truncate(["a"], 5) == ["a"]


def test_join_truncated_sorts_and_counts():
    assert text.join_truncated(["c", "a", "b", "d"], 2) == "a, b, +2 more"
    assert text.join_truncated(["a"], 3) == "a"
