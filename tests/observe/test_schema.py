"""Structural differences between two profiles.

Separate from drift on purpose: drift asks whether the values moved, this asks
whether the shape moved. A retyped column breaks a downstream query outright; a
drifted one makes its answer quietly wrong. The tests keep the two apart.
"""

from __future__ import annotations

from fathom.core.ids import normalize_table
from fathom.observe import schema
from fathom.observe.profile import ColumnProfile, Profile

RAW = normalize_table("raw.events", system="duckdb")


def profile(*columns: ColumnProfile, rows: int = 5000) -> Profile:
    return Profile(dataset=RAW, row_count=rows, columns=columns)


AMOUNT = ColumnProfile("amount", "double", row_count=5000, null_count=0)
ID = ColumnProfile("id", "string", row_count=5000, null_count=0)


def test_identical_profiles_have_no_schema_diff():
    before = profile(ID, AMOUNT)
    assert schema.diff_schemas(before, before).is_empty
    assert not schema.is_breaking(before, before)
    assert "unchanged" in schema.diff_schemas(before, before).summary()


def test_an_added_column_is_not_breaking():
    """Consumers name the columns they read; a new one cannot break them."""
    before = profile(ID)
    after = profile(ID, AMOUNT)
    result = schema.diff_schemas(before, after)

    assert result.added == ["amount"]
    assert result.removed == []
    assert not result.breaking
    assert not schema.is_breaking(before, after)


def test_a_removed_column_is_breaking():
    before = profile(ID, AMOUNT)
    after = profile(ID)
    result = schema.diff_schemas(before, after)

    assert result.removed == ["amount"]
    assert [c.column for c in result.breaking] == ["amount"]
    assert schema.is_breaking(before, after)
    assert "[BREAKING]" in result.summary()


def test_a_retyped_column_is_breaking():
    before = profile(AMOUNT)
    after = profile(ColumnProfile("amount", "int64", row_count=5000, null_count=0))
    result = schema.diff_schemas(before, after)

    assert result.retyped == ["amount"]
    assert result.breaking
    assert schema.breaking_schema_changes(before, after)[0].kind == "retyped"


def test_change_renders_its_direction():
    before = profile(ID, AMOUNT)
    after = profile(
        ColumnProfile("amount", "int64", row_count=5000),
        ColumnProfile("new", "string"),
    )
    rendered = {str(c) for c in schema.diff_schemas(before, after).changes}

    assert "- id" in rendered
    assert "+ new" in rendered
    assert "~ amount: double => int64" in rendered


def test_column_change_kinds():
    assert schema.ColumnChange("x", None, AMOUNT).kind == "added"
    assert schema.ColumnChange("x", AMOUNT, None).kind == "removed"
    retyped = schema.ColumnChange("x", AMOUNT, ColumnProfile("amount", "int64"))
    assert retyped.kind == "retyped"
    assert retyped.is_breaking


def test_diff_profiles_returns_shape_and_values_separately():
    """One list whose entries mean two different things is what this avoids."""
    before = profile(ColumnProfile("amount", "double", row_count=5000, null_count=0))
    after = profile(ColumnProfile("amount", "int64", row_count=5000, null_count=4000))

    shape, drifted = schema.diff_profiles(before, after)
    assert shape.retyped == ["amount"]
    assert any("null rate" in line for line in drifted)


def test_worst_severity_reads_rendered_findings():
    assert schema.worst_severity(["[warn] a", "[error] b"]) == "error"
    assert schema.worst_severity(["[info] a"]) == "info"
    assert schema.worst_severity([]) == "none"
