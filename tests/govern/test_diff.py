"""Label changes between two runs of inference and propagation.

The one line of this diff that belongs in a compliance review is `new_pii`. Everything
else is churn from a confidence threshold moving, and burying the first in the second
is how a review stops being read.
"""

from __future__ import annotations

from fathom.core.ids import normalize_table
from fathom.core.types import ColumnRef
from fathom.govern.diff import diff_labels
from fathom.govern.policy import Label

RAW = normalize_table("raw.users", system="duckdb")
EMAIL = ColumnRef(RAW, "email")
PHONE = ColumnRef(RAW, "phone")


def test_no_change_is_empty():
    labels = {EMAIL: {Label("pii", 0.9, "inferred")}}
    result = diff_labels(labels, labels)

    assert result.is_empty
    assert result.summary() == "labels: no change"


def test_an_added_label_is_reported():
    result = diff_labels({}, {EMAIL: {Label("email", 0.75, "inferred")}})

    assert {label.name for label in result.added[EMAIL]} == {"email"}
    assert not result.removed
    assert "+1" in result.summary()


def test_a_removed_label_is_reported():
    """A label that stopped being inferred is a change somebody may need to explain."""
    result = diff_labels({EMAIL: {Label("email", 0.75)}}, {})

    assert {label.name for label in result.removed[EMAIL]} == {"email"}
    assert not result.added


def test_a_confidence_move_is_its_own_event():
    """Not an add and not a remove: the claim is the same, the evidence changed."""
    before = {EMAIL: {Label("pii", 0.6, "inferred:name")}}
    after = {EMAIL: {Label("pii", 0.9, "inferred:name+stats")}}
    result = diff_labels(before, after)

    assert not result.added
    assert not result.removed
    assert len(result.reconfidenced) == 1
    ref, old, new = result.reconfidenced[0]
    assert (ref, old.confidence, new.confidence) == (EMAIL, 0.6, 0.9)
    assert "1 confidence change" in result.summary()


def test_a_human_confirmation_registers_as_a_change():
    before = {EMAIL: {Label("pii", 0.9, "inferred", confirmed=False)}}
    after = {EMAIL: {Label("pii", 0.9, "inferred", confirmed=True)}}

    assert diff_labels(before, after).reconfidenced


def test_new_pii_is_surfaced_separately_from_everything_else():
    before = {EMAIL: {Label("email", 0.75)}}
    after = {
        EMAIL: {Label("email", 0.75), Label("pii", 0.75, "implied")},
        PHONE: {Label("phone", 0.7)},
    }
    result = diff_labels(before, after)

    assert result.new_pii == [EMAIL]  # phone gained a label, but not the pii label
    assert "newly labelled pii" in result.summary()


def test_no_new_pii_leaves_the_headline_quiet():
    result = diff_labels({}, {PHONE: {Label("phone", 0.7)}})

    assert result.new_pii == []
    assert "newly labelled pii" not in result.summary()


def test_changes_across_several_columns_are_kept_apart():
    before = {EMAIL: {Label("pii", 0.9)}}
    after = {PHONE: {Label("pii", 0.9)}}
    result = diff_labels(before, after)

    assert list(result.added) == [PHONE]
    assert list(result.removed) == [EMAIL]
