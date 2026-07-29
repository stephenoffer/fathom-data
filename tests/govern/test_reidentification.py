"""Quasi-identifiers, and the refusal to ever certify a dataset as safe."""

from __future__ import annotations

import pytest

from fathom.core.partitions import PartitionMapping
from fathom.core.types import UNPARTITIONED, ColumnRef, DatasetId
from fathom.govern import reidentification as reid
from fathom.govern.policy import Label
from fathom.graph import Edge, Graph
from fathom.observe.profile import ColumnProfile, Profile, Severity

RAW = DatasetId("duckdb", "raw.people")
EXPORT_A = DatasetId("duckdb", "export.demographics")
EXPORT_B = DatasetId("duckdb", "export.locations")
UNRELATED = DatasetId("duckdb", "other.weather")


def column(name: str, *, distinct: int | None = None, rows: int = 1000) -> ColumnProfile:
    return ColumnProfile(name, "string", row_count=rows, distinct_estimate=distinct)


def profile(dataset: DatasetId, *columns: ColumnProfile, rows: int = 1000) -> Profile:
    return Profile(dataset=dataset, row_count=rows, columns=columns)


def labelled(dataset: DatasetId, **assignments: str) -> dict:
    return {
        ColumnRef(dataset, name): {Label(name=label, confidence=0.8)}
        for name, label in assignments.items()
    }


# -- identifying one at a time -------------------------------------------------


def test_a_direct_identifier_is_an_error():
    found = reid.assess(profile(RAW, column("email")), labelled(RAW, email="email"))
    assert not found.is_clear
    assert found.findings[0].kind == "direct_identifier"
    assert found.findings[0].severity is Severity.ERROR


def test_direct_identifiers_lists_only_labelled_columns():
    result = reid.direct_identifiers(
        profile(RAW, column("email"), column("colour")), labelled(RAW, email="email")
    )
    assert result == ["email"]


def test_a_near_unique_column_singles_out_whatever_it_is_called():
    """A salted hash carries no identifying label and identifies perfectly."""
    found = reid.assess(profile(RAW, column("order_ref", distinct=990)), {})
    assert [f.kind for f in found.findings] == ["singling_out"]
    assert "singles a row out" in found.findings[0].detail


def test_a_low_cardinality_column_does_not_single_out():
    assert reid.singling_out(profile(RAW, column("country", distinct=12))) == []


def test_singling_out_needs_a_distinct_count():
    assert reid.singling_out(profile(RAW, column("country"))) == []


def test_an_empty_dataset_singles_nobody_out():
    assert reid.singling_out(profile(RAW, column("x", distinct=5), rows=0)) == []


# -- identifying in combination ------------------------------------------------


def test_quasi_identifiers_are_recognized():
    found = reid.quasi_identifiers(
        profile(RAW, column("dob"), column("zip"), column("colour")),
        labelled(RAW, dob="date_of_birth", zip="postal_address"),
    )
    assert found == ["dob", "zip"]


def test_two_quasi_identifiers_with_a_small_group_are_an_error():
    """The whole point: neither column is an identifier and together they are."""
    found = reid.assess(
        profile(RAW, column("dob", distinct=800), column("zip", distinct=400)),
        labelled(RAW, dob="date_of_birth", zip="postal_address"),
    )
    risk = next(f for f in found.findings if f.kind == "quasi_identifier_set")
    assert risk.columns == ("dob", "zip")
    assert risk.k_upper is not None and risk.k_upper < 5


def test_a_single_quasi_identifier_is_not_a_combination_risk():
    found = reid.assess(
        profile(RAW, column("zip", distinct=400)), labelled(RAW, zip="postal_address")
    )
    assert [f for f in found.findings if f.kind == "quasi_identifier_set"] == []


def test_a_large_group_is_not_flagged():
    found = reid.assess(
        profile(RAW, column("dob", distinct=10), column("zip", distinct=5)),
        labelled(RAW, dob="date_of_birth", zip="postal_address"),
    )
    assert found.is_clear


def test_the_threshold_is_the_callers_choice():
    made = profile(RAW, column("dob", distinct=10), column("zip", distinct=5))
    labels = labelled(RAW, dob="date_of_birth", zip="postal_address")
    assert reid.assess(made, labels, k_threshold=5).is_clear
    assert not reid.assess(made, labels, k_threshold=500).is_clear


# -- the bound itself ----------------------------------------------------------


def test_the_bound_divides_rows_by_the_widest_column():
    made = profile(RAW, column("a", distinct=100), column("b", distinct=250))
    assert reid.k_upper_bound(made, ["a", "b"]) == pytest.approx(1000 / 250)


def test_the_bound_refuses_without_a_distinct_count():
    assert reid.k_upper_bound(profile(RAW, column("a")), ["a"]) is None


def test_the_bound_refuses_on_an_empty_dataset():
    assert reid.k_upper_bound(profile(RAW, column("a", distinct=5), rows=0), ["a"]) is None


def test_unknown_columns_do_not_contribute_to_the_bound():
    made = profile(RAW, column("a", distinct=100))
    assert reid.k_upper_bound(made, ["a", "absent"]) == pytest.approx(10.0)


def test_an_unmeasurable_quasi_set_is_reported_not_silently_cleared():
    found = reid.assess(
        profile(RAW, column("dob"), column("zip")),
        labelled(RAW, dob="date_of_birth", zip="postal_address"),
    )
    assert found.is_clear
    assert set(found.unmeasurable) == {"dob", "zip"}
    assert "not measurable" in found.summary()


# -- what a clear result means -------------------------------------------------


def test_a_clear_report_never_claims_safety():
    """`is_clear` means no risk was proven, which is not the same as safe."""
    found = reid.assess(profile(RAW, column("colour", distinct=4)), {})
    assert found.is_clear
    assert "not that the data is" in found.summary()
    assert "safe" in found.summary()


def test_the_summary_lists_the_quasi_identifiers_present():
    found = reid.assess(
        profile(RAW, column("dob", distinct=10), column("zip", distinct=5)),
        labelled(RAW, dob="date_of_birth", zip="postal_address"),
    )
    assert "quasi-identifiers present: dob, zip" in found.summary()


def test_risky_datasets_ranks_by_finding_count():
    worse = reid.assess(
        profile(RAW, column("email"), column("ref", distinct=999)),
        labelled(RAW, email="email"),
    )
    milder = reid.assess(profile(EXPORT_A, column("email")), labelled(EXPORT_A, email="email"))
    clean = reid.assess(profile(EXPORT_B, column("colour", distinct=3)), {})
    assert reid.risky_datasets([milder, worse, clean]) == [RAW, EXPORT_A]


# -- across datasets -----------------------------------------------------------


@pytest.fixture
def linked_graph() -> Graph:
    g = Graph()
    for ds in (RAW, EXPORT_A, EXPORT_B, UNRELATED):
        g.add_dataset(ds, UNPARTITIONED)
    mapping = PartitionMapping.identity(UNPARTITIONED)
    g.add_edge(Edge(RAW, EXPORT_A, mapping, evidence="sql:1"))
    g.add_edge(Edge(RAW, EXPORT_B, mapping, evidence="sql:2"))
    return g


def test_linkable_columns_are_the_shared_ones():
    left = profile(EXPORT_A, column("person_key"), column("dob"))
    right = profile(EXPORT_B, column("person_key"), column("zip"))
    assert reid.linkable_columns(left, right) == ["person_key"]


def test_two_defensible_exports_are_jointly_identifying(linked_graph):
    """Neither export's own review can see this, because the risk is in neither."""
    profiles = {
        EXPORT_A: profile(EXPORT_A, column("person_key"), column("dob")),
        EXPORT_B: profile(EXPORT_B, column("person_key"), column("zip")),
    }
    labels = {
        **labelled(EXPORT_A, dob="date_of_birth"),
        **labelled(EXPORT_B, zip="postal_address"),
    }
    (risk,) = reid.linkage_risks(linked_graph, profiles, labels)
    assert {risk.left, risk.right} == {EXPORT_A, EXPORT_B}
    assert risk.combined == ("date_of_birth", "postal_address")
    assert RAW in risk.via


def test_datasets_with_no_common_ancestor_are_not_linkable(linked_graph):
    profiles = {
        EXPORT_A: profile(EXPORT_A, column("person_key"), column("dob")),
        UNRELATED: profile(UNRELATED, column("person_key"), column("zip")),
    }
    labels = {
        **labelled(EXPORT_A, dob="date_of_birth"),
        **labelled(UNRELATED, zip="postal_address"),
    }
    assert reid.linkage_risks(linked_graph, profiles, labels) == []


def test_datasets_with_no_shared_column_are_not_linkable(linked_graph):
    profiles = {
        EXPORT_A: profile(EXPORT_A, column("a_key"), column("dob")),
        EXPORT_B: profile(EXPORT_B, column("b_key"), column("zip")),
    }
    labels = {
        **labelled(EXPORT_A, dob="date_of_birth"),
        **labelled(EXPORT_B, zip="postal_address"),
    }
    assert reid.linkage_risks(linked_graph, profiles, labels) == []


def test_joining_adds_nothing_when_one_side_has_no_quasi_identifier(linked_graph):
    profiles = {
        EXPORT_A: profile(EXPORT_A, column("person_key"), column("dob")),
        EXPORT_B: profile(EXPORT_B, column("person_key"), column("colour")),
    }
    assert reid.linkage_risks(linked_graph, profiles, labelled(EXPORT_A, dob="date_of_birth")) == []


def test_joining_the_same_quasi_identifier_twice_adds_nothing(linked_graph):
    profiles = {
        EXPORT_A: profile(EXPORT_A, column("person_key"), column("dob")),
        EXPORT_B: profile(EXPORT_B, column("person_key"), column("dob")),
    }
    labels = {**labelled(EXPORT_A, dob="date_of_birth"), **labelled(EXPORT_B, dob="date_of_birth")}
    assert reid.linkage_risks(linked_graph, profiles, labels) == []
