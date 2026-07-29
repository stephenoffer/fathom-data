"""Properties of re-identification and contracts.

These two fail *open* when they are wrong — a risk that is not reported reads as
safety, and a contract that reports "met" because nothing was supplied reads as a
promise kept. So the properties here are one-directional claims:

- `assess` never returns clear while holding a finding, and never reports a
  quasi-identifier risk it could not measure
- `k_upper_bound` is a genuine upper bound: monotone down in the widest column
- `verify` never reports met while holding a breach, and never counts an unsupplied
  promise as passed
"""

from __future__ import annotations

from datetime import timedelta

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fathom.core.types import ColumnRef, DatasetId
from fathom.govern import contracts
from fathom.govern import reidentification as reid
from fathom.govern.policy import Label
from fathom.observe.profile import ColumnProfile, Profile, Severity

DATASET = DatasetId("duckdb", "raw.people")

quasi_labels = st.sampled_from(sorted(reid.QUASI_IDENTIFIERS))
direct_labels = st.sampled_from(sorted(reid.DIRECT_IDENTIFIERS))
counts = st.integers(min_value=1, max_value=5000)


def profile_of(columns: list[tuple[str, int | None]], rows: int) -> Profile:
    return Profile(
        dataset=DATASET,
        row_count=rows,
        columns=tuple(
            ColumnProfile(name, "string", row_count=rows, distinct_estimate=distinct)
            for name, distinct in columns
        ),
    )


def labels_of(assignments: dict[str, str]) -> dict:
    return {
        ColumnRef(DATASET, column): {Label(name=label, confidence=0.8)}
        for column, label in assignments.items()
    }


# -- the bound -----------------------------------------------------------------


@given(rows=counts, distincts=st.lists(counts, min_size=1, max_size=5))
@settings(max_examples=300)
def test_the_bound_divides_by_the_widest_column(rows, distincts):
    """`C >= max(dᵢ)`, so the average group is at most `N / max(dᵢ)`."""
    columns = [(f"c{i}", d) for i, d in enumerate(distincts)]
    bound = reid.k_upper_bound(profile_of(columns, rows), [name for name, _ in columns])
    assert bound is not None
    assert bound == rows / max(distincts)


@given(rows=counts, distincts=st.lists(counts, min_size=1, max_size=5))
@settings(max_examples=300)
def test_adding_a_column_never_raises_the_bound(rows, distincts):
    """More quasi-identifiers can only make a group smaller, never larger."""
    columns = [(f"c{i}", d) for i, d in enumerate(distincts)]
    names = [name for name, _ in columns]
    before = reid.k_upper_bound(profile_of(columns, rows), names)

    wider = [*columns, ("extra", max(distincts) + 1)]
    after = reid.k_upper_bound(profile_of(wider, rows), [*names, "extra"])
    assert before is not None and after is not None
    assert after <= before


@given(rows=counts)
@settings(max_examples=200)
def test_a_bound_is_never_produced_without_a_distinct_count(rows):
    """Refusing to answer is not the same as answering zero."""
    assert reid.k_upper_bound(profile_of([("a", None)], rows), ["a"]) is None


# -- assess fails closed -------------------------------------------------------


@given(
    rows=counts,
    quasi=st.lists(st.tuples(quasi_labels, counts), max_size=4, unique_by=lambda t: t[0]),
    k=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=300)
def test_a_clear_report_holds_no_findings(rows, quasi, k):
    columns = [(f"q{i}", d) for i, (_, d) in enumerate(quasi)]
    labels = labels_of({f"q{i}": label for i, (label, _) in enumerate(quasi)})
    report = reid.assess(profile_of(columns, rows), labels, k_threshold=k)

    assert report.is_clear == (report.findings == [])
    assert ("no re-identification risk proven" in report.summary()) == report.is_clear


@given(
    rows=counts,
    quasi=st.lists(
        st.tuples(quasi_labels, counts), min_size=2, max_size=4, unique_by=lambda t: t[0]
    ),
    k=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=300)
def test_a_quasi_set_finding_appears_exactly_when_the_bound_is_under_k(rows, quasi, k):
    columns = [(f"q{i}", d) for i, (_, d) in enumerate(quasi)]
    names = [name for name, _ in columns]
    labels = labels_of({f"q{i}": label for i, (label, _) in enumerate(quasi)})

    report = reid.assess(profile_of(columns, rows), labels, k_threshold=k)
    bound = reid.k_upper_bound(profile_of(columns, rows), names)
    flagged = [f for f in report.findings if f.kind == "quasi_identifier_set"]

    assert bool(flagged) == (bound is not None and bound < k)
    if flagged:
        assert flagged[0].k_upper == bound


@given(rows=counts, label=direct_labels)
@settings(max_examples=200)
def test_a_direct_identifier_is_always_an_error(rows, label):
    report = reid.assess(profile_of([("who", None)], rows), labels_of({"who": label}))
    assert not report.is_clear
    assert any(f.severity is Severity.ERROR for f in report.findings)


@given(rows=counts, distinct=counts)
@settings(max_examples=300)
def test_singling_out_fires_exactly_at_the_ratio(rows, distinct):
    assume(distinct <= rows)
    found = reid.singling_out(profile_of([("ref", distinct)], rows), ratio=0.9)
    assert bool(found) == (distinct / rows >= 0.9)


@given(rows=counts)
@settings(max_examples=200)
def test_every_summary_carries_the_refusal_to_certify_safety(rows):
    """A clear result means no risk was proven, which is never the same as safe."""
    report = reid.assess(profile_of([("colour", 3)], rows), {})
    assert "not that the data is" in report.summary()


# -- contracts fail closed -----------------------------------------------------

column_names = st.lists(
    st.sampled_from(["order_id", "amount", "currency", "ts"]), max_size=4, unique=True
)


@given(
    promised=column_names, present=column_names, consumers=st.lists(st.text(max_size=4), max_size=3)
)
@settings(max_examples=300)
def test_a_met_report_holds_no_breaches(promised, present, consumers):
    contract = contracts.Contract(
        DATASET, "platform", consumers=tuple(consumers), columns=tuple(promised)
    )
    report = contracts.verify(contract, profile=profile_of([(c, None) for c in present], 10))

    assert report.is_met == (report.breaches == [])
    missing = set(promised) - set(present)
    assert {
        b.detail.split("'")[1] for b in report.breaches if b.kind == "missing_column"
    } == missing


@given(
    promised=column_names,
    present=column_names,
    consumers=st.lists(st.text(max_size=4), min_size=1, max_size=3),
)
@settings(max_examples=300)
def test_a_breach_with_consumers_is_always_an_error(promised, present, consumers):
    """Severity follows the blast radius, in one direction only."""
    contract = contracts.Contract(
        DATASET, "platform", consumers=tuple(consumers), columns=tuple(promised)
    )
    report = contracts.verify(contract, profile=profile_of([(c, None) for c in present], 10))
    assert all(b.severity is Severity.ERROR for b in report.breaches if b.kind == "missing_column")


@given(promised=column_names.filter(bool), staleness=st.integers(min_value=1, max_value=48))
@settings(max_examples=200)
def test_an_unsupplied_promise_is_unchecked_and_never_counted_as_passed(promised, staleness):
    contract = contracts.Contract(
        DATASET,
        "platform",
        columns=tuple(promised),
        max_staleness=timedelta(hours=staleness),
    )
    report = contracts.verify(contract)

    assert report.breaches == []
    assert len(report.unchecked) == 2  # columns and staleness both lacked evidence
    assert "not checked" in report.summary()


@given(age=st.integers(min_value=0, max_value=200), promise=st.integers(min_value=1, max_value=100))
@settings(max_examples=300)
def test_staleness_breaches_exactly_when_the_age_exceeds_the_promise(age, promise):
    contract = contracts.Contract(
        DATASET, "platform", consumers=("finance",), max_staleness=timedelta(hours=promise)
    )
    report = contracts.verify(contract, profile=profile_of([], 1), age=timedelta(hours=age))
    assert bool([b for b in report.breaches if b.kind == "staleness"]) == (age > promise)
