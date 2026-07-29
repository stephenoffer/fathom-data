"""Tests for the partition mapping lattice.

The property tests matter more than the unit tests here. A composition rule that is
merely *usually* right produces a planner that serves stale data intermittently,
which is worse than one that is obviously broken.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fathom.core.grains import Grain, convert_window
from fathom.core.partitions import (
    UNBOUNDED,
    PartitionMapping,
    Passthrough,
    TimeWindow,
    apply,
    compose,
    join,
    leq,
)
from fathom.core.types import ANY, KeyPredicate, PartitionField, PartitionSpec

BIG = 10_000_000  # effectively disable the enumeration cap inside property tests


def spec_for(grain: Grain, name: str = "dt") -> PartitionSpec:
    return PartitionSpec.of(PartitionField.time(name, grain))


def subsumes(outer: KeyPredicate, inner: KeyPredicate) -> bool:
    names = {k for k, _ in outer.bindings} | {k for k, _ in inner.bindings}
    for n in names:
        ov, iv = outer.get(n), inner.get(n)
        if ov is ANY:
            continue
        if iv is ANY or ov != iv:
            return False
    return True


def covers(wide: frozenset[KeyPredicate], narrow: frozenset[KeyPredicate]) -> bool:
    """Every predicate in `narrow` is covered by some predicate in `wide`."""
    return all(any(subsumes(w, n) for w in wide) for n in narrow)


# -- unit ---------------------------------------------------------------------


def test_identity_maps_a_day_to_itself():
    m = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY))
    got = apply(m, KeyPredicate.of(dt=datetime(2026, 3, 14)), spec_for(Grain.DAY))
    assert got == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 14))})


def test_rollup_maps_a_day_to_its_month():
    m = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.MONTH))
    got = apply(m, KeyPredicate.of(dt=datetime(2026, 3, 14)), spec_for(Grain.MONTH))
    assert got == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 1))})


def test_trailing_window_taints_following_days():
    m = PartitionMapping.of(dt=TimeWindow("dt", 0, 6, Grain.DAY, Grain.DAY))
    got = apply(m, KeyPredicate.of(dt=datetime(2026, 3, 14)), spec_for(Grain.DAY))
    assert {k.get("dt").day for k in got} == set(range(14, 21))


def test_unbounded_field_yields_any():
    m = PartitionMapping.of(dt=UNBOUNDED)
    got = apply(m, KeyPredicate.of(dt=datetime(2026, 3, 14)), spec_for(Grain.DAY))
    assert got == frozenset({KeyPredicate.of(dt=ANY)})


def test_unknown_input_stays_unknown():
    """An unconstrained input cannot produce a constrained output."""
    m = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY))
    got = apply(m, KeyPredicate.of(dt=ANY), spec_for(Grain.DAY))
    assert got == frozenset({KeyPredicate.of(dt=ANY)})


def test_grain_mismatch_widens_rather_than_guesses():
    """Two edges disagreeing about the middle grain must not be silently reconciled."""
    a = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.HOUR, Grain.DAY))
    b = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.MONTH, Grain.YEAR))
    assert compose(a, b).get("dt") is UNBOUNDED


def test_refinement_is_rejected_at_construction():
    """A coarse source feeding a finer table has no useful bound; say so loudly."""
    with pytest.raises(ValueError, match="finer than input"):
        TimeWindow("dt", 0, 0, Grain.MONTH, Grain.DAY)


def test_rollup_helper_widens_on_refinement():
    src = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))
    dst = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
    assert PartitionMapping.rollup(src, dst).get("dt") is UNBOUNDED


def test_compose_accumulates_windows():
    a = PartitionMapping.of(dt=TimeWindow("dt", -1, 2, Grain.DAY, Grain.DAY))
    b = PartitionMapping.of(dt=TimeWindow("dt", 0, 3, Grain.DAY, Grain.DAY))
    got = compose(a, b).get("dt")
    assert isinstance(got, TimeWindow)
    assert (got.lo, got.hi) == (-1, 5)


def test_compose_through_passthrough_keeps_origin_field():
    """A verbatim copy in the middle must not lose the original source column."""
    a = PartitionMapping.of(day=Passthrough("event_date"))
    b = PartitionMapping.of(dt=TimeWindow("day", 0, 0, Grain.DAY, Grain.MONTH))
    got = compose(a, b).get("dt")
    assert isinstance(got, TimeWindow)
    assert got.source == "event_date"


def test_join_takes_the_union_of_windows():
    a = PartitionMapping.of(dt=TimeWindow("dt", -2, 0, Grain.DAY, Grain.DAY))
    b = PartitionMapping.of(dt=TimeWindow("dt", 0, 5, Grain.DAY, Grain.DAY))
    got = join(a, b).get("dt")
    assert isinstance(got, TimeWindow)
    assert (got.lo, got.hi) == (-2, 5)


def test_join_of_incompatible_forms_widens():
    a = PartitionMapping.of(dt=Passthrough("dt"))
    b = PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY))
    assert join(a, b).get("dt") is UNBOUNDED


def test_enumeration_cap_widens_instead_of_exploding():
    """Past the cap we must collapse to ANY, never truncate the key list."""
    m = PartitionMapping.of(dt=TimeWindow("dt", 0, 5000, Grain.HOUR, Grain.HOUR))
    got = apply(m, KeyPredicate.of(dt=datetime(2026, 1, 1)), spec_for(Grain.HOUR), max_keys=64)
    assert got == frozenset({KeyPredicate.of(dt=ANY)})


def test_empty_window_is_rejected():
    with pytest.raises(ValueError, match="empty window"):
        TimeWindow("dt", 3, 1, Grain.DAY, Grain.DAY)


def test_multi_field_spec_takes_cross_product():
    m = PartitionMapping.of(
        dt=TimeWindow("dt", 0, 1, Grain.DAY, Grain.DAY),
        region=Passthrough("region"),
    )
    spec = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
    got = apply(m, KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu"), spec)
    assert len(got) == 2
    assert all(k.get("region") == "eu" for k in got)


# -- properties ---------------------------------------------------------------

grains = st.sampled_from(list(Grain))
offsets = st.integers(min_value=-2, max_value=2)
instants = st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2040, 12, 31))


@st.composite
def windows(draw, in_grain: Grain, out_grain: Grain) -> TimeWindow:
    lo = draw(offsets)
    hi = draw(st.integers(min_value=lo, max_value=lo + 4))
    return TimeWindow("dt", lo, hi, in_grain, out_grain)


@st.composite
def grain_chain(draw, length: int) -> list[Grain]:
    """A non-decreasing run of grains, the only shape a real pipeline produces."""
    return sorted(draw(st.lists(grains, min_size=length, max_size=length)))


@given(chain=grain_chain(3), k=instants, data=st.data())
@settings(max_examples=500, deadline=None)
def test_compose_over_approximates_stepwise_application(chain, k, data):
    """The invariant the whole planner rests on.

    Collapsing A→B→C into one mapping must never claim fewer dirty partitions than
    walking the two edges separately would.
    """
    ga, gb, gc = chain
    m1 = PartitionMapping(fields=(("dt", data.draw(windows(ga, gb))),))
    m2 = PartitionMapping(fields=(("dt", data.draw(windows(gb, gc))),))
    key = KeyPredicate.of(dt=k)

    stepwise: set[KeyPredicate] = set()
    for mid in apply(m1, key, spec_for(gb), max_keys=BIG):
        stepwise |= apply(m2, mid, spec_for(gc), max_keys=BIG)

    composed = compose(m1, m2)
    collapsed = apply(composed, key, spec_for(gc), max_keys=BIG)
    missing = sorted(str(s) for s in stepwise if not any(subsumes(c, s) for c in collapsed))

    assert covers(collapsed, frozenset(stepwise)), (
        f"composed mapping missed partitions\n"
        f"  m1={m1}\n  m2={m2}\n  composed={composed}\n  missing={missing}"
    )


@given(chain=grain_chain(2), k=instants, data=st.data())
@settings(max_examples=300, deadline=None)
def test_join_covers_both_operands(chain, k, data):
    ga, gb = chain
    a = PartitionMapping(fields=(("dt", data.draw(windows(ga, gb))),))
    b = PartitionMapping(fields=(("dt", data.draw(windows(ga, gb))),))
    key = KeyPredicate.of(dt=k)
    joined = apply(join(a, b), key, spec_for(gb), max_keys=BIG)
    assert covers(joined, apply(a, key, spec_for(gb), max_keys=BIG))
    assert covers(joined, apply(b, key, spec_for(gb), max_keys=BIG))


@given(chain=grain_chain(2), data=st.data())
@settings(max_examples=200, deadline=None)
def test_join_is_commutative_and_idempotent(chain, data):
    ga, gb = chain
    a = PartitionMapping(fields=(("dt", data.draw(windows(ga, gb))),))
    b = PartitionMapping(fields=(("dt", data.draw(windows(ga, gb))),))
    assert join(a, b) == join(b, a)
    assert join(a, a) == a


@given(chain=grain_chain(2), data=st.data())
@settings(max_examples=200, deadline=None)
def test_join_result_dominates_operands_in_the_order(chain, data):
    ga, gb = chain
    a = PartitionMapping(fields=(("dt", data.draw(windows(ga, gb))),))
    b = PartitionMapping(fields=(("dt", data.draw(windows(ga, gb))),))
    assert leq(a, join(a, b))
    assert leq(b, join(a, b))


@given(chain=grain_chain(2), data=st.data())
@settings(max_examples=200, deadline=None)
def test_unbounded_absorbs_under_composition(chain, data):
    ga, gb = chain
    m = PartitionMapping(fields=(("dt", data.draw(windows(ga, gb))),))
    top = PartitionMapping.of(dt=UNBOUNDED)
    assert compose(top, m).get("dt") is UNBOUNDED
    assert compose(m, top).get("dt") is UNBOUNDED
    assert join(m, top).get("dt") is UNBOUNDED


@given(g=grains, data=st.data())
@settings(max_examples=100, deadline=None)
def test_identity_is_a_unit_for_composition(g, data):
    """Composing through an unchanged hop must not cost precision."""
    m = PartitionMapping(fields=(("dt", data.draw(windows(g, g))),))
    ident = PartitionMapping.identity(spec_for(g))
    assert compose(ident, m) == m
    assert compose(m, ident) == m


@given(lo=st.integers(-30, 30), width=st.integers(0, 30), chain=grain_chain(2))
def test_window_conversion_only_ever_widens(lo, width, chain):
    """Structural invariants of the conversion. End-to-end soundness is covered by
    the composition property above, which checks actual partition coverage."""
    frm, to = chain
    hi = lo + width
    got = convert_window(lo, hi, frm, to)
    assert got is not None
    out_lo, out_hi = got
    assert out_lo <= out_hi
    if frm is to:
        assert (out_lo, out_hi) == (lo, hi)
    else:
        # Coarsening always brackets the source bucket itself, because the input
        # instant sits somewhere inside it.
        assert out_lo <= 0 < out_hi


@given(lo=st.integers(-20, 0), extra=st.integers(0, 20), chain=grain_chain(2))
def test_window_conversion_is_monotone(lo, extra, chain):
    """A wider input window can never convert to a narrower output window."""
    frm, to = chain
    narrow = convert_window(lo, 0, frm, to)
    wide = convert_window(lo - extra, extra, frm, to)
    assert narrow is not None and wide is not None
    assert wide[0] <= narrow[0]
    assert wide[1] >= narrow[1]


@given(chain=grain_chain(2))
def test_window_conversion_refuses_refinement(chain):
    fine, coarse = chain
    if fine is coarse:
        return
    assert convert_window(0, 0, coarse, fine) is None


# -- covered_by ----------------------------------------------------------------


def test_covered_by_is_what_shadow_mode_grades_against():
    """A planned `dt=ANY` legitimately covers every concrete partition under it."""
    from fathom.core.types import ANY, KeyPredicate, covered_by

    concrete = KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
    wide = KeyPredicate.of(dt=ANY, region="eu")
    other = KeyPredicate.of(dt=datetime(2026, 3, 15), region="eu")

    assert covered_by([wide], concrete)
    assert covered_by([concrete, other], concrete)
    assert not covered_by([other], concrete)
    assert not covered_by([], concrete)


def test_covered_by_needs_every_field_to_agree():
    from fathom.core.types import ANY, KeyPredicate, covered_by

    concrete = KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu")
    wrong_region = KeyPredicate.of(dt=ANY, region="us")

    assert not covered_by([wrong_region], concrete)


# -- named constructors and explanations ---------------------------------------
#
# A mapping is the one thing in the graph nobody can check by reading the code that
# produced it. These cover the two things that make that reviewable: constructors
# named for the shape they build, and a sentence saying what the mapping claims.


def test_identity_is_the_same_bucket_on_both_sides():
    assert TimeWindow.identity("dt", "day") == TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)


def test_rollup_states_the_grain_change_and_nothing_else():
    assert TimeWindow.rollup("dt", "day", "month") == TimeWindow("dt", 0, 0, Grain.DAY, Grain.MONTH)


def test_trailing_takes_a_length_rather_than_offsets():
    """`length=7` cannot be off by one the way `(0, 6)` written by hand can."""
    assert TimeWindow.trailing("dt", 7, "day") == TimeWindow("dt", 0, 6, Grain.DAY, Grain.DAY)


def test_a_trailing_window_must_cover_at_least_one_bucket():
    with pytest.raises(ValueError, match="at least one bucket"):
        TimeWindow.trailing("dt", 0, "day")


def test_the_constructors_accept_grain_names():
    assert TimeWindow.identity("dt", "daily") == TimeWindow.identity("dt", Grain.DAY)


def test_a_backwards_window_suggests_the_constructor_that_gets_it_right():
    with pytest.raises(ValueError) as exc:
        TimeWindow("dt", 6, 0, Grain.DAY, Grain.DAY)
    assert "TimeWindow.trailing('dt', 7, Grain.DAY)" in str(exc.value)


def test_a_refining_window_explains_why_it_is_refused():
    with pytest.raises(ValueError) as exc:
        TimeWindow("dt", 0, 0, Grain.MONTH, Grain.DAY)
    message = str(exc.value)
    assert "UNBOUNDED" in message
    assert "serves stale data" in message


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (TimeWindow.identity("dt", "day"), "the same day"),
        (TimeWindow.rollup("dt", "day", "month"), "the month containing it"),
        (TimeWindow.trailing("dt", 7, "day"), "the 6 days after it"),
        (TimeWindow("dt", -3, 0, Grain.DAY, Grain.DAY), "the 3 days before it"),
        (Passthrough("region"), "that same region"),
        (UNBOUNDED, "no relationship was provable"),
    ],
)
def test_every_field_mapping_can_say_what_it_claims(mapping, expected: str):
    assert expected in mapping.explain()


def test_a_whole_mapping_explains_one_field_per_line():
    mapping = PartitionMapping.of(
        dt=TimeWindow.rollup("dt", "day", "month"),
        region=Passthrough("region"),
    )
    lines = mapping.explain().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("dt: ")
    assert lines[1].startswith("region: ")


def test_an_empty_mapping_says_what_it_will_cost():
    assert "whole output" in PartitionMapping().explain()


def test_reprs_are_the_expression_that_rebuilds_them():
    assert repr(Passthrough("region")) == "Passthrough('region')"
    assert repr(UNBOUNDED) == "UNBOUNDED"
    assert repr(TimeWindow.identity("dt", "day")) == "TimeWindow('dt', 0, 0, day, day)"
