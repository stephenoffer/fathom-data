"""Properties of seasonal baselines.

The module's whole claim is that it refuses to model what it has not seen enough of.
That claim is only worth something if it holds for every shape of history, not the
three in the unit tests — so these are the properties:

- a learned band always contains the observations it was learned from
- a bucket is either modelled or recorded as unmodelled, never neither and never both
- an unmodelled bucket produces no findings, so silence always means "not modelled"
- learning is independent of the order observations arrive
"""

from __future__ import annotations

from datetime import datetime, timedelta

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fathom.core.types import DatasetId
from fathom.observe import seasonal
from fathom.observe.profile import ColumnProfile, Profile

EVENTS = DatasetId("duckdb", "raw.events")
MONDAY = datetime(2026, 3, 2)

rows = st.integers(min_value=0, max_value=100_000)
histories = st.lists(st.tuples(st.integers(min_value=0, max_value=41), rows), max_size=42)


def profile(count: int) -> Profile:
    return Profile(
        dataset=EVENTS,
        row_count=count,
        columns=(ColumnProfile("amount", "double", row_count=count, min=0, max=count),),
    )


def observations(raw) -> list[seasonal.Observation]:
    return [seasonal.Observation(MONDAY + timedelta(days=day), profile(n)) for day, n in raw]


@given(raw=histories.filter(bool))
@settings(max_examples=200)
def test_every_bucket_is_modelled_or_recorded_unmodelled_never_both(raw):
    """Silence has to mean exactly one thing, and the baseline has to say which."""
    history = observations(raw)
    baseline = seasonal.learn_seasonal(history)

    seen = {seasonal.bucket_of(o.when, baseline.cycle) for o in history}
    modelled = set(baseline.modelled_buckets)
    unmodelled = set(baseline.unmodelled)

    assert modelled | unmodelled == seen
    assert modelled.isdisjoint(unmodelled)


@given(raw=histories.filter(bool), minimum=st.integers(min_value=1, max_value=8))
@settings(max_examples=200)
def test_a_modelled_bucket_always_had_enough_observations(raw, minimum):
    history = observations(raw)
    baseline = seasonal.learn_seasonal(history, min_observations=minimum)

    counts: dict[int, int] = {}
    for o in history:
        index = seasonal.bucket_of(o.when, baseline.cycle)
        counts[index] = counts.get(index, 0) + 1

    for index in baseline.modelled_buckets:
        assert counts[index] >= minimum
    for index, seen in baseline.unmodelled.items():
        assert seen < minimum
        assert seen == counts[index]


@given(raw=histories.filter(bool))
@settings(max_examples=200)
def test_a_band_contains_every_observation_it_was_learned_from(raw):
    """Bounds widen and never tighten, so what was seen must always pass."""
    history = observations(raw)
    baseline = seasonal.learn_seasonal(history)

    for item in history:
        index = seasonal.bucket_of(item.when, baseline.cycle)
        band = baseline.band(None, "row_count", index)
        if band is not None:
            assert band.contains(float(item.profile.row_count))


@given(raw=histories.filter(bool))
@settings(max_examples=200)
def test_checking_an_observation_it_learned_from_produces_no_finding(raw):
    """A suite that fires on its own training data is a suite that gets disabled."""
    history = observations(raw)
    baseline = seasonal.learn_seasonal(history)
    for item in history:
        assert seasonal.check_seasonal(item, baseline) == []


@given(raw=histories.filter(bool), extreme=rows)
@settings(max_examples=200)
def test_an_unmodelled_bucket_never_produces_a_finding(raw, extreme):
    """Silence means 'not modelled', and `unmodelled` is where the baseline says so."""
    baseline = seasonal.learn_seasonal(observations(raw))
    for index in baseline.unmodelled:
        when = MONDAY + timedelta(days=index)
        assume(seasonal.bucket_of(when, baseline.cycle) == index)
        assert seasonal.check_seasonal(seasonal.Observation(when, profile(extreme)), baseline) == []


@given(raw=histories.filter(bool), shuffle=st.randoms(use_true_random=False))
@settings(max_examples=150)
def test_learning_does_not_depend_on_the_order_observations_arrive(raw, shuffle):
    history = observations(raw)
    reordered = list(history)
    shuffle.shuffle(reordered)

    first = seasonal.learn_seasonal(history)
    second = seasonal.learn_seasonal(reordered)

    assert first.bands.keys() == second.bands.keys()
    assert first.unmodelled == second.unmodelled
    for key, band in first.bands.items():
        other = second.bands[key]
        assert (band.low, band.high, band.observations) == (
            other.low,
            other.high,
            other.observations,
        )


@given(raw=histories.filter(bool))
@settings(max_examples=200)
def test_a_baseline_is_usable_exactly_when_it_has_a_band(raw):
    baseline = seasonal.learn_seasonal(observations(raw))
    assert baseline.is_usable == bool(baseline.bands)
    if not baseline.is_usable:
        assert "none reaching the minimum" in baseline.summary()


@given(raw=histories.filter(bool))
@settings(max_examples=150)
def test_strength_stays_within_zero_and_one_or_refuses(raw):
    score = seasonal.strength(observations(raw))
    assert score is None or 0.0 <= score <= 1.0


@given(cycle=st.sampled_from(list(seasonal.Cycle)), raw=histories.filter(bool))
@settings(max_examples=150)
def test_every_cycle_produces_a_consistent_baseline(cycle, raw):
    baseline = seasonal.learn_seasonal(observations(raw), cycle=cycle)
    assert baseline.cycle is cycle
    assert set(baseline.modelled_buckets).isdisjoint(baseline.unmodelled)
    for item in observations(raw):
        index = seasonal.bucket_of(item.when, cycle)
        band = baseline.band(None, "row_count", index)
        if band is not None:
            assert band.contains(float(item.profile.row_count))
