"""Sampling weights as an auditable decision.

The distinction worth having: upweighting math to 20% when the math corpus holds 3%
of the tokens is not "sample it more", it is "read it seven times". Those have
different effects on a model and a mixture that cannot tell them apart is not
recording the decision anyone actually made.
"""

from __future__ import annotations

import pytest

from fathom.ai.assets import corpus, model
from fathom.ai.train.mixtures import (
    Component,
    Mixture,
    component_edges,
    effective_epochs,
    epoch_pressure,
    mixture_edges,
    normalize,
    remix_plan,
    token_budget,
    unattributed,
    untested,
    validate,
)

BUDGET = 1_000_000


def mixture(*components: Component, name: str = "base", total: int = BUDGET) -> Mixture:
    return Mixture(name, components, total_tokens=total)


WEB = Component("web", 0.6, available_tokens=10_000_000, rationale="broad coverage")
CODE = Component("code", 0.4, available_tokens=10_000_000, decided_by="ablation-7")


# -- structure -----------------------------------------------------------------


def test_a_mixture_has_an_asset_identity():
    assert "base" in str(mixture(WEB, CODE).asset)


def test_components_are_addressable_by_source():
    assert mixture(WEB, CODE).component("code") is CODE
    assert mixture(WEB, CODE).component("missing") is None


def test_total_weight_sums_the_components():
    assert mixture(WEB, CODE).total_weight == pytest.approx(1.0)


def test_sources_are_listed_in_order():
    assert mixture(WEB, CODE).sources == ("web", "code")


# -- normalising ---------------------------------------------------------------


def test_normalising_rescales_while_preserving_proportion():
    """The missing 5% has to come from somewhere, and where it comes from changes the
    model."""
    rough = mixture(Component("a", 0.6), Component("b", 0.3))
    fixed = normalize(rough)
    assert fixed.total_weight == pytest.approx(1.0)
    assert fixed.component("a").weight == pytest.approx(2 / 3)


def test_normalising_preserves_rationale_and_size():
    fixed = normalize(mixture(WEB, CODE))
    assert fixed.component("web").rationale == "broad coverage"
    assert fixed.component("web").available_tokens == 10_000_000


def test_normalising_a_zero_weight_mixture_raises():
    with pytest.raises(ValueError, match="nothing would be sampled"):
        normalize(mixture(Component("a", 0.0)))


# -- validation ----------------------------------------------------------------


def test_a_well_formed_mixture_has_no_problems():
    assert validate(mixture(WEB, CODE)) == []


def test_an_empty_mixture_is_caught():
    assert validate(mixture()) == ["mixture has no components; nothing would be sampled"]


def test_weights_that_do_not_sum_to_one_are_caught():
    problems = validate(mixture(Component("a", 0.5), Component("b", 0.3)))
    assert any("sum to" in p for p in problems)


def test_a_zero_weight_component_is_caught():
    """Leaving it in implies it was considered."""
    problems = validate(mixture(Component("a", 1.0), Component("b", 0.0)))
    assert any("contributes nothing" in p for p in problems)


def test_a_negative_weight_is_caught():
    problems = validate(mixture(Component("a", 1.2), Component("b", -0.2)))
    assert any("negative weight" in p for p in problems)


def test_a_duplicated_source_is_caught():
    """Its shares would silently add."""
    problems = validate(mixture(Component("a", 0.5), Component("a", 0.5)))
    assert any("more than once" in p for p in problems)


# -- budgets -------------------------------------------------------------------


def test_the_token_budget_splits_by_weight():
    assert token_budget(mixture(WEB, CODE)) == {"web": 600_000, "code": 400_000}


def test_an_explicit_budget_overrides_the_recorded_one():
    assert token_budget(mixture(WEB, CODE), 100)["web"] == 60


# -- epoching ------------------------------------------------------------------


def test_a_component_with_enough_data_needs_less_than_one_epoch():
    pressure = effective_epochs(WEB, BUDGET)
    assert pressure.epochs < 1
    assert not pressure.repeats


def test_a_small_component_at_a_high_weight_must_repeat():
    """Upweighting math to 20% when the corpus holds 3% means reading it seven times."""
    math = Component("math", 0.2, available_tokens=30_000)
    pressure = effective_epochs(math, BUDGET)
    assert pressure.repeats
    assert pressure.epochs == pytest.approx(200_000 / 30_000)


def test_an_unrecorded_size_is_unknown_rather_than_infinite_confidence():
    pressure = effective_epochs(Component("x", 0.5), BUDGET)
    assert pressure.is_unknown
    assert pressure.epochs == float("inf")


def test_epoch_pressure_reports_only_the_components_that_repeat():
    small = Component("math", 0.2, available_tokens=30_000)
    pressures = epoch_pressure(mixture(WEB, small, name="m"), BUDGET)
    assert [p.source for p in pressures] == ["math"]


def test_components_with_no_recorded_size_are_reported_too():
    """A silent omission reads as "this one is fine"."""
    pressures = epoch_pressure(mixture(WEB, Component("x", 0.4)), BUDGET)
    assert any(p.is_unknown for p in pressures)


def test_pressures_come_worst_first():
    a = Component("a", 0.5, available_tokens=100_000)
    b = Component("b", 0.5, available_tokens=10_000)
    assert [p.source for p in epoch_pressure(mixture(a, b), BUDGET)] == ["b", "a"]


# -- provenance ----------------------------------------------------------------


def test_a_weight_with_a_rationale_is_attributed():
    assert WEB.is_attributed
    assert CODE.is_attributed


def test_a_bare_weight_is_an_inheritance_not_a_decision():
    assert not Component("books", 0.1).is_attributed
    assert unattributed(mixture(WEB, Component("books", 0.1))) == ["books"]


def test_a_component_no_ablation_varied_is_untested():
    """An ablation that holds a weight fixed says nothing about it."""
    ablations = [{"web": 0.6, "code": 0.4}, {"web": 0.5, "code": 0.4}]
    assert untested(mixture(WEB, CODE), ablations) == ["code"]


def test_everything_is_untested_when_there_are_no_ablations():
    assert untested(mixture(WEB, CODE), []) == ["web", "code"]


# -- lineage -------------------------------------------------------------------


def test_components_with_a_dataset_become_edges():
    known = Component("web", 0.5, dataset=corpus("web"))
    edges = component_edges(mixture(known, CODE))
    assert edges == [(corpus("web"), mixture(known, CODE).asset)]


def test_a_component_fathom_cannot_identify_produces_no_edge():
    """That gap is worth seeing, not papering over with a synthesised identity."""
    assert component_edges(mixture(WEB, CODE)) == []


def test_the_mixture_feeds_every_model_trained_under_it():
    edges = mixture_edges(mixture(WEB, CODE), [model("m1"), model("m2")])
    assert len(edges) == 2


# -- re-mixture ----------------------------------------------------------------


def test_an_identical_remix_costs_nothing():
    plan = remix_plan(mixture(WEB, CODE), mixture(WEB, CODE))
    assert plan.unchanged
    assert plan.resample_tokens == 0
    assert "identical" in plan.summary()


def test_a_remix_prices_only_the_increases():
    """A component that shrank simply contributes less, which costs nothing to do."""
    after = mixture(Component("web", 0.4), Component("code", 0.6), name="v2")
    plan = remix_plan(mixture(WEB, CODE), after)
    assert plan.resample_tokens == 200_000


def test_an_added_component_is_marked():
    after = mixture(
        Component("web", 0.5), Component("code", 0.4), Component("math", 0.1), name="v2"
    )
    plan = remix_plan(mixture(WEB, CODE), after)
    assert next(c for c in plan.changes if c.source == "math").added


def test_a_dropped_component_is_marked():
    after = mixture(Component("web", 1.0), name="v2")
    plan = remix_plan(mixture(WEB, CODE), after)
    assert next(c for c in plan.changes if c.source == "code").dropped
    assert "(dropped)" in plan.summary()


def test_changes_are_ranked_by_magnitude():
    after = mixture(Component("web", 0.35), Component("code", 0.65), name="v2")
    plan = remix_plan(mixture(WEB, CODE), after)
    assert abs(plan.significant[0].delta) >= abs(plan.significant[-1].delta)


def test_newly_repeated_components_are_called_out():
    """Repeating data is a different decision from sampling more of it."""
    before = mixture(
        Component("web", 0.9, available_tokens=10_000_000),
        Component("math", 0.1, available_tokens=200_000),
    )
    after = mixture(
        Component("web", 0.5, available_tokens=10_000_000),
        Component("math", 0.5, available_tokens=200_000),
        name="v2",
    )
    plan = remix_plan(before, after, BUDGET)
    assert [p.source for p in plan.newly_repeated] == ["math"]
    assert "repeats data" in plan.summary()


def test_a_component_already_repeating_is_not_newly_repeated():
    before = mixture(Component("math", 1.0, available_tokens=1000))
    after = mixture(Component("math", 1.0, available_tokens=1000), name="v2")
    assert remix_plan(before, after, BUDGET).newly_repeated == ()


def test_relative_change_of_a_new_component_does_not_divide_by_zero():
    after = mixture(Component("web", 0.6), Component("code", 0.3), Component("new", 0.1), name="v2")
    plan = remix_plan(mixture(WEB, CODE), after)
    assert next(c for c in plan.changes if c.source == "new").relative == 0.0
