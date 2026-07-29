"""Deployments, traffic, and quantization regressions.

Two behaviours carry the module. A rollback must refuse when there is nothing to roll
back to, and a quantization check must catch the one capability that collapsed while
the aggregate barely moved — which is what actually ships broken models.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fathom.ai.serve import (
    CapabilityResult,
    Deployment,
    DeploymentState,
    RolloutStrategy,
    TrafficSplit,
    Variant,
    active_variants,
    can_rollback,
    deployment_edges,
    is_canary,
    promote,
    regression_report,
    rollback,
    rollout_plan,
    traffic_to,
    validate_split,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def live(name: str = "v1", *, weight: float = 1.0, retained: bool = True) -> Variant:
    return Variant(name, f"model-{name}", DeploymentState.LIVE, weight, retained=retained)


def canary(name: str = "v2", *, weight: float = 0.1) -> Variant:
    return Variant(name, f"model-{name}", DeploymentState.CANARY, weight)


def endpoint(*variants: Variant, strategy: RolloutStrategy = RolloutStrategy.CANARY) -> Deployment:
    return Deployment("chat", list(variants), strategy=strategy)


# -- state ---------------------------------------------------------------------


def test_pending_and_retired_variants_do_not_serve():
    assert not DeploymentState.PENDING.serves_traffic
    assert not DeploymentState.RETIRED.serves_traffic


def test_draining_still_serves():
    """It has connections open. Treating it as gone is how requests get dropped."""
    assert DeploymentState.DRAINING.serves_traffic


def test_active_variants_excludes_the_ones_not_serving():
    deployment = endpoint(live(), Variant("old", "m", DeploymentState.RETIRED))
    assert [v.name for v in active_variants(deployment)] == ["v1"]


def test_traffic_to_a_non_serving_variant_is_zero():
    deployment = endpoint(Variant("v9", "m", DeploymentState.PENDING, weight=0.5))
    assert traffic_to(deployment, "v9") == 0.0


def test_traffic_to_an_unknown_variant_is_zero():
    assert traffic_to(endpoint(live()), "nope") == 0.0


def test_is_canary_identifies_the_variant_under_test():
    deployment = endpoint(live(weight=0.9), canary())
    assert is_canary(deployment, "v2")
    assert not is_canary(deployment, "v1")


def test_a_variant_knows_whether_it_is_quantized():
    assert Variant("v", "m", quantization="int4").is_quantized
    assert not Variant("v", "m").is_quantized


# -- traffic splits ------------------------------------------------------------


def test_a_valid_split_has_no_problems():
    assert validate_split(endpoint(live(weight=0.9), canary(weight=0.1))) == []


def test_weights_that_do_not_sum_to_one_are_caught():
    """Some share of traffic is unrouted, which is a silent partial outage."""
    problems = validate_split(endpoint(live(weight=0.5), canary(weight=0.1)))
    assert any("sum to" in p for p in problems)


def test_no_serving_variant_is_caught():
    problems = validate_split(endpoint(Variant("v", "m", DeploymentState.PENDING)))
    assert problems == ["no variant is serving traffic"]


def test_negative_weights_are_caught():
    problems = validate_split(endpoint(live(weight=1.2), canary(weight=-0.2)))
    assert any("negative weight" in p for p in problems)


def test_blue_green_with_two_live_variants_is_caught():
    deployment = endpoint(
        live("a", weight=0.5), live("b", weight=0.5), strategy=RolloutStrategy.BLUE_GREEN
    )
    assert any("blue/green" in p for p in validate_split(deployment))


def test_two_live_variants_are_fine_under_a_canary_strategy():
    deployment = endpoint(live("a", weight=0.5), live("b", weight=0.5))
    assert validate_split(deployment) == []


def test_a_traffic_split_reports_shares():
    split = TrafficSplit({"a": 0.7, "b": 0.3})
    assert split.total == pytest.approx(1.0)
    assert split.share("a") == 0.7
    assert split.share("missing") == 0.0


# -- lineage -------------------------------------------------------------------


def test_serving_variants_become_edges_into_the_endpoint():
    """Without these, "which model is answering production traffic" is unanswerable."""
    edges = deployment_edges(endpoint(live(), canary()))
    assert len(edges) == 2
    assert all(target == endpoint().dataset for _, target in edges)


def test_non_serving_variants_produce_no_edges():
    assert deployment_edges(endpoint(Variant("v", "m", DeploymentState.RETIRED))) == []


def test_the_endpoint_identity_includes_the_environment():
    staging = Deployment("chat", [], environment="staging")
    assert Deployment("chat", []).dataset != staging.dataset


# -- rollout -------------------------------------------------------------------


def test_a_rollout_plan_names_the_full_split_at_each_step():
    """A rollout halted midway should leave a state somebody can read."""
    plan = rollout_plan(endpoint(live()), "v2", steps=(0.1, 1.0))
    assert plan[0]["v2"] == 0.1
    assert plan[0]["v1"] == pytest.approx(0.9)
    assert plan[-1]["v2"] == 1.0


def test_each_rollout_step_sums_to_one():
    plan = rollout_plan(endpoint(live("a", weight=0.5), live("b", weight=0.5)), "new")
    for step in plan:
        assert sum(step.values()) == pytest.approx(1.0)


def test_promotion_takes_a_variant_to_full_traffic():
    deployment = promote(endpoint(live(), canary()), "v2", at=NOW)
    assert traffic_to(deployment, "v2") == 1.0
    assert deployment.variant("v2").state is DeploymentState.LIVE


def test_promotion_drains_the_others_rather_than_deleting_them():
    """A drained variant still has open connections and is still the rollback target."""
    deployment = promote(endpoint(live(), canary()), "v2", at=NOW)
    assert deployment.variant("v1").state is DeploymentState.DRAINING
    assert traffic_to(deployment, "v1") == 0.0


def test_promoting_an_unknown_variant_raises():
    with pytest.raises(KeyError, match="no variant"):
        promote(endpoint(live()), "ghost")


# -- rollback ------------------------------------------------------------------


def test_rollback_is_possible_while_the_previous_variant_is_retained():
    ok, why = can_rollback(endpoint(live(), canary()), "v1")
    assert ok
    assert why == ""


def test_rollback_refuses_when_the_artefact_was_garbage_collected():
    """Discovered before the rollout rather than during the rollback."""
    deployment = endpoint(live("v1", retained=False), canary())
    ok, why = can_rollback(deployment, "v1")
    assert not ok
    assert "garbage-collected" in why


def test_rollback_refuses_an_unknown_variant():
    ok, why = can_rollback(endpoint(live()), "never-existed")
    assert not ok
    assert "no variant" in why


def test_rollback_raises_rather_than_silently_doing_nothing():
    deployment = endpoint(live("v1", retained=False), canary())
    with pytest.raises(RuntimeError, match="garbage-collected"):
        rollback(deployment, "v1")


def test_a_successful_rollback_restores_traffic():
    deployment = promote(endpoint(live(), canary()), "v2", at=NOW)
    rollback(deployment, "v1", at=NOW)
    assert traffic_to(deployment, "v1") == 1.0


# -- quantization regression ---------------------------------------------------


def test_a_narrow_collapse_is_caught_when_the_aggregate_looks_fine():
    """Int4 costs half a point of perplexity and destroys one capability. The
    aggregate moving 0.4% is exactly what ships a broken model."""
    report = regression_report(
        "int4",
        "bf16",
        [
            CapabilityResult("general", 0.800, 0.798, samples=10_000),
            CapabilityResult("arithmetic", 0.700, 0.400, samples=200),
        ],
    )
    assert abs(report.aggregate_delta) < 0.01
    assert not report.safe
    assert [c.capability for c in report.collapsed] == ["arithmetic"]


def test_an_even_variant_is_safe():
    report = regression_report(
        "int8", "bf16", [CapabilityResult("general", 0.80, 0.79, samples=100)]
    )
    assert report.safe
    assert "SAFE" in report.summary()


def test_the_threshold_is_relative_not_absolute():
    """A capability scoring 0.9 and one scoring 0.3 get the same proportional test."""
    low = regression_report("q", "b", [CapabilityResult("rare", 0.30, 0.24, samples=50)])
    assert not low.safe


def test_a_capability_that_improved_is_never_a_regression():
    report = regression_report("q", "b", [CapabilityResult("c", 0.5, 0.9, samples=10)])
    assert report.safe
    assert report.capabilities[0].delta > 0


def test_a_zero_baseline_does_not_divide_by_zero():
    assert regression_report("q", "b", [CapabilityResult("c", 0.0, 0.0)]).safe


def test_the_aggregate_is_weighted_by_sample_count():
    """A 200-sample capability should not move the aggregate like a 10,000-sample one."""
    report = regression_report(
        "q",
        "b",
        [
            CapabilityResult("big", 0.8, 0.8, samples=10_000),
            CapabilityResult("small", 0.8, 0.0, samples=1),
        ],
    )
    assert report.aggregate_delta > -0.01


def test_an_empty_comparison_is_not_a_regression():
    report = regression_report("q", "b", [])
    assert report.safe
    assert report.aggregate_delta == 0.0


def test_the_summary_names_the_collapsed_capability():
    report = regression_report("int4", "bf16", [CapabilityResult("arithmetic", 0.7, 0.4, 200)])
    summary = report.summary()
    assert "arithmetic" in summary
    assert "REGRESSED" in summary
