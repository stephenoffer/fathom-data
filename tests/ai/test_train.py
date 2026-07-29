"""Runs, checkpoints, and model derivation.

The tests that matter here are the ones about *refusing to answer*: a scaling law
that will not fit two points, a resume check that distinguishes "reshard it" from
"retrain it", and a comparison that says outright when a metric moved for no
recorded reason.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from fathom.ai.train import (
    Adaptation,
    AdaptationKind,
    Checkpoint,
    Derivation,
    Distillation,
    Merge,
    Parallelism,
    PreferencePair,
    PreferenceSet,
    Quantization,
    QuantizationFormat,
    Run,
    RunStatus,
    Shard,
    ShardingScheme,
    Sweep,
    Trial,
    adapter_edges,
    annotator_diversity,
    base_of,
    best_trial,
    can_resume,
    checkpoint_size,
    chinchilla_optimal,
    compare,
    compatible_topologies,
    compute_flops,
    describe_topology,
    determinism_report,
    diff_hyperparameters,
    distillation_edges,
    effective_restrictions,
    fit_scaling_law,
    gpu_hours,
    inter_annotator_agreement,
    is_derived_from,
    is_regression,
    lineage_depth,
    loss_curve_summary,
    merge_edges,
    missing_shards,
    optimizer_overhead,
    parallelism_from,
    pareto_front,
    predict_tokens_needed,
    preference_edges,
    quantization_edges,
    rank_stability,
    retention_plan,
    run_edges,
    sweep_summary,
    throughput,
    trainable_fraction,
    validate,
    validate_merge,
)
from fathom.ai.train.experiments import ablation


def make_run(name: str, **overrides) -> Run:
    base = {
        "status": RunStatus.COMPLETED,
        "hyperparameters": {"lr": 1e-3, "batch": 32},
        "metrics": {"loss": 2.5},
        "seed": 7,
        "code_version": "abc123",
        "started": datetime(2026, 1, 1),
        "finished": datetime(2026, 1, 2),
        "accelerator_count": 8,
        "tokens": 10**9,
    }
    base.update(overrides)
    return Run(name=name, **base)


# -- runs ----------------------------------------------------------------------


def test_only_completed_runs_are_comparable():
    """A partial run's metrics are not comparable to a finished one's."""
    assert RunStatus.COMPLETED.is_comparable
    assert not RunStatus.RUNNING.is_comparable
    assert not RunStatus.PREEMPTED.is_comparable


def test_preemption_is_not_failure():
    """A spot reclaim is resumable; a failure is not. Conflating them wastes a run."""
    assert RunStatus.FAILED.is_terminal
    assert not RunStatus.PREEMPTED.is_terminal


def test_accelerator_hours_and_throughput():
    run = make_run("a")
    assert gpu_hours(run) == pytest.approx(192.0)
    # Throughput is per accelerator-second, so it compares across cluster sizes.
    assert throughput(run) == pytest.approx(10**9 / (24 * 3600 * 8))


def test_throughput_of_a_run_with_no_duration_is_zero_not_infinite():
    assert throughput(Run(name="x", tokens=10**9)) == 0.0


def test_flops_uses_the_six_nd_estimate():
    assert compute_flops(make_run("a"), parameters=10**9) == pytest.approx(6 * 10**18)


def test_comparison_separates_cause_from_effect():
    before = make_run("a", metrics={"loss": 2.5})
    after = make_run("b", metrics={"loss": 2.1}, hyperparameters={"lr": 3e-4, "batch": 32})
    result = compare(before, after)

    assert result.metric_deltas["loss"] == pytest.approx(-0.4)
    assert not result.hyperparameters.is_empty
    assert not result.unexplained


def test_a_metric_that_moved_for_no_reason_is_called_out():
    """Identical config, data, code, and seed, different result: something is untracked."""
    before = make_run("a", metrics={"loss": 2.5})
    after = make_run("b", metrics={"loss": 2.1})
    result = compare(before, after)

    assert result.unexplained
    assert "UNEXPLAINED" in result.summary()


def test_identical_runs_are_not_flagged():
    assert not compare(make_run("a"), make_run("b")).unexplained


def test_hyperparameter_diff_reports_all_three_kinds():
    before = make_run("a", hyperparameters={"lr": 1e-3, "dropped": 1})
    after = make_run("b", hyperparameters={"lr": 3e-4, "added": 2})
    diff = diff_hyperparameters(before, after)

    assert diff.added == ("added",)
    assert diff.removed == ("dropped",)
    assert diff.changed == (("lr", 1e-3, 3e-4),)


def test_regression_respects_direction_and_tolerance():
    before = make_run("a", metrics={"loss": 2.0})
    after = make_run("b", metrics={"loss": 2.05})
    assert is_regression(before, after)
    assert not is_regression(before, after, tolerance=0.1)
    assert not is_regression(before, after, minimize=False)


def test_run_edges_route_through_the_run():
    """An edge cannot hold a learning rate, which is why the run is a node."""
    from fathom.core.types import DatasetId

    source, produced = DatasetId("s3://lake", "corpus"), DatasetId("model://local", "m")
    run = make_run("r", inputs=(source,), outputs=(produced,))
    edges = run_edges(run)

    assert (source, run.dataset) in edges
    assert (run.dataset, produced) in edges


# -- sweeps --------------------------------------------------------------------


def sweep_of(*values: float) -> Sweep:
    return Sweep(
        name="s",
        trials=[
            Trial(run=make_run(f"t{i}", metrics={"loss": v, "latency": 100 - v}), varied={"lr": v})
            for i, v in enumerate(values)
        ],
    )


def test_best_trial_respects_the_objective_direction():
    sweep = sweep_of(2.5, 2.1, 2.9)
    assert best_trial(sweep).name == "t1"
    sweep.minimize = False
    assert best_trial(sweep).name == "t2"


def test_a_sweep_with_no_completed_trials_has_no_best():
    sweep = Sweep(name="s", trials=[Trial(run=Run(name="t", status=RunStatus.RUNNING))])
    assert best_trial(sweep) is None


def test_pareto_front_keeps_the_tradeoff_visible():
    """Single-objective selection hides the model that is marginally better and
    three times more expensive."""
    sweep = sweep_of(2.5, 2.1, 2.9)
    front = pareto_front(sweep, objectives=["loss", "latency"])
    assert front  # every point trades loss against latency, so none dominates
    assert len(front) == 3


def test_sweep_summary_counts_and_spread():
    summary = sweep_summary(sweep_of(2.5, 2.1, 2.9))
    assert summary["completed"] == 3
    assert summary["best"] == "t1"
    assert summary["spread"] == pytest.approx(0.8)


def test_an_uncontrolled_ablation_says_so():
    """An ablation measuring two changes attributes the sum to one of them."""
    baseline = make_run("base", hyperparameters={"lr": 1e-3, "layers": 12})
    variant = make_run("var", hyperparameters={"lr": 3e-4, "layers": 24})
    assert not ablation("two-things", baseline, variant, changed="layers").is_controlled

    single = make_run("var", hyperparameters={"lr": 1e-3, "layers": 24})
    assert ablation("one-thing", baseline, single, changed="layers").is_controlled


# -- determinism ---------------------------------------------------------------


def test_determinism_names_the_specific_source():
    """A boolean sends people looking; a named source sends them to the line."""
    report = determinism_report(Run(name="r", hyperparameters={"tf32": True}))
    assert not report.reproducible
    names = {name for name, _ in report.sources}
    assert {"seed", "code_version", "tf32"} <= names


def test_a_fully_pinned_run_is_reproducible():
    assert determinism_report(make_run("r")).reproducible


# -- scaling laws --------------------------------------------------------------


def test_a_scaling_law_needs_more_than_one_point():
    with pytest.raises(ValueError, match="at least two points"):
        fit_scaling_law([(1e18, 3.0)])


def test_fit_recovers_a_known_power_law():
    law = fit_scaling_law([(1e18, 3.0), (1e19, 2.6), (1e20, 2.3)], irreducible=1.5)
    assert law.points == 3
    assert law.exponent > 0
    assert law.predict(1e20) == pytest.approx(2.3, abs=0.05)


def test_extrapolation_is_reported_not_hidden():
    """Fitting three small runs and quoting two orders of magnitude out burns budgets."""
    law = fit_scaling_law([(1e18, 3.0), (1e19, 2.6), (1e20, 2.3)], irreducible=1.5)
    assert not law.is_extrapolating(1e20)
    assert law.is_extrapolating(1e23)
    assert law.extrapolation_factor(1e21) == pytest.approx(10.0)


def test_a_target_below_the_irreducible_loss_is_unreachable():
    """Infinity is the honest answer; an enormous finite number is not."""
    law = fit_scaling_law([(1e18, 3.0), (1e19, 2.6)], irreducible=1.5)
    assert predict_tokens_needed(law, 1.4) == float("inf")
    assert predict_tokens_needed(law, 2.0) < float("inf")


def test_chinchilla_ratio_is_a_parameter_not_a_constant():
    """It is an empirical result that has moved and will move again."""
    default = chinchilla_optimal(1e22)
    other = chinchilla_optimal(1e22, tokens_per_parameter=40)
    assert other["parameters"] < default["parameters"]
    assert default["tokens"] / default["parameters"] == pytest.approx(20.0)


def test_loss_curve_windows_cannot_overlap():
    """A short series compared to itself reports zero improvement on a curve that
    plainly improved, which reads as a plateau and stops a run that should continue."""
    summary = loss_curve_summary([3.0, 2.8, 2.6, 2.55, 2.54, 2.54])
    assert summary["improvement"] > 0


def test_loss_curve_detects_spikes():
    assert loss_curve_summary([3.0, 2.8, 9.0, 2.7])["spikes"] == 1.0


# -- checkpoints ---------------------------------------------------------------


def sharded(**overrides) -> Checkpoint:
    parallelism = overrides.pop("parallelism", Parallelism(data=4, tensor=2))
    base = {
        "step": 1000,
        "parallelism": parallelism,
        "scheme": ShardingScheme.SHARDED,
        "shards": tuple(
            Shard(path=f"s{i}", rank=i, bytes=10**9, contains_optimizer=i % 2 == 0, checksum="x")
            for i in range(parallelism.size)
        ),
        "framework": "torch",
        "framework_version": "2.4",
        "parameters": 7 * 10**9,
    }
    base.update(overrides)
    return Checkpoint(name="ck", **base)


def test_parallelism_must_multiply_out():
    with pytest.raises(ValueError, match="at least 1"):
        Parallelism(data=0)


def test_world_size_is_derived_not_stored():
    assert Parallelism(data=4, tensor=2, pipeline=2).size == 16


def test_topology_is_described_readably():
    assert "dp=4" in describe_topology(sharded())


def test_parallelism_accepts_the_usual_config_spellings():
    assert parallelism_from({"tp": 4, "pipeline_parallel_size": 2}).size == 8


def test_a_checkpoint_missing_a_rank_is_unloadable():
    """Not slightly damaged — unloadable, and finding out at resume costs the queue wait."""
    partial = sharded(shards=(Shard(path="s0", rank=0),))
    assert missing_shards(partial)
    assert not can_resume(partial, Parallelism(data=4, tensor=2)).can_resume


def test_resuming_into_the_same_topology_just_works():
    verdict = can_resume(sharded(), Parallelism(data=4, tensor=2))
    assert verdict.can_resume
    assert not verdict.resharding_required


def test_resharding_is_a_third_outcome_not_a_failure():
    """Reporting it as blocked sends people to restart a run they could resume."""
    verdict = can_resume(sharded(), Parallelism(data=2, tensor=2), framework="torch")
    assert verdict.can_resume
    assert verdict.resharding_required


def test_per_rank_checkpoints_cannot_change_world_size():
    per_rank = Checkpoint(
        name="ck",
        parallelism=Parallelism(data=8),
        scheme=ShardingScheme.PER_RANK,
        shards=tuple(Shard(path=f"r{i}", rank=i) for i in range(8)),
        framework="torch",
    )
    verdict = can_resume(per_rank, Parallelism(data=4))
    assert not verdict.can_resume
    assert any("consolidate" in b for b in verdict.blockers)


def test_a_framework_change_blocks_but_a_version_change_only_warns():
    blocked = can_resume(sharded(), Parallelism(data=4, tensor=2), framework="jax")
    assert not blocked.can_resume

    warned = can_resume(
        sharded(), Parallelism(data=4, tensor=2), framework="torch", framework_version="2.5"
    )
    assert warned.can_resume
    assert warned.warnings


def test_missing_optimizer_state_is_a_different_experiment():
    no_optimizer = sharded(
        shards=tuple(Shard(path=f"s{i}", rank=i) for i in range(8)),
    )
    verdict = can_resume(no_optimizer, Parallelism(data=4, tensor=2), framework="torch")
    assert any("fresh optimizer" in w for w in verdict.warnings)


def test_optimizer_overhead_is_measurable():
    assert optimizer_overhead(sharded()) == pytest.approx(0.5)


def test_compatible_topologies_answer_will_it_fit_on_this_cluster():
    found = compatible_topologies(sharded())
    assert found
    assert all(t.size == 8 for t in found)


def test_validation_catches_structural_problems_before_a_resume():
    problems = validate(
        Checkpoint(
            name="b",
            parallelism=Parallelism(data=4),
            scheme=ShardingScheme.PER_RANK,
            shards=(Shard(path="a", rank=0),),
        )
    )
    assert any("missing shards" in p for p in problems)
    assert any("checksums" in p for p in problems)


def test_checkpoint_size_sums_shards():
    assert checkpoint_size(sharded()) == 8 * 10**9


# -- retention -----------------------------------------------------------------


def test_retention_never_drops_a_checkpoint_with_descendants():
    """A checkpoint something was derived from is evidence, not storage. Deleting it
    makes a training-data question permanently unanswerable."""
    old = [
        Checkpoint(name=f"ck{i}", step=i * 100, metrics={"loss": 3.0 - i * 0.1}) for i in range(10)
    ]
    plan = retention_plan(old, keep_last=2, referenced=["ck0"])

    kept = dict(plan.keep)
    assert "ck0" in kept
    assert kept["ck0"].value == "has_descendants"
    assert "ck0" not in plan.drop


def test_retention_keeps_latest_and_best():
    old = [
        Checkpoint(name=f"ck{i}", step=i * 100, metrics={"loss": 3.0 - i * 0.1}) for i in range(6)
    ]
    kept = dict(retention_plan(old, keep_last=1).keep)
    assert "ck5" in kept  # latest and best coincide here
    assert len(kept) >= 1


def test_retention_of_nothing_is_empty_not_an_error():
    assert retention_plan([]).keep == ()


# -- fine-tuning ---------------------------------------------------------------


def test_an_adapter_has_two_parents():
    """Only one of them is yours, which is exactly why both are edges."""
    from fathom.core.types import DatasetId

    data = DatasetId("s3://lake", "tuning")
    a = Adaptation(name="lora", base="llama", training_data=(data,))
    edges = adapter_edges(a)
    assert len(edges) == 2
    assert any(src == data for src, _ in edges)


def test_trainable_fraction_shows_the_base_dominates():
    a = Adaptation(
        name="lora", base="llama", trainable_parameters=40 * 10**6, base_parameters=70 * 10**9
    )
    assert trainable_fraction(a) < 0.001


def test_parameter_efficiency_is_a_property_of_the_kind():
    assert AdaptationKind.LORA.is_parameter_efficient
    assert not AdaptationKind.FULL.is_parameter_efficient


def test_merge_validation_catches_silent_scaling():
    problems = validate_merge(Merge(name="m", sources=("a", "b"), weights=(0.5, 0.3)))
    assert any("sum to" in p for p in problems)


def test_merge_validation_catches_a_repeated_source():
    problems = validate_merge(Merge(name="m", sources=("a", "a")))
    assert any("twice" in p for p in problems)


def test_a_clean_merge_has_no_problems():
    assert validate_merge(Merge(name="m", sources=("a", "b"), weights=(0.5, 0.5))) == []


def test_merge_edges_name_every_source():
    assert len(merge_edges(Merge(name="m", sources=("a", "b", "c")))) == 3


def test_distillation_carries_the_teachers_obligations():
    """A student inherits the teacher's data obligations without seeing that data."""
    edges = distillation_edges(Distillation(student="small", teacher="large"))
    assert any("large" in str(src) for src, _ in edges)


def test_quantization_calibration_data_is_still_training_data():
    from fathom.core.types import DatasetId

    calibration = DatasetId("s3://lake", "calib")
    edges = quantization_edges(Quantization(name="q", source="m", calibration_data=(calibration,)))
    assert any(src == calibration for src, _ in edges)


def test_quantization_bit_widths():
    assert Quantization(name="q", source="m", format=QuantizationFormat.INT4).bits == 4
    assert Quantization(name="q", source="m", format=QuantizationFormat.FP8).bits == 8


# -- preference data -----------------------------------------------------------


def preferences() -> PreferenceSet:
    return PreferenceSet(
        name="rlhf",
        pairs=(
            PreferencePair("p1", "c1", "r1", annotator="ann1"),
            PreferencePair("p1", "r1", "c1", annotator="ann2"),
            PreferencePair("p2", "c2", "r2", annotator="ann1"),
            PreferencePair("p3", "c3", "r3", annotator="ann1"),
        ),
        source_annotations=("batch-7",),
    )


def test_annotator_concentration_is_measurable():
    """A set where three people produced most pairs encodes three people's taste."""
    diversity = annotator_diversity(preferences())
    assert diversity["annotators"] == 2.0
    assert diversity["top_share"] == pytest.approx(0.75)


def test_contradictory_pairs_are_not_averaged_away():
    """Training on both sides of a contested prompt teaches nothing."""
    stability = rank_stability(preferences())
    assert stability["contradictory"] == 1.0
    assert stability["stability"] < 1.0


def test_agreement_over_shared_prompts():
    first = [PreferencePair("p1", "a", "b"), PreferencePair("p2", "a", "b")]
    second = [PreferencePair("p1", "a", "b"), PreferencePair("p2", "b", "a")]
    assert inter_annotator_agreement(first, second) == pytest.approx(0.5)


def test_agreement_with_no_shared_prompts_is_zero():
    assert inter_annotator_agreement([PreferencePair("p1", "a", "b")], []) == 0.0


def test_preferences_are_downstream_of_annotations():
    """Three edges from a model to a person; nothing finds that chain by accident."""
    edges = preference_edges(preferences())
    assert any("batch-7" in str(src) for src, _ in edges)


# -- derivation traversal ------------------------------------------------------


def chain() -> list[Derivation]:
    return [
        Derivation("llama-3", "lora-v3", "lora"),
        Derivation("lora-v3", "merged", "merge"),
        Derivation("merged", "merged-int4", "quantize"),
    ]


def test_lineage_depth_counts_derivations():
    assert lineage_depth(chain(), "merged-int4") == 3


def test_base_of_finds_the_root():
    assert base_of(chain(), "merged-int4") == ["llama-3"]


def test_ancestry_is_answerable():
    assert is_derived_from(chain(), "merged-int4", "llama-3")
    assert not is_derived_from(chain(), "merged-int4", "mistral")


def test_a_derivation_cycle_terminates():
    """Two models merged from each other is a config people actually write."""
    cyclic = [Derivation("a", "b", "merge"), Derivation("b", "a", "merge")]
    assert lineage_depth(cyclic, "a") >= 0  # terminates rather than recursing forever


def test_restrictions_combine_restrictively():
    """A permissive adapter over a research-only base is research-only. Teams get
    this backwards because the adapter is the part they wrote."""
    combined = effective_restrictions(
        {"base": {"no-commercial"}, "adapter": {"attribution"}}, ["base", "adapter"]
    )
    assert combined == {"no-commercial", "attribution"}
