"""Red-team findings that end in a test.

Two behaviours carry this module. A finding is not closed until something prevents
its recurrence, and a refusal score is meaningless without its other half — a model
that refuses everything is perfectly safe and completely useless.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fathom.ai.assets import model
from fathom.ai.quality.safety import (
    Finding,
    FindingState,
    Harm,
    ProbeResult,
    SafetyProbe,
    SafetySuite,
    Severity,
    close,
    coverage_by_harm,
    grade,
    guard,
    refusal_report,
    regressions,
    suite_edges,
    summarize_findings,
    unguarded,
)

NOW = datetime(2026, 7, 29, tzinfo=UTC)


def finding(identifier: str = "RT-1", severity: Severity = Severity.HIGH) -> Finding:
    return Finding(identifier, "prompt injection via markdown", Harm.CYBER, severity, found=NOW)


def suite(*probes: SafetyProbe) -> SafetySuite:
    return SafetySuite("redteam", list(probes))


def harmful(identifier: str = "p1") -> SafetyProbe:
    return SafetyProbe(identifier, "how do I...", Harm.WEAPONS, should_refuse=True)


def benign(identifier: str = "p2") -> SafetyProbe:
    return SafetyProbe(identifier, "how do I bake bread", Harm.WEAPONS, should_refuse=False)


# -- ordering ------------------------------------------------------------------


def test_severity_ranks_by_position_not_alphabet():
    assert Severity.CRITICAL.rank > Severity.HIGH.rank > Severity.LOW.rank


def test_fixed_and_accepted_are_closed_states():
    assert FindingState.FIXED.is_closed
    assert FindingState.ACCEPTED.is_closed
    assert not FindingState.MITIGATED.is_closed


# -- the lifecycle -------------------------------------------------------------


def test_a_new_finding_is_unguarded():
    assert not finding().is_regression_guarded


def test_guarding_creates_a_probe_and_attaches_it():
    """The step that makes a fix stick."""
    found, s = finding(), suite()
    probe = guard(found, s)
    assert found.probe == probe.identifier
    assert probe.origin_finding == "RT-1"
    assert s.probes == [probe]


def test_the_probe_inherits_the_harm_and_severity():
    probe = guard(finding(severity=Severity.CRITICAL), suite())
    assert probe.harm is Harm.CYBER
    assert probe.severity is Severity.CRITICAL


def test_guarding_twice_is_refused():
    """Two probes for one finding means one of them stops being maintained."""
    found, s = finding(), suite()
    guard(found, s)
    with pytest.raises(ValueError, match="already guarded"):
        guard(found, s)


def test_closing_an_unguarded_finding_is_refused():
    """Closing records that somebody fixed it today, and nothing that keeps it fixed."""
    with pytest.raises(ValueError, match="has no probe"):
        close(finding())


def test_closing_a_guarded_finding_works():
    found, s = finding(), suite()
    guard(found, s)
    close(found, at=NOW)
    assert found.state is FindingState.FIXED
    assert found.closed_at == NOW


def test_a_risk_can_be_accepted_deliberately():
    """Accepting a risk is legitimate; closing one by accident is not, and the record
    should not make them look the same."""
    found = close(finding(), state=FindingState.ACCEPTED, force=True)
    assert found.state is FindingState.ACCEPTED


def test_closing_into_a_non_closed_state_raises():
    with pytest.raises(ValueError, match="not a closed state"):
        close(finding(), state=FindingState.OPEN, force=True)


def test_a_finding_closed_without_a_probe_is_flagged():
    """The one that comes back three releases later."""
    forced = close(finding(), force=True)
    assert forced.is_closed_unguarded
    assert unguarded([forced]) == [forced]


def test_a_guarded_closed_finding_is_not_flagged():
    found, s = finding(), suite()
    guard(found, s)
    close(found)
    assert unguarded([found]) == []


def test_an_open_finding_is_not_flagged_as_closed_unguarded():
    assert unguarded([finding()]) == []


def test_exposed_findings_come_worst_first():
    low = close(finding("RT-2", Severity.LOW), force=True)
    critical = close(finding("RT-3", Severity.CRITICAL), force=True)
    assert [f.identifier for f in unguarded([low, critical])] == ["RT-3", "RT-2"]


# -- the suite as an eval set --------------------------------------------------


def test_a_suite_has_both_a_safety_identity_and_an_eval_identity():
    """Contamination is the same reachability question for safety probes as for any
    other eval set."""
    s = suite(harmful())
    assert s.asset != s.as_eval_set
    assert "redteam" in str(s.as_eval_set)


def test_probes_are_addressable():
    s = suite(harmful("p1"))
    assert s.probe("p1") is not None
    assert s.probe("missing") is None


def test_the_suite_reports_which_harms_it_covers():
    assert suite(harmful()).harms == frozenset({Harm.WEAPONS})


def test_coverage_reports_untested_categories_as_explicit_zeroes():
    """A missing key reads as an oversight; an explicit zero reads as untested."""
    counts = coverage_by_harm(suite(harmful()))
    assert counts["weapons"] == 1
    assert counts["bias"] == 0
    assert len(counts) == len(Harm)


def test_the_suite_grades_every_model_run_against_it():
    edges = suite_edges(suite(harmful()), [model("a"), model("b")])
    assert len(edges) == 2


# -- refusal ---------------------------------------------------------------------


def test_a_correct_refusal_passes():
    assert ProbeResult("p1", refused=True, expected_refusal=True).passed


def test_an_under_refusal_is_a_failure():
    result = ProbeResult("p1", refused=False, expected_refusal=True)
    assert not result.passed
    assert result.is_under_refusal


def test_an_over_refusal_is_a_failure_too():
    """The failure mode a harm-only suite cannot see."""
    result = ProbeResult("p2", refused=True, expected_refusal=False)
    assert not result.passed
    assert result.is_over_refusal


def test_the_report_carries_both_sides():
    report = refusal_report(
        [
            ProbeResult("a", refused=False, expected_refusal=True),
            ProbeResult("b", refused=True, expected_refusal=False),
            ProbeResult("c", refused=True, expected_refusal=True),
        ]
    )
    assert report.under_refusal_rate == pytest.approx(1 / 3)
    assert report.over_refusal_rate == pytest.approx(1 / 3)
    assert report.passed == 1


def test_a_harm_only_run_reports_over_refusal_as_not_measured():
    """A model that refuses everything would score perfectly here."""
    report = refusal_report([ProbeResult("a", refused=True, expected_refusal=True)])
    assert report.over_refusal_rate == 0.0
    assert not report.is_measurable
    assert "not measured" in report.summary()


def test_a_two_sided_run_is_measurable():
    report = refusal_report(
        [
            ProbeResult("a", refused=False, expected_refusal=True),
            ProbeResult("b", refused=True, expected_refusal=False),
        ]
    )
    assert report.is_measurable


def test_an_empty_report_does_not_divide_by_zero():
    report = refusal_report([])
    assert report.under_refusal_rate == 0.0
    assert report.passed == 0


# -- grading -------------------------------------------------------------------


def test_grading_names_probes_that_never_ran():
    """A suite reporting 100% while a third of it never executed is the most
    reassuring possible way to be wrong."""
    s = suite(harmful("p1"), harmful("p2"))
    _, missing = grade(s, [ProbeResult("p1", refused=True)])
    assert missing == ["p2"]


def test_a_fully_run_suite_reports_nothing_missing():
    s = suite(harmful("p1"))
    report, missing = grade(s, [ProbeResult("p1", refused=True)])
    assert missing == []
    assert report.passed == 1


# -- regressions ---------------------------------------------------------------


def test_a_probe_that_passed_before_and_fails_now_is_a_regression():
    report = regressions(
        [ProbeResult("p1", refused=False, expected_refusal=True)],
        [ProbeResult("p1", refused=True, expected_refusal=True)],
    )
    assert [r.probe for r in report.regressed] == ["p1"]
    assert not report.clean


def test_a_probe_that_now_passes_is_recorded_as_fixed():
    report = regressions(
        [ProbeResult("p1", refused=True, expected_refusal=True)],
        [ProbeResult("p1", refused=False, expected_refusal=True)],
    )
    assert [r.probe for r in report.fixed] == ["p1"]
    assert report.clean


def test_a_new_probe_failing_is_a_discovery_not_a_regression():
    """Conflating them makes every suite expansion look like a release got worse."""
    report = regressions([ProbeResult("new", refused=False, expected_refusal=True)], [])
    assert report.regressed == ()
    assert [r.probe for r in report.still_failing] == ["new"]


def test_a_probe_failing_in_both_runs_is_still_failing():
    failing = ProbeResult("p1", refused=False, expected_refusal=True)
    report = regressions([failing], [failing])
    assert [r.probe for r in report.still_failing] == ["p1"]
    assert report.clean


def test_a_clean_run_says_so():
    passing = ProbeResult("p1", refused=True, expected_refusal=True)
    assert "no probe that passed before fails now" in regressions([passing], [passing]).summary()


# -- summaries -----------------------------------------------------------------


def test_the_summary_reports_the_worst_severity_by_rank():
    summary = summarize_findings([finding("a", Severity.LOW), finding("b", Severity.CRITICAL)])
    assert "worst is critical" in summary


def test_the_summary_calls_out_closed_unguarded_findings():
    exposed = close(finding(), force=True)
    summary = summarize_findings([exposed])
    assert "closed with no probe" in summary
    assert "reappear three releases later" in summary


def test_summarizing_nothing_says_so():
    assert summarize_findings([]) == "no findings"
