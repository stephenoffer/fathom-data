"""Findings becoming incidents.

The load-bearing behaviour is correlation: one upstream breakage produces a finding
on every downstream dataset, and fourteen tickets against fourteen teams is how an
incident goes unowned. Grouping needs both lineage *and* time — either alone merges
things that have nothing to do with each other.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.types import DatasetId
from fathom.observe.incidents import (
    Finding,
    Incident,
    IncidentState,
    Severity,
    acknowledge,
    correlate,
    impact,
    mean_time_to_detect,
    mean_time_to_resolve,
    open_incidents,
    postmortem,
    recurring,
    resolve,
    root_datasets,
    severity_of,
    summarize_incidents,
    time_to_detect,
    time_to_resolve,
    timeline,
)

RAW = DatasetId("warehouse", "raw.events")
MID = DatasetId("warehouse", "silver.sessions")
GOLD = DatasetId("warehouse", "gold.daily")
OTHER = DatasetId("warehouse", "unrelated.table")

UPSTREAM = {MID: frozenset({RAW}), GOLD: frozenset({RAW, MID})}
DOWNSTREAM = {RAW: frozenset({MID, GOLD}), MID: frozenset({GOLD})}

BROKE = datetime(2026, 7, 29, 2, 0, tzinfo=UTC)
FOUND = datetime(2026, 7, 29, 9, 0, tzinfo=UTC)


def finding(
    dataset: DatasetId = RAW,
    *,
    check: str = "freshness",
    severity: Severity = Severity.MAJOR,
    detected: datetime = FOUND,
    started: datetime | None = BROKE,
) -> Finding:
    return Finding(check, dataset, "stale", severity, detected=detected, started=started)


# -- ordering ------------------------------------------------------------------


def test_severity_ranks_by_position_not_alphabet():
    """A plain max() over a StrEnum puts "minor" above "critical"."""
    assert Severity.CRITICAL.rank > Severity.MAJOR.rank > Severity.MINOR.rank


def test_severity_of_a_batch_is_the_worst_one():
    assert severity_of([finding(severity=Severity.MINOR), finding(severity=Severity.CRITICAL)]) is (
        Severity.CRITICAL
    )


def test_severity_of_nothing_is_info():
    assert severity_of([]) is Severity.INFO


def test_an_incident_takes_its_severity_from_its_worst_finding():
    incident = Incident("I", [finding(severity=Severity.MINOR), finding(severity=Severity.MAJOR)])
    assert incident.severity is Severity.MAJOR


def test_an_empty_incident_is_info():
    assert Incident("I").severity is Severity.INFO


def test_resolved_and_wont_fix_are_terminal():
    assert IncidentState.RESOLVED.is_terminal
    assert IncidentState.WONT_FIX.is_terminal
    assert not IncidentState.ACKNOWLEDGED.is_terminal


# -- correlation ---------------------------------------------------------------


def test_one_breakage_across_a_lineage_chain_is_one_incident():
    """Fourteen tickets against fourteen teams is how an incident goes unowned."""
    incidents = correlate([finding(RAW), finding(MID), finding(GOLD)], UPSTREAM)
    assert len(incidents) == 1
    assert set(incidents[0].datasets) == {RAW, MID, GOLD}


def test_unrelated_datasets_stay_separate():
    assert len(correlate([finding(RAW), finding(OTHER)], UPSTREAM)) == 2


def test_lineage_alone_does_not_merge_things_a_month_apart():
    """Otherwise a genuine breach merges with unrelated drift on the same table."""
    later = finding(GOLD, detected=FOUND + timedelta(days=30), started=None)
    assert len(correlate([finding(RAW), later], UPSTREAM)) == 2


def test_the_correlation_window_is_configurable():
    later = finding(GOLD, detected=FOUND + timedelta(hours=3), started=None)
    assert len(correlate([finding(RAW), later], UPSTREAM, window=timedelta(hours=6))) == 1


def test_findings_without_times_group_rather_than_split():
    """Grouping is the safer error: a split incident has no owner at all."""
    a = finding(RAW, detected=None, started=None)
    b = finding(GOLD, detected=None, started=None)
    assert len(correlate([a, b], UPSTREAM)) == 1


def test_two_findings_on_the_same_dataset_are_one_incident():
    both = [finding(RAW, check="freshness"), finding(RAW, check="schema")]
    assert len(correlate(both, UPSTREAM)) == 1


def test_correlating_nothing_produces_nothing():
    assert correlate([], UPSTREAM) == []


def test_an_incident_records_where_it_originates():
    incident = correlate([finding(RAW), finding(GOLD)], UPSTREAM)[0]
    assert "raw.events" in incident.cause


# -- roots and blast radius ----------------------------------------------------


def test_the_root_is_the_affected_dataset_with_nothing_affected_above_it():
    assert root_datasets([finding(RAW), finding(MID), finding(GOLD)], UPSTREAM) == {RAW}


def test_every_dataset_is_a_root_when_none_are_connected():
    assert root_datasets([finding(RAW), finding(OTHER)], UPSTREAM) == {RAW, OTHER}


def test_impact_comes_from_the_graph_not_from_the_findings():
    """The datasets that matter are the ones nobody has checked yet."""
    incident = Incident("I", [finding(RAW)])
    assert set(impact(incident, DOWNSTREAM)) == {RAW, MID, GOLD}


def test_impact_of_a_leaf_is_just_itself():
    assert impact(Incident("I", [finding(GOLD)]), DOWNSTREAM) == (GOLD,)


# -- lifecycle -----------------------------------------------------------------


def test_acknowledging_records_an_owner():
    incident = acknowledge(Incident("I", [finding()]), "data-platform", at=FOUND)
    assert incident.owner == "data-platform"
    assert incident.state is IncidentState.ACKNOWLEDGED


def test_acknowledging_without_an_owner_is_refused():
    """Somebody seeing it is not the same as somebody fixing it."""
    with pytest.raises(ValueError, match="named owner"):
        acknowledge(Incident("I"), "")


def test_resolving_records_the_cause_and_time():
    incident = resolve(Incident("I", opened=FOUND), cause="upstream job failed", at=FOUND)
    assert incident.state is IncidentState.RESOLVED
    assert incident.cause == "upstream job failed"
    assert not incident.is_open


def test_wont_fix_is_a_valid_ending():
    assert resolve(Incident("I"), state=IncidentState.WONT_FIX).state is IncidentState.WONT_FIX


def test_resolving_into_a_non_terminal_state_is_refused():
    with pytest.raises(ValueError, match="not a terminal state"):
        resolve(Incident("I"), state=IncidentState.OPEN)


# -- durations -----------------------------------------------------------------


def test_time_to_detect_is_the_gap_between_breaking_and_noticing():
    """A team with fast resolution and slow detection has customers doing its
    monitoring."""
    assert time_to_detect(Incident("I", [finding()])) == timedelta(hours=7)


def test_time_to_detect_is_unknown_without_a_start_time():
    assert time_to_detect(Incident("I", [finding(started=None)])) is None


def test_time_to_detect_takes_the_worst_finding():
    early = finding(RAW, started=BROKE - timedelta(hours=5))
    assert time_to_detect(Incident("I", [finding(), early])) == timedelta(hours=12)


def test_time_to_resolve_needs_both_ends():
    assert time_to_resolve(Incident("I", opened=FOUND)) is None
    assert time_to_resolve(
        Incident("I", opened=FOUND, resolved=FOUND + timedelta(hours=2))
    ) == timedelta(hours=2)


def test_mean_durations_ignore_incidents_with_missing_times():
    incidents = [
        Incident("A", [finding()], opened=FOUND, resolved=FOUND + timedelta(hours=2)),
        Incident("B", [finding(started=None)]),
    ]
    assert mean_time_to_detect(incidents) == timedelta(hours=7)
    assert mean_time_to_resolve(incidents) == timedelta(hours=2)


def test_means_of_nothing_are_none():
    assert mean_time_to_detect([]) is None
    assert mean_time_to_resolve([]) is None


# -- queries -------------------------------------------------------------------


def test_open_incidents_come_worst_first():
    minor = Incident("A", [finding(severity=Severity.MINOR)], opened=FOUND)
    critical = Incident("B", [finding(severity=Severity.CRITICAL)], opened=FOUND)
    assert [i.identifier for i in open_incidents([minor, critical])] == ["B", "A"]


def test_resolved_incidents_are_excluded():
    closed = resolve(Incident("A", [finding()]))
    assert open_incidents([closed]) == []


def test_recurring_finds_what_the_previous_fix_only_symptom_treated():
    incidents = [Incident("A", [finding(RAW)]), Incident("B", [finding(RAW)])]
    assert recurring(incidents) == {f"freshness:{RAW}": 2}


def test_a_one_off_is_not_recurring():
    assert recurring([Incident("A", [finding(RAW)])]) == {}


# -- postmortem ----------------------------------------------------------------


def test_the_timeline_is_ordered_and_separates_breaking_from_detecting():
    incident = acknowledge(Incident("I", [finding()]), "team", at=FOUND + timedelta(minutes=5))
    events = timeline(incident)
    assert [when for when, _ in events] == sorted(when for when, _ in events)
    assert any("broke" in what for _, what in events)
    assert any("detected" in what for _, what in events)


def test_a_postmortem_proposes_a_regression_check_per_failed_check():
    """An incident that produces prose and no check will happen again."""
    incident = Incident("INC-1", [finding(check="freshness"), finding(MID, check="schema")])
    report = postmortem(incident)
    assert len(report.regression_checks) == 2
    assert any("freshness" in c for c in report.regression_checks)


def test_a_postmortem_renders_the_checks_as_a_task_list():
    markdown = postmortem(Incident("INC-1", [finding()])).to_markdown()
    assert "- [ ]" in markdown
    assert "## Timeline" in markdown


def test_a_postmortem_with_no_findings_says_no_check_was_proposed():
    markdown = postmortem(Incident("INC-1")).to_markdown()
    assert "will happen again" in markdown


def test_the_summary_reports_counts_and_the_worst_severity():
    incidents = correlate([finding(RAW, severity=Severity.CRITICAL)], UPSTREAM)
    summary = summarize_incidents(incidents)
    assert "1 incident(s)" in summary
    assert "worst is critical" in summary


def test_summarizing_nothing_says_so():
    assert summarize_incidents([]) == "no incidents"
