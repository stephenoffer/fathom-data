"""Notification routing.

The behaviour worth testing is the suppression, not the sending. A notifier that
pages on everything gets muted inside a week, and a muted channel is worse than no
channel because everyone believes it is working.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from fathom.report.notify import (
    Channel,
    Notification,
    QuietHours,
    Route,
    Router,
    Severity,
    Suppression,
    dedupe,
    escalate,
    fingerprint,
    format_email,
    format_pagerduty,
    format_slack,
    format_teams,
    format_webhook,
    route,
    should_notify,
    summarize_batch,
)

DAY = datetime(2026, 7, 29, 10, 0, tzinfo=UTC)
NIGHT = datetime(2026, 7, 29, 3, 0, tzinfo=UTC)


def make_router() -> Router:
    router = Router(quiet_hours=QuietHours(), dedupe_window=timedelta(hours=1))
    router.add(Route(Channel.SLACK, "#data", min_severity=Severity.WARNING))
    router.add(Route(Channel.PAGERDUTY, "key", min_severity=Severity.CRITICAL))
    return router


def warning(at: datetime = DAY, target: str = "gold.monthly") -> Notification:
    return Notification(
        "Freshness breach", severity=Severity.WARNING, source="freshness", target=target, at=at
    )


def critical(at: datetime = NIGHT) -> Notification:
    return Notification(
        "Erasure incomplete", severity=Severity.CRITICAL, source="erase", target="raw.users", at=at
    )


# -- routing -------------------------------------------------------------------


def test_a_matching_notification_routes():
    assert route(make_router(), warning(), now=DAY) == [(Channel.SLACK, "#data")]


def test_severity_selects_which_channels_fire():
    """A critical reaches both; a warning reaches only the low-urgency one."""
    router = make_router()
    channels = {c for c, _ in route(router, critical(at=DAY), now=DAY)}
    assert channels == {Channel.SLACK, Channel.PAGERDUTY}


def test_below_threshold_is_suppressed_with_a_reason():
    """A notification that vanishes with no explanation is indistinguishable from a
    check that did not run."""
    info = Notification("New column", severity=Severity.INFO, source="check")
    allowed, reason = should_notify(make_router(), info, now=DAY)
    assert not allowed
    assert reason is Suppression.BELOW_THRESHOLD


def test_no_matching_route_is_distinguished_from_below_threshold():
    router = Router()
    router.add(Route(Channel.SLACK, "#a", min_severity=Severity.INFO, sources=frozenset({"other"})))
    allowed, reason = should_notify(router, warning(), now=DAY)
    assert not allowed
    assert reason is Suppression.NO_ROUTE


def test_routes_can_filter_by_source_and_target():
    router = Router()
    router.add(
        Route(Channel.SLACK, "#a", min_severity=Severity.INFO, targets=frozenset({"gold.monthly"}))
    )
    assert route(router, warning(), now=DAY)
    assert not route(router, warning(target="other"), now=DAY)


# -- deduplication -------------------------------------------------------------


def test_the_same_finding_recurring_is_suppressed():
    """Ninety-six messages for one problem is how a channel gets muted."""
    router = make_router()
    assert route(router, warning(), now=DAY)
    allowed, reason = should_notify(router, warning(), now=DAY + timedelta(minutes=15))
    assert not allowed
    assert reason is Suppression.DUPLICATE


def test_it_sends_again_once_the_window_passes():
    router = make_router()
    route(router, warning(), now=DAY)
    assert route(router, warning(), now=DAY + timedelta(hours=2))


def test_the_fingerprint_ignores_time_and_body():
    """Including the timestamp would defeat deduplication entirely."""
    early = Notification("t", body="one", severity=Severity.WARNING, source="s", target="d", at=DAY)
    later = Notification(
        "t",
        body="two",
        severity=Severity.WARNING,
        source="s",
        target="d",
        at=DAY + timedelta(hours=5),
    )
    assert fingerprint(early) == fingerprint(later)


def test_different_targets_are_different_problems():
    assert fingerprint(warning(target="a")) != fingerprint(warning(target="b"))


def test_dedupe_keeps_the_first_and_counts_the_rest():
    """The count is what makes a collapsed notification more useful than one alert."""
    collapsed = dedupe([warning(), warning(), critical()])
    assert len(collapsed) == 2
    assert collapsed[0][1] == 2


def test_dedupe_of_nothing_is_empty():
    assert dedupe([]) == []


# -- quiet hours ---------------------------------------------------------------


def test_quiet_hours_defer_what_can_wait():
    allowed, reason = should_notify(make_router(), warning(at=NIGHT), now=NIGHT)
    assert not allowed
    assert reason is Suppression.QUIET_HOURS


def test_quiet_hours_never_silence_a_critical():
    """Quiet hours that silence a page are quiet hours somebody disables after the
    first outage."""
    assert route(make_router(), critical(), now=NIGHT)


def test_quiet_hours_wrap_midnight():
    hours = QuietHours(start=time(22, 0), end=time(7, 0))
    assert hours.covers(datetime(2026, 7, 29, 23, 0, tzinfo=UTC))
    assert hours.covers(datetime(2026, 7, 29, 3, 0, tzinfo=UTC))
    assert not hours.covers(datetime(2026, 7, 29, 12, 0, tzinfo=UTC))


def test_quiet_hours_within_a_day_also_work():
    hours = QuietHours(start=time(1, 0), end=time(5, 0))
    assert hours.covers(datetime(2026, 7, 29, 3, 0, tzinfo=UTC))
    assert not hours.covers(datetime(2026, 7, 29, 23, 0, tzinfo=UTC))


# -- escalation ----------------------------------------------------------------


def test_escalation_is_on_age_not_repetition():
    """Something firing twice in a minute is noisy; something unacknowledged for an
    hour is genuinely worse than it was."""
    assert escalate(warning(), unacknowledged_for=timedelta(minutes=10)) is Severity.WARNING
    assert escalate(warning(), unacknowledged_for=timedelta(hours=1)) is Severity.ERROR
    assert escalate(warning(), unacknowledged_for=timedelta(hours=3)) is Severity.CRITICAL


def test_escalation_stops_at_the_top():
    assert escalate(critical(), unacknowledged_for=timedelta(days=7)) is Severity.CRITICAL


# -- summaries -----------------------------------------------------------------


def test_a_batch_summary_reports_the_worst_severity_by_rank():
    """Severity is a StrEnum, so a plain max() compares strings and decides
    "warning" outranks "critical", quietly deprioritising the finding that mattered."""
    summary = summarize_batch([warning(), critical(at=DAY)])
    assert "worst is critical" in summary


def test_a_batch_summary_collapses_repeats():
    summary = summarize_batch([warning(), warning(), warning()])
    assert "×3" in summary
    assert "1 distinct finding" in summary


def test_an_empty_batch_says_so():
    assert summarize_batch([]) == "nothing to report"


# -- transports ----------------------------------------------------------------


def test_pagerduty_dedup_key_matches_our_fingerprint():
    """So PagerDuty's own deduplication agrees with ours rather than opening a second
    incident for the same problem."""
    note = critical()
    assert format_pagerduty(note, routing_key="k")["dedup_key"] == note.fingerprint()


def test_pagerduty_carries_a_runbook_when_there_is_one():
    note = Notification("t", severity=Severity.CRITICAL, runbook="https://runbook")
    assert format_pagerduty(note, routing_key="k")["links"]


def test_pagerduty_omits_links_when_there_is_no_runbook():
    assert "links" not in format_pagerduty(critical(), routing_key="k")


def test_slack_colours_track_severity():
    assert (
        format_slack(critical())["attachments"][0]["color"]
        != (format_slack(warning())["attachments"][0]["color"])
    )


def test_slack_can_target_a_channel():
    assert format_slack(warning(), channel="#ops")["channel"] == "#ops"


def test_email_subject_carries_the_severity():
    assert format_email(critical(), to="a@b.c")["subject"].startswith("[CRITICAL]")


def test_teams_payload_is_a_message_card():
    assert format_teams(warning())["@type"] == "MessageCard"


def test_the_webhook_envelope_is_stable_json():
    import json

    body = json.loads(format_webhook(warning()))
    assert body["severity"] == "warning"
    assert body["fingerprint"] == warning().fingerprint()
