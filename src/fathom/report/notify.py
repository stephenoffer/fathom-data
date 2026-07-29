"""Getting a finding to a person, without becoming the reason people mute the channel.

Every check here exits non-zero and stops. That works in CI and nowhere else: a
freshness breach at 03:00 has nobody watching an exit code.

The hard part is not sending. It is not sending too much. A notifier that pages on
every drift finding gets muted inside a week, and a muted channel is worse than no
channel because everyone believes it is working. So the routing rules are the
substance here and the transports are deliberately thin:

- **Deduplication by fingerprint.** The same finding recurring every fifteen minutes
  is one notification with a count, not ninety-six.
- **Quiet hours that only defer what can wait.** A `CRITICAL` still pages at 03:00;
  an `INFO` waits until morning. Suppressing by severity rather than by clock is the
  difference between a policy people keep and one they disable.
- **Escalation on age, not on repetition.** Something unacknowledged for an hour is
  more serious than something that fired twice.

Transports are pure functions from a notification to a payload. Nothing here opens a
socket, which is what makes routing testable without a network and what lets a
caller use whatever HTTP client it already has.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from typing import Any

__all__ = [
    "Channel",
    "Notification",
    "QuietHours",
    "Route",
    "Router",
    "Severity",
    "Suppression",
    "dedupe",
    "escalate",
    "fingerprint",
    "format_email",
    "format_pagerduty",
    "format_slack",
    "format_teams",
    "format_webhook",
    "route",
    "should_notify",
    "summarize_batch",
]


class Severity(StrEnum):
    """Ordered. `CRITICAL` is what may wake someone."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)

    @property
    def may_wake_someone(self) -> bool:
        return self is Severity.CRITICAL


class Channel(StrEnum):
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    EMAIL = "email"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    LOG = "log"


@dataclass(frozen=True)
class Notification:
    """Something worth telling someone about."""

    title: str
    body: str = ""
    severity: Severity = Severity.INFO
    source: str = ""  # which check produced it
    target: str = ""  # which dataset it is about
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    detail: Mapping[str, Any] = field(default_factory=dict)
    runbook: str = ""

    def fingerprint(self) -> str:
        """What makes two notifications "the same thing recurring".

        Deliberately excludes the timestamp and the body. The same freshness breach
        at 03:00 and 03:15 is one problem, and including the time would defeat
        deduplication entirely.
        """
        payload = f"{self.source}\x1f{self.target}\x1f{self.title}\x1f{self.severity.value}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def fingerprint(notification: Notification) -> str:
    return notification.fingerprint()


@dataclass(frozen=True)
class QuietHours:
    """When to defer what can wait.

    `override_at` is the point of the whole thing: quiet hours that silence a
    critical page are quiet hours somebody disables after the first outage.
    """

    start: time = time(22, 0)
    end: time = time(7, 0)
    override_at: Severity = Severity.CRITICAL

    def covers(self, when: datetime) -> bool:
        moment = when.time()
        if self.start <= self.end:
            return self.start <= moment < self.end
        # Wrapping midnight, which is the usual case.
        return moment >= self.start or moment < self.end

    def suppresses(self, notification: Notification, *, when: datetime | None = None) -> bool:
        moment = when or notification.at
        if not self.covers(moment):
            return False
        return notification.severity.rank < self.override_at.rank


@dataclass(frozen=True)
class Route:
    """Where a notification goes, and what has to be true for it to."""

    channel: Channel
    destination: str = ""
    min_severity: Severity = Severity.WARNING
    sources: frozenset[str] = frozenset()  # empty means any
    targets: frozenset[str] = frozenset()  # empty means any

    def accepts(self, notification: Notification) -> bool:
        return (
            notification.severity.rank >= self.min_severity.rank
            and (not self.sources or notification.source in self.sources)
            and (not self.targets or notification.target in self.targets)
        )


class Suppression(StrEnum):
    """Why something was not sent. Recorded rather than silent."""

    DUPLICATE = "duplicate"
    QUIET_HOURS = "quiet_hours"
    BELOW_THRESHOLD = "below_threshold"
    NO_ROUTE = "no_route"


@dataclass
class Router:
    """Routing rules plus the state deduplication needs."""

    routes: list[Route] = field(default_factory=list)
    quiet_hours: QuietHours | None = None
    dedupe_window: timedelta = timedelta(hours=1)
    # fingerprint -> (first seen, last sent, count)
    _seen: dict[str, tuple[datetime, datetime, int]] = field(default_factory=dict, repr=False)

    def add(self, route: Route) -> None:
        self.routes.append(route)

    def reset(self) -> None:
        self._seen.clear()


def should_notify(
    router: Router, notification: Notification, *, now: datetime | None = None
) -> tuple[bool, Suppression | None]:
    """Whether to send, and if not, which rule stopped it.

    Returning the reason matters: a notification that vanishes with no explanation is
    indistinguishable from a check that did not run.
    """
    moment = now or notification.at

    if not any(r.accepts(notification) for r in router.routes):
        below = any(notification.severity.rank < r.min_severity.rank for r in router.routes)
        return False, (Suppression.BELOW_THRESHOLD if below else Suppression.NO_ROUTE)

    if router.quiet_hours is not None and router.quiet_hours.suppresses(notification, when=moment):
        return False, Suppression.QUIET_HOURS

    key = notification.fingerprint()
    previous = router._seen.get(key)
    if previous is not None:
        _first, last_sent, _count = previous
        if moment - last_sent < router.dedupe_window:
            return False, Suppression.DUPLICATE

    return True, None


def route(
    router: Router, notification: Notification, *, now: datetime | None = None
) -> list[tuple[Channel, str]]:
    """Decide where a notification goes, updating deduplication state.

    Returns an empty list when suppressed. Use `should_notify` when the reason
    matters.
    """
    moment = now or notification.at
    allowed, _reason = should_notify(router, notification, now=moment)
    if not allowed:
        key = notification.fingerprint()
        if key in router._seen:
            first, last, count = router._seen[key]
            router._seen[key] = (first, last, count + 1)
        return []

    key = notification.fingerprint()
    first = router._seen.get(key, (moment, moment, 0))[0]
    count = router._seen.get(key, (moment, moment, 0))[2]
    router._seen[key] = (first, moment, count + 1)

    return [(r.channel, r.destination) for r in router.routes if r.accepts(notification)]


def dedupe(notifications: Iterable[Notification]) -> list[tuple[Notification, int]]:
    """Collapse repeats, keeping the first and counting the rest.

    The count is what makes a collapsed notification more useful than a single one:
    "freshness breach, 96 occurrences" says something a lone alert does not.
    """
    order: list[str] = []
    grouped: dict[str, tuple[Notification, int]] = {}
    for notification in notifications:
        key = notification.fingerprint()
        if key not in grouped:
            grouped[key] = (notification, 1)
            order.append(key)
        else:
            first, count = grouped[key]
            grouped[key] = (first, count + 1)
    return [grouped[key] for key in order]


def escalate(
    notification: Notification,
    *,
    unacknowledged_for: timedelta,
    after: timedelta = timedelta(hours=1),
) -> Severity:
    """Raise severity for something nobody has picked up.

    On age rather than on repetition. Something firing twice in a minute is noisy;
    something unacknowledged for an hour is genuinely worse than it was.
    """
    if unacknowledged_for < after:
        return notification.severity
    steps = int(unacknowledged_for / after)
    index = min(notification.severity.rank + steps, len(Severity) - 1)
    return list(Severity)[index]


# -- transports ----------------------------------------------------------------
#
# Pure functions to a payload. Nothing opens a socket, which is what makes routing
# testable without a network and lets callers use whatever HTTP client they have.

_SLACK_COLOURS = {
    Severity.INFO: "#36a64f",
    Severity.WARNING: "#daa038",
    Severity.ERROR: "#d93f0b",
    Severity.CRITICAL: "#b60205",
}


def format_slack(notification: Notification, *, channel: str = "") -> dict[str, Any]:
    fields = [
        {"title": "source", "value": notification.source or "-", "short": True},
        {"title": "target", "value": notification.target or "-", "short": True},
    ]
    fields.extend(
        {"title": str(k), "value": str(v), "short": True}
        for k, v in list(notification.detail.items())[:6]
    )
    attachment: dict[str, Any] = {
        "color": _SLACK_COLOURS[notification.severity],
        "title": notification.title,
        "text": notification.body,
        "fields": fields,
        "ts": int(notification.at.timestamp()),
    }
    if notification.runbook:
        attachment["actions"] = [{"type": "button", "text": "Runbook", "url": notification.runbook}]
    payload: dict[str, Any] = {"attachments": [attachment]}
    if channel:
        payload["channel"] = channel
    return payload


def format_pagerduty(
    notification: Notification, *, routing_key: str, action: str = "trigger"
) -> dict[str, Any]:
    """PagerDuty Events API v2.

    `dedup_key` is our fingerprint, so PagerDuty's own deduplication agrees with
    ours rather than opening a second incident for the same problem.
    """
    return {
        "routing_key": routing_key,
        "event_action": action,
        "dedup_key": notification.fingerprint(),
        "payload": {
            "summary": notification.title,
            "severity": (
                "critical"
                if notification.severity is Severity.CRITICAL
                else notification.severity.value
            ),
            "source": notification.target or notification.source or "fathom",
            "component": notification.source,
            "timestamp": notification.at.isoformat(),
            "custom_details": dict(notification.detail),
        },
        **(
            {"links": [{"href": notification.runbook, "text": "Runbook"}]}
            if notification.runbook
            else {}
        ),
    }


def format_email(notification: Notification, *, to: str, sender: str = "fathom") -> dict[str, str]:
    lines = [notification.body, ""]
    if notification.target:
        lines.append(f"Dataset: {notification.target}")
    if notification.source:
        lines.append(f"Check:   {notification.source}")
    lines.extend(f"{k}: {v}" for k, v in notification.detail.items())
    if notification.runbook:
        lines.extend(["", f"Runbook: {notification.runbook}"])
    return {
        "to": to,
        "from": sender,
        "subject": f"[{notification.severity.value.upper()}] {notification.title}",
        "body": "\n".join(lines),
    }


def format_teams(notification: Notification) -> dict[str, Any]:
    return {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "themeColor": _SLACK_COLOURS[notification.severity].lstrip("#"),
        "summary": notification.title,
        "sections": [
            {
                "activityTitle": notification.title,
                "activitySubtitle": notification.target,
                "text": notification.body,
                "facts": [
                    {"name": str(k), "value": str(v)} for k, v in notification.detail.items()
                ],
            }
        ],
    }


def format_webhook(notification: Notification) -> str:
    """A stable JSON envelope, for anything that is not one of the above."""
    return json.dumps(
        {
            "fingerprint": notification.fingerprint(),
            "title": notification.title,
            "body": notification.body,
            "severity": notification.severity.value,
            "source": notification.source,
            "target": notification.target,
            "at": notification.at.isoformat(),
            "detail": dict(notification.detail),
            "runbook": notification.runbook,
        },
        sort_keys=True,
        default=str,
    )


_FORMATTERS: dict[Channel, Callable[[Notification], Any]] = {
    Channel.SLACK: format_slack,
    Channel.TEAMS: format_teams,
    Channel.WEBHOOK: format_webhook,
    Channel.LOG: format_webhook,
}


def summarize_batch(notifications: Sequence[Notification]) -> str:
    """One message for a run's worth of findings.

    A batch summary is what a nightly job should send. Ninety-six separate messages
    is how a channel gets muted.
    """
    if not notifications:
        return "nothing to report"

    collapsed = dedupe(notifications)
    by_severity: dict[str, int] = {}
    for notification, count in collapsed:
        by_severity[notification.severity.value] = (
            by_severity.get(notification.severity.value, 0) + count
        )

    # By rank, not by value. Severity is a StrEnum, so a plain max() compares the
    # strings and decides "warning" outranks "critical" -- which would quietly
    # deprioritise the one finding in the batch that mattered.
    worst = max((n.severity for n, _ in collapsed), key=lambda s: s.rank)
    head = ", ".join(f"{count} {name}" for name, count in sorted(by_severity.items()))
    lines = [f"{len(collapsed)} distinct finding(s) ({head}); worst is {worst.value}"]
    for notification, count in collapsed[:10]:
        repeat = f" (×{count})" if count > 1 else ""
        lines.append(
            f"  [{notification.severity.value}] {notification.title}{repeat}"
            f"{' — ' + notification.target if notification.target else ''}"
        )
    if len(collapsed) > 10:
        lines.append(f"  ... and {len(collapsed) - 10} more")
    return "\n".join(lines)
