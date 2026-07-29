"""Making fathom's own behaviour observable.

A tool that watches other systems and reports nothing about itself is asking to be
trusted on faith. When invalidation takes four minutes on a graph that took four
seconds last week, nothing here would currently say so.

Two decisions worth stating.

**Nothing is exported over a network.** `render_prometheus` and `otlp_payload`
produce text and dicts. A library that opens a socket to a collector during
`import` is a library that fails in an air-gapped environment and hangs in CI, and
the format is the hard part anyway — the transport is four lines of whatever the
host already uses.

**Histograms are cumulative, as the format requires.** A Prometheus histogram bucket
counts observations *less than or equal to* its bound, and every implementation that
gets this wrong produces quantiles that are quietly, unfixably wrong.

Metric names follow the OpenTelemetry semantic conventions where one exists, so
`fathom.invalidation.duration` sits beside everything else in the same dashboard
rather than needing its own.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Counter",
    "DEFAULT_BUCKETS",
    "Gauge",
    "Histogram",
    "MetricKind",
    "Registry",
    "Span",
    "Timer",
    "escape_label_value",
    "measure",
    "metric_name",
    "otlp_payload",
    "quantile",
    "render_prometheus",
    "span_tree",
    "trace_summary",
]

# Chosen for graph work: sub-millisecond operations up to multi-minute rebuilds. The
# default client buckets top out at 10s, which puts every slow invalidation in +Inf.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
    30.0,
    60.0,
    300.0,
)


class MetricKind(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


def metric_name(*parts: str) -> str:
    """`fathom.invalidation.duration` — OTel dotted convention, lowercase."""
    return ".".join(p.strip(".").lower().replace(" ", "_") for p in ("fathom", *parts) if p)


def escape_label_value(value: str) -> str:
    """Prometheus exposition escaping.

    A label carrying a dataset name with a quote in it produces a line the scraper
    rejects, and the whole endpoint fails rather than that one series.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _key(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


@dataclass
class Counter:
    """Monotonic. Never decreases, so a scraper can compute a rate across restarts."""

    name: str
    description: str = ""
    unit: str = "1"
    values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    kind = MetricKind.COUNTER

    def add(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError(
                f"{self.name} is a counter and cannot decrease by {amount}; a decreasing "
                "counter makes every rate() over it wrong"
            )
        key = _key(labels)
        self.values[key] = self.values.get(key, 0.0) + amount

    def get(self, **labels: str) -> float:
        return self.values.get(_key(labels), 0.0)

    @property
    def total(self) -> float:
        return sum(self.values.values())


@dataclass
class Gauge:
    """A point-in-time value. May go up or down."""

    name: str
    description: str = ""
    unit: str = "1"
    values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    kind = MetricKind.GAUGE

    def set(self, value: float, **labels: str) -> None:
        self.values[_key(labels)] = value

    def add(self, amount: float, **labels: str) -> None:
        key = _key(labels)
        self.values[key] = self.values.get(key, 0.0) + amount

    def get(self, **labels: str) -> float:
        return self.values.get(_key(labels), 0.0)


@dataclass
class Histogram:
    """Bucketed observations.

    Buckets are cumulative — each counts observations `<=` its bound — because that
    is what the exposition format means. Storing per-bucket counts here and summing
    at render time is the mistake that produces plausible, wrong quantiles.
    """

    name: str
    description: str = ""
    unit: str = "s"
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    counts: dict[tuple[tuple[str, str], ...], list[int]] = field(default_factory=dict)
    sums: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    kind = MetricKind.HISTOGRAM

    def observe(self, value: float, **labels: str) -> None:
        key = _key(labels)
        counts = self.counts.setdefault(key, [0] * (len(self.buckets) + 1))
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                counts[index] += 1
        counts[-1] += 1  # +Inf always counts everything
        self.sums[key] = self.sums.get(key, 0.0) + value

    def count(self, **labels: str) -> int:
        counts = self.counts.get(_key(labels))
        return counts[-1] if counts else 0

    def sum(self, **labels: str) -> float:
        return self.sums.get(_key(labels), 0.0)

    def mean(self, **labels: str) -> float:
        total = self.count(**labels)
        return self.sum(**labels) / total if total else 0.0


def quantile(histogram: Histogram, q: float, **labels: str) -> float | None:
    """Interpolate a quantile from cumulative buckets.

    Returns `None` rather than a number when the answer falls in the `+Inf` bucket:
    there is no upper bound to interpolate against, and every library that returns
    the last finite bound instead reports a p99 that is exactly the bucket edge, no
    matter how bad things actually are.
    """
    if not 0 <= q <= 1:
        raise ValueError(f"quantile must be in [0, 1], got {q}")
    counts = histogram.counts.get(_key(labels))
    if not counts or counts[-1] == 0:
        return None

    target = q * counts[-1]
    previous_bound, previous_count = 0.0, 0
    for index, bound in enumerate(histogram.buckets):
        if counts[index] >= target:
            span = counts[index] - previous_count
            if span == 0:
                return bound
            fraction = (target - previous_count) / span
            return previous_bound + fraction * (bound - previous_bound)
        previous_bound, previous_count = bound, counts[index]
    return None  # the quantile lives above the highest bucket


@dataclass
class Registry:
    """Everything being measured. Held by the caller, not a module global.

    A module-level registry makes two runs in one process contaminate each other,
    which is exactly what happens in tests and in any long-lived server.
    """

    counters: dict[str, Counter] = field(default_factory=dict)
    gauges: dict[str, Gauge] = field(default_factory=dict)
    histograms: dict[str, Histogram] = field(default_factory=dict)

    def counter(self, name: str, *, description: str = "", unit: str = "1") -> Counter:
        return self.counters.setdefault(name, Counter(name, description=description, unit=unit))

    def gauge(self, name: str, *, description: str = "", unit: str = "1") -> Gauge:
        return self.gauges.setdefault(name, Gauge(name, description=description, unit=unit))

    def histogram(
        self,
        name: str,
        *,
        description: str = "",
        unit: str = "s",
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        return self.histograms.setdefault(
            name,
            Histogram(name, description=description, unit=unit, buckets=tuple(buckets)),
        )

    @property
    def names(self) -> list[str]:
        return sorted([*self.counters, *self.gauges, *self.histograms])


@contextmanager
def measure(histogram: Histogram, **labels: str) -> Iterator[None]:
    """Time a block, recording it even when it raises.

    Recording only successes is how a system reports excellent latency while timing
    out on half its requests.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - started, **labels)


@dataclass
class Timer:
    """A stopwatch for code that cannot be wrapped in a `with`."""

    histogram: Histogram
    labels: Mapping[str, str] = field(default_factory=dict)
    _started: float | None = None

    def start(self) -> Timer:
        self._started = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._started is None:
            raise RuntimeError("timer was never started")
        elapsed = time.perf_counter() - self._started
        self.histogram.observe(elapsed, **dict(self.labels))
        self._started = None
        return elapsed


# -- exposition ----------------------------------------------------------------


def _prom_name(name: str) -> str:
    """OTel dots become Prometheus underscores, per the OTel-to-Prometheus rules."""
    return name.replace(".", "_").replace("-", "_")


def _le(bound: str) -> str:
    """The `le` label a histogram bucket carries. Built without an f-string so the
    quoting stays legal on Python 3.11, which predates PEP 701."""
    return 'le="' + bound + '"'


def _labels_text(key: tuple[tuple[str, str], ...], extra: str = "") -> str:
    pairs = [f'{k}="{escape_label_value(v)}"' for k, v in key]
    if extra:
        pairs.append(extra)
    return "{" + ",".join(pairs) + "}" if pairs else ""


def render_prometheus(registry: Registry) -> str:
    """The text exposition format, ready to serve at `/metrics`.

    Deterministic ordering, so a diff between two scrapes shows what changed rather
    than what got re-hashed.
    """
    lines: list[str] = []

    for counter in sorted(registry.counters.values(), key=lambda c: c.name):
        name = _prom_name(counter.name)
        lines.append(f"# HELP {name} {counter.description or counter.name}")
        lines.append(f"# TYPE {name} counter")
        for key in sorted(counter.values):
            lines.append(f"{name}{_labels_text(key)} {counter.values[key]:g}")

    for gauge in sorted(registry.gauges.values(), key=lambda g: g.name):
        name = _prom_name(gauge.name)
        lines.append(f"# HELP {name} {gauge.description or gauge.name}")
        lines.append(f"# TYPE {name} gauge")
        for key in sorted(gauge.values):
            lines.append(f"{name}{_labels_text(key)} {gauge.values[key]:g}")

    for histogram in sorted(registry.histograms.values(), key=lambda h: h.name):
        name = _prom_name(histogram.name)
        lines.append(f"# HELP {name} {histogram.description or histogram.name}")
        lines.append(f"# TYPE {name} histogram")
        for key in sorted(histogram.counts):
            counts = histogram.counts[key]
            for index, bound in enumerate(histogram.buckets):
                bucket = _labels_text(key, _le(format(bound, "g")))
                lines.append(f"{name}_bucket{bucket} {counts[index]}")
            infinite = _labels_text(key, _le("+Inf"))
            lines.append(f"{name}_bucket{infinite} {counts[-1]}")
            lines.append(f"{name}_sum{_labels_text(key)} {histogram.sums.get(key, 0.0):g}")
            lines.append(f"{name}_count{_labels_text(key)} {counts[-1]}")

    return "\n".join(lines) + "\n" if lines else ""


def otlp_payload(registry: Registry, *, service: str = "fathom") -> dict[str, object]:
    """An OTLP/JSON metrics body.

    Structural only — no transport. Hand it to whatever the host already uses to talk
    to its collector.
    """
    metrics: list[dict[str, object]] = []

    for counter in sorted(registry.counters.values(), key=lambda c: c.name):
        metrics.append(
            {
                "name": counter.name,
                "unit": counter.unit,
                "description": counter.description,
                "sum": {
                    "isMonotonic": True,
                    "aggregationTemporality": 2,  # cumulative
                    "dataPoints": [
                        {
                            "asDouble": value,
                            "attributes": [{"key": k, "value": {"stringValue": v}} for k, v in key],
                        }
                        for key, value in sorted(counter.values.items())
                    ],
                },
            }
        )

    for gauge in sorted(registry.gauges.values(), key=lambda g: g.name):
        metrics.append(
            {
                "name": gauge.name,
                "unit": gauge.unit,
                "description": gauge.description,
                "gauge": {
                    "dataPoints": [
                        {
                            "asDouble": value,
                            "attributes": [{"key": k, "value": {"stringValue": v}} for k, v in key],
                        }
                        for key, value in sorted(gauge.values.items())
                    ]
                },
            }
        )

    for histogram in sorted(registry.histograms.values(), key=lambda h: h.name):
        points: list[dict[str, object]] = []
        for key, counts in sorted(histogram.counts.items()):
            # OTLP wants per-bucket counts; ours are cumulative, so difference them.
            per_bucket = [counts[0], *(counts[i] - counts[i - 1] for i in range(1, len(counts)))]
            points.append(
                {
                    "count": counts[-1],
                    "sum": histogram.sums.get(key, 0.0),
                    "explicitBounds": list(histogram.buckets),
                    "bucketCounts": per_bucket,
                    "attributes": [{"key": k, "value": {"stringValue": v}} for k, v in key],
                }
            )
        metrics.append(
            {
                "name": histogram.name,
                "unit": histogram.unit,
                "description": histogram.description,
                "histogram": {"aggregationTemporality": 2, "dataPoints": points},
            }
        )

    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": service}}]
                },
                "scopeMetrics": [{"scope": {"name": "fathom"}, "metrics": metrics}],
            }
        ]
    }


# -- tracing -------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """One unit of work, and what it happened inside of."""

    name: str
    span_id: str
    parent_id: str = ""
    duration_seconds: float = 0.0
    attributes: Mapping[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def is_root(self) -> bool:
        return not self.parent_id

    @property
    def failed(self) -> bool:
        return bool(self.error)


def span_tree(spans: Iterable[Span]) -> dict[str, list[Span]]:
    """Children indexed by parent id. Roots live under `""`."""
    tree: dict[str, list[Span]] = {}
    for span in spans:
        tree.setdefault(span.parent_id, []).append(span)
    for children in tree.values():
        children.sort(key=lambda s: s.name)
    return tree


def trace_summary(spans: Sequence[Span]) -> str:
    """An indented tree with self-time.

    Self-time rather than total, because a parent showing 40 seconds tells you
    nothing about which of its six children spent them.
    """
    if not spans:
        return "no spans"
    tree = span_tree(spans)
    lines: list[str] = []

    def walk(parent: str, depth: int) -> None:
        for span in tree.get(parent, []):
            children = tree.get(span.span_id, [])
            self_time = span.duration_seconds - sum(c.duration_seconds for c in children)
            mark = " FAILED" if span.failed else ""
            lines.append(
                f"{'  ' * depth}{span.name} {span.duration_seconds:.3f}s "
                f"(self {max(0.0, self_time):.3f}s){mark}"
            )
            walk(span.span_id, depth + 1)

    walk("", 0)
    return "\n".join(lines)
