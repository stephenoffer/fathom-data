"""Metrics and traces.

The tests that matter are about the histogram. Buckets are cumulative because that
is what the exposition format means, and getting it wrong produces quantiles that
look plausible and are wrong in a way nobody catches from a dashboard.
"""

from __future__ import annotations

import pytest

from fathom.report.telemetry import (
    DEFAULT_BUCKETS,
    Gauge,
    MetricKind,
    Registry,
    Span,
    Timer,
    escape_label_value,
    measure,
    metric_name,
    otlp_payload,
    quantile,
    render_prometheus,
    span_tree,
    trace_summary,
)


@pytest.fixture
def registry() -> Registry:
    return Registry()


# -- naming --------------------------------------------------------------------


def test_metric_names_follow_the_dotted_otel_convention():
    assert metric_name("invalidation", "duration") == "fathom.invalidation.duration"


def test_metric_names_are_lowercased_and_despaced():
    assert metric_name("Graph Build") == "fathom.graph_build"


def test_label_values_are_escaped():
    """An unescaped quote makes the scraper reject the whole endpoint, not one series."""
    assert escape_label_value('a"b') == 'a\\"b'
    assert escape_label_value("a\\b") == "a\\\\b"


# -- counters and gauges -------------------------------------------------------


def test_a_counter_accumulates_per_label_set(registry):
    counter = registry.counter("fathom.partitions.invalidated")
    counter.add(3, dataset="gold")
    counter.add(2, dataset="gold")
    counter.add(1, dataset="silver")
    assert counter.get(dataset="gold") == 5
    assert counter.total == 6


def test_an_unobserved_label_set_reads_zero(registry):
    assert registry.counter("c").get(dataset="never") == 0.0


def test_a_counter_refuses_to_decrease(registry):
    """A decreasing counter makes every rate() computed over it wrong."""
    with pytest.raises(ValueError, match="cannot decrease"):
        registry.counter("c").add(-1)


def test_label_order_does_not_create_a_second_series(registry):
    counter = registry.counter("c")
    counter.add(1, a="1", b="2")
    counter.add(1, b="2", a="1")
    assert len(counter.values) == 1


def test_a_gauge_moves_in_both_directions():
    gauge = Gauge("fathom.graph.datasets")
    gauge.set(10)
    gauge.add(-3)
    assert gauge.get() == 7


def test_registry_lookups_are_idempotent(registry):
    assert registry.counter("c") is registry.counter("c")
    assert registry.names == ["c"]


def test_the_kinds_are_distinguishable():
    assert Gauge("g").kind is MetricKind.GAUGE


# -- histograms ----------------------------------------------------------------


def test_buckets_are_cumulative(registry):
    """Each bucket counts observations <= its bound. Storing per-bucket counts and
    summing at render time is the mistake that produces wrong quantiles."""
    histogram = registry.histogram("h", buckets=(1.0, 10.0))
    histogram.observe(0.5)
    counts = histogram.counts[()]
    assert counts == [1, 1, 1]  # <=1, <=10, +Inf


def test_a_value_above_every_bound_only_lands_in_inf(registry):
    histogram = registry.histogram("h", buckets=(1.0, 10.0))
    histogram.observe(100.0)
    assert histogram.counts[()] == [0, 0, 1]


def test_a_value_exactly_on_a_bound_is_included(registry):
    histogram = registry.histogram("h", buckets=(1.0,))
    histogram.observe(1.0)
    assert histogram.counts[()][0] == 1


def test_count_and_sum_track_observations(registry):
    histogram = registry.histogram("h")
    histogram.observe(1.0)
    histogram.observe(3.0)
    assert histogram.count() == 2
    assert histogram.sum() == 4.0
    assert histogram.mean() == 2.0


def test_the_mean_of_nothing_is_zero_rather_than_a_crash(registry):
    assert registry.histogram("h").mean() == 0.0


def test_the_default_buckets_reach_minutes():
    """The usual client defaults top out at 10s, which puts every slow rebuild in
    +Inf and makes the p99 unreadable."""
    assert max(DEFAULT_BUCKETS) >= 300.0


def test_a_quantile_interpolates_within_a_bucket(registry):
    histogram = registry.histogram("h", buckets=(1.0, 2.0, 3.0))
    for value in (0.5, 1.5, 2.5):
        histogram.observe(value)
    assert quantile(histogram, 0.5) == pytest.approx(1.5, abs=0.6)


def test_a_quantile_above_the_highest_bucket_is_unknown_not_the_last_bound():
    """Returning the last finite bound reports a p99 equal to the bucket edge no
    matter how bad things actually are."""
    histogram = Registry().histogram("h", buckets=(1.0,))
    histogram.observe(500.0)
    assert quantile(histogram, 0.99) is None


def test_a_quantile_of_an_empty_histogram_is_none(registry):
    assert quantile(registry.histogram("h"), 0.5) is None


def test_an_out_of_range_quantile_raises(registry):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quantile(registry.histogram("h"), 1.5)


# -- timing --------------------------------------------------------------------


def test_measure_records_the_duration(registry):
    histogram = registry.histogram("h")
    with measure(histogram, phase="plan"):
        pass
    assert histogram.count(phase="plan") == 1


def test_measure_records_even_when_the_block_raises(registry):
    """Recording only successes is how a system reports excellent latency while
    timing out on half its requests."""
    histogram = registry.histogram("h")
    with pytest.raises(RuntimeError), measure(histogram):
        raise RuntimeError("boom")
    assert histogram.count() == 1


def test_a_timer_records_on_stop(registry):
    histogram = registry.histogram("h")
    elapsed = Timer(histogram).start().stop()
    assert elapsed >= 0
    assert histogram.count() == 1


def test_stopping_an_unstarted_timer_raises(registry):
    with pytest.raises(RuntimeError, match="never started"):
        Timer(registry.histogram("h")).stop()


# -- prometheus ----------------------------------------------------------------


def test_prometheus_output_carries_help_and_type(registry):
    registry.counter("fathom.runs", description="runs started").add(1)
    text = render_prometheus(registry)
    assert "# HELP fathom_runs runs started" in text
    assert "# TYPE fathom_runs counter" in text


def test_dots_become_underscores_in_prometheus(registry):
    registry.gauge("fathom.graph.datasets").set(4)
    assert "fathom_graph_datasets 4" in render_prometheus(registry)


def test_labels_are_rendered(registry):
    registry.counter("c").add(2, dataset="gold.daily")
    assert 'c{dataset="gold.daily"} 2' in render_prometheus(registry)


def test_a_histogram_renders_buckets_sum_and_count(registry):
    histogram = registry.histogram("h", buckets=(1.0,))
    histogram.observe(0.5)
    text = render_prometheus(registry)
    assert 'h_bucket{le="1"} 1' in text
    assert 'h_bucket{le="+Inf"} 1' in text
    assert "h_sum 0.5" in text
    assert "h_count 1" in text


def test_output_is_deterministic(registry):
    registry.counter("c").add(1, b="2", a="1")
    assert render_prometheus(registry) == render_prometheus(registry)


def test_an_empty_registry_renders_nothing(registry):
    assert render_prometheus(registry) == ""


# -- otlp ----------------------------------------------------------------------


def test_otlp_marks_counters_monotonic_and_cumulative(registry):
    registry.counter("fathom.runs").add(1)
    body = otlp_payload(registry)
    metric = body["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
    assert metric["sum"]["isMonotonic"] is True
    assert metric["sum"]["aggregationTemporality"] == 2


def test_otlp_histogram_buckets_are_differenced_not_cumulative(registry):
    """OTLP wants per-bucket counts; ours are cumulative. Shipping cumulative counts
    into an OTLP field is a silent double-count in every collector."""
    histogram = registry.histogram("h", buckets=(1.0, 10.0))
    histogram.observe(0.5)
    histogram.observe(5.0)
    point = otlp_payload(registry)["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0][
        "histogram"
    ]["dataPoints"][0]
    assert point["bucketCounts"] == [1, 1, 0]
    assert sum(point["bucketCounts"]) == point["count"]


def test_otlp_carries_the_service_name(registry):
    registry.gauge("g").set(1)
    resource = otlp_payload(registry, service="fathom-ci")["resourceMetrics"][0]["resource"]
    assert resource["attributes"][0]["value"]["stringValue"] == "fathom-ci"


def test_otlp_of_an_empty_registry_has_no_metrics(registry):
    assert otlp_payload(registry)["resourceMetrics"][0]["scopeMetrics"][0]["metrics"] == []


# -- tracing -------------------------------------------------------------------


def test_roots_are_indexed_under_the_empty_parent():
    spans = [Span("plan", "1"), Span("resolve", "2", parent_id="1")]
    tree = span_tree(spans)
    assert [s.name for s in tree[""]] == ["plan"]
    assert [s.name for s in tree["1"]] == ["resolve"]


def test_a_span_knows_whether_it_is_a_root_and_whether_it_failed():
    assert Span("a", "1").is_root
    assert not Span("a", "1", parent_id="0").is_root
    assert Span("a", "1", error="timeout").failed


def test_the_summary_reports_self_time_not_just_total():
    """A parent showing 40 seconds says nothing about which child spent them."""
    spans = [
        Span("plan", "1", duration_seconds=10.0),
        Span("resolve", "2", parent_id="1", duration_seconds=9.0),
    ]
    summary = trace_summary(spans)
    assert "self 1.000s" in summary
    assert "self 9.000s" in summary


def test_self_time_never_goes_negative():
    spans = [
        Span("parent", "1", duration_seconds=1.0),
        Span("child", "2", parent_id="1", duration_seconds=5.0),
    ]
    assert "self -" not in trace_summary(spans)


def test_a_failed_span_is_marked():
    assert "FAILED" in trace_summary([Span("plan", "1", error="timeout")])


def test_an_empty_trace_says_so():
    assert trace_summary([]) == "no spans"
