from __future__ import annotations

from services.observability import MetricsRegistry


def test_metrics_registry_renders_prometheus_counters_and_histograms() -> None:
    registry = MetricsRegistry()
    registry.increment("kawal_events_total", labels={"topic": "cases.ready.v1"})
    registry.observe("kawal_latency_ms", 12.5, labels={"worker": "case-ready"})

    rendered = registry.render_prometheus()

    assert 'kawal_events_total{topic="cases.ready.v1"} 1.0' in rendered
    assert 'kawal_latency_ms_count{worker="case-ready"} 1' in rendered
    assert 'kawal_latency_ms_sum{worker="case-ready"} 12.5' in rendered
