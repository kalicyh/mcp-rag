from __future__ import annotations

import unittest

from mcp_rag.observability import ObservabilityCollector


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class ObservabilityCollectorTests(unittest.TestCase):
    def test_timer_snapshot_and_health_summary(self) -> None:
        clock = _Clock()
        collector = ObservabilityCollector(
            clock=clock,
            warning_error_rate=0.4,
            critical_error_rate=0.8,
            slow_request_ms=20.0,
        )

        with collector.timer("search"):
            clock.advance(0.01)

        collector.record_request("upload", 50.0, success=False, error="ValueError")

        snapshot = collector.snapshot()
        self.assertEqual(snapshot.total_requests, 2)
        self.assertEqual(snapshot.error_count, 1)
        self.assertAlmostEqual(snapshot.average_latency_ms, 30.0)
        self.assertAlmostEqual(snapshot.operations["search"].average_latency_ms, 10.0)
        self.assertEqual(snapshot.operations["upload"].last_error, "ValueError")

        health = collector.health_summary()
        self.assertEqual(health.status, "degraded")
        self.assertFalse(health.healthy)
        self.assertIn("upload", health.slow_operations)
        self.assertTrue(any("error_rate" in reason for reason in health.reasons))

        payload = collector.as_dict()
        self.assertEqual(payload["health"]["status"], "degraded")
        self.assertIn("search", payload["metrics"]["operations"])


if __name__ == "__main__":
    unittest.main()
