"""Lightweight request metrics and health summary helpers."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass, field
from math import ceil
from threading import RLock
from time import monotonic
from typing import Any, Callable, Mapping


def _read_attr(value: object | Mapping[str, Any] | None, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(slots=True)
class _OperationState:
    count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    last_error: str | None = None
    last_seen_at: float | None = None
    samples: list[float] = field(default_factory=list)

    def observe(self, duration_ms: float, *, success: bool, error: str | None, seen_at: float) -> None:
        self.count += 1
        if not success:
            self.error_count += 1
            self.last_error = error
        self.total_latency_ms += duration_ms
        self.min_latency_ms = min(self.min_latency_ms, duration_ms)
        self.max_latency_ms = max(self.max_latency_ms, duration_ms)
        self.last_latency_ms = duration_ms
        self.last_seen_at = seen_at
        self.samples.append(duration_ms)
        if len(self.samples) > 512:
            self.samples.pop(0)

    def snapshot(self) -> OperationStats:
        average = self.total_latency_ms / self.count if self.count else 0.0
        p50, p95, p99 = _percentiles(self.samples)
        return OperationStats(
            count=self.count,
            error_count=self.error_count,
            total_latency_ms=self.total_latency_ms,
            average_latency_ms=average,
            min_latency_ms=0.0 if self.count == 0 else self.min_latency_ms,
            max_latency_ms=self.max_latency_ms,
            last_latency_ms=self.last_latency_ms,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            last_error=self.last_error,
            last_seen_at=self.last_seen_at,
        )


@dataclass(slots=True, frozen=True)
class OperationStats:
    """Structured per-operation metrics."""

    count: int
    error_count: int
    total_latency_ms: float
    average_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    last_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    last_error: str | None = None
    last_seen_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "error_count": self.error_count,
            "total_latency_ms": self.total_latency_ms,
            "average_latency_ms": self.average_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "last_latency_ms": self.last_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "last_error": self.last_error,
            "last_seen_at": self.last_seen_at,
        }


@dataclass(slots=True, frozen=True)
class MetricsSnapshot:
    """Structured global metrics."""

    total_requests: int
    error_count: int
    total_latency_ms: float
    average_latency_ms: float
    uptime_seconds: float
    operations: dict[str, OperationStats] = field(default_factory=dict)
    providers: dict[str, OperationStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "total_latency_ms": self.total_latency_ms,
            "average_latency_ms": self.average_latency_ms,
            "uptime_seconds": self.uptime_seconds,
            "operations": {name: stats.to_dict() for name, stats in self.operations.items()},
            "providers": {name: stats.to_dict() for name, stats in self.providers.items()},
        }


@dataclass(slots=True, frozen=True)
class HealthSummary:
    """Simple health summary suitable for a `/health` handler."""

    status: str
    healthy: bool
    total_requests: int
    error_count: int
    error_rate: float
    average_latency_ms: float
    slow_operations: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "healthy": self.healthy,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "average_latency_ms": self.average_latency_ms,
            "slow_operations": list(self.slow_operations),
            "reasons": list(self.reasons),
        }


class _ObservationTimer(AbstractContextManager["_ObservationTimer"], AbstractAsyncContextManager["_ObservationTimer"]):
    def __init__(self, collector: ObservabilityCollector, operation: str) -> None:
        self._collector = collector
        self._operation = operation
        self._start: float | None = None

    def __enter__(self) -> _ObservationTimer:
        self._start = self._collector._clock()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._record(exc_type, exc)
        return False

    async def __aenter__(self) -> _ObservationTimer:
        self._start = self._collector._clock()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._record(exc_type, exc)
        return False

    def _record(self, exc_type, exc) -> None:
        if self._start is None:
            return
        end = self._collector._clock()
        duration_ms = (end - self._start) * 1000.0
        self._collector.record_request(
            self._operation,
            duration_ms,
            success=exc_type is None,
            error=exc_type.__name__ if exc_type is not None else None,
        )


class ObservabilityCollector:
    """Collect request counts, latencies, and health signals."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        warning_error_rate: float = 0.05,
        critical_error_rate: float = 0.2,
        slow_request_ms: float = 1000.0,
    ) -> None:
        if not 0 <= warning_error_rate <= 1:
            raise ValueError("warning_error_rate must be between 0 and 1")
        if not 0 <= critical_error_rate <= 1:
            raise ValueError("critical_error_rate must be between 0 and 1")
        if slow_request_ms < 0:
            raise ValueError("slow_request_ms must be non-negative")
        self._clock = clock
        self._started_at = clock()
        self._warning_error_rate = warning_error_rate
        self._critical_error_rate = critical_error_rate
        self._slow_request_ms = slow_request_ms
        self._lock = RLock()
        self._total_requests = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self._operations: dict[str, _OperationState] = {}
        self._providers: dict[str, _OperationState] = {}

    @classmethod
    def from_settings(cls, settings: object) -> ObservabilityCollector:
        obs = _read_attr(settings, "observability", settings)
        return cls(
            warning_error_rate=float(_read_attr(obs, "warning_error_rate", 0.05)),
            critical_error_rate=float(_read_attr(obs, "critical_error_rate", 0.2)),
            slow_request_ms=float(_read_attr(obs, "slow_request_ms", 1000.0)),
        )

    def timer(self, operation: str) -> _ObservationTimer:
        return _ObservationTimer(self, operation)

    def configure_from_settings(self, settings: object) -> None:
        """Refresh health thresholds without dropping accumulated metrics."""

        obs = _read_attr(settings, "observability", settings)
        warning_error_rate = float(_read_attr(obs, "warning_error_rate", 0.05))
        critical_error_rate = float(_read_attr(obs, "critical_error_rate", 0.2))
        slow_request_ms = float(_read_attr(obs, "slow_request_ms", 1000.0))

        if not 0 <= warning_error_rate <= 1:
            raise ValueError("warning_error_rate must be between 0 and 1")
        if not 0 <= critical_error_rate <= 1:
            raise ValueError("critical_error_rate must be between 0 and 1")
        if slow_request_ms < 0:
            raise ValueError("slow_request_ms must be non-negative")

        with self._lock:
            self._warning_error_rate = warning_error_rate
            self._critical_error_rate = critical_error_rate
            self._slow_request_ms = slow_request_ms

    def record_request(
        self,
        operation: str,
        duration_ms: float,
        *,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        seen_at = self._clock()
        with self._lock:
            state = self._operations.setdefault(operation, _OperationState())
            state.observe(duration_ms, success=success, error=error, seen_at=seen_at)
            self._total_requests += 1
            self._total_latency_ms += duration_ms
            if not success:
                self._error_count += 1

    def record_provider_latency(
        self,
        provider: str,
        operation: str,
        duration_ms: float,
        *,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        key = f"{provider}.{operation}"
        seen_at = self._clock()
        with self._lock:
            state = self._providers.setdefault(key, _OperationState())
            state.observe(duration_ms, success=success, error=error, seen_at=seen_at)

    def observe_latency(self, operation: str, duration_ms: float) -> None:
        self.record_request(operation, duration_ms)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            operations = {name: state.snapshot() for name, state in self._operations.items()}
            providers = {name: state.snapshot() for name, state in self._providers.items()}
            average = self._total_latency_ms / self._total_requests if self._total_requests else 0.0
            return MetricsSnapshot(
                total_requests=self._total_requests,
                error_count=self._error_count,
                total_latency_ms=self._total_latency_ms,
                average_latency_ms=average,
                uptime_seconds=max(0.0, self._clock() - self._started_at),
                operations=operations,
                providers=providers,
            )

    def health_summary(self) -> HealthSummary:
        snapshot = self.snapshot()
        error_rate = snapshot.error_count / snapshot.total_requests if snapshot.total_requests else 0.0
        slow_operations = tuple(
            sorted(
                name
                for name, stats in snapshot.operations.items()
                if stats.count and (
                    stats.average_latency_ms >= self._slow_request_ms
                    or stats.max_latency_ms >= self._slow_request_ms
                )
            )
        )

        reasons: list[str] = []
        status = "healthy"
        if error_rate >= self._critical_error_rate:
            status = "unhealthy"
            reasons.append(f"error_rate {error_rate:.2f} >= {self._critical_error_rate:.2f}")
        elif error_rate >= self._warning_error_rate:
            status = "degraded"
            reasons.append(f"error_rate {error_rate:.2f} >= {self._warning_error_rate:.2f}")

        if slow_operations and status == "healthy":
            status = "degraded"
        if slow_operations:
            reasons.append(f"slow operations: {', '.join(slow_operations)}")

        return HealthSummary(
            status=status,
            healthy=status == "healthy",
            total_requests=snapshot.total_requests,
            error_count=snapshot.error_count,
            error_rate=error_rate,
            average_latency_ms=snapshot.average_latency_ms,
            slow_operations=slow_operations,
            reasons=tuple(reasons),
        )

    def as_dict(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        health = self.health_summary()
        return {
            "metrics": snapshot.to_dict(),
            "health": health.to_dict(),
        }


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    ordered = sorted(samples)

    def _pick(percentile: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = max(0, min(len(ordered) - 1, ceil(percentile * len(ordered)) - 1))
        return ordered[index]

    return _pick(0.50), _pick(0.95), _pick(0.99)


MetricsCollector = ObservabilityCollector
