"""Lightweight security, rate limiting, and quota enforcement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from hmac import compare_digest
from threading import RLock
from time import monotonic
from typing import Any, Callable, Mapping, Sequence


def _read_attr(value: object | Mapping[str, Any] | None, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(slots=True, frozen=True)
class TenantIdentity:
    """Canonical tenant identity used by auth and quotas."""

    tenant_key: str | None = None
    base_collection: str = "default"
    user_id: int | None = None
    agent_id: int | None = None

    @classmethod
    def from_value(cls, value: object | Mapping[str, Any] | None) -> TenantIdentity | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        tenant_key = _read_attr(value, "tenant_key")
        if tenant_key is None:
            tenant_key = _read_attr(value, "tenant_id")
        base_collection = str(_read_attr(value, "base_collection", "default") or "default")
        user_id = _read_attr(value, "user_id", _read_attr(value, "_user_id"))
        agent_id = _read_attr(value, "agent_id", _read_attr(value, "_agent_id"))
        return cls(
            tenant_key=str(tenant_key) if tenant_key is not None else None,
            base_collection=base_collection,
            user_id=user_id,
            agent_id=agent_id,
        )

    def canonical_key(self) -> str:
        if self.tenant_key:
            return self.tenant_key
        if self.user_id is None:
            return self.base_collection or "default"
        if self.agent_id is None:
            return f"u{self.user_id}_{self.base_collection or 'default'}"
        return f"u{self.user_id}_a{self.agent_id}_{self.base_collection or 'default'}"


@dataclass(slots=True)
class AuthDecision:
    """Result returned by the auth validator."""

    allowed: bool
    reason: str
    tenant_key: str | None = None
    api_key_present: bool = False
    matched_scope: str = "none"


@dataclass(slots=True)
class QuotaDecision:
    """Result returned by quota checks."""

    kind: str
    allowed: bool
    reason: str
    observed: dict[str, int] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class RateLimitDecision:
    """Result returned by the in-memory rate limiter."""

    subject: str
    allowed: bool
    limit: int
    window_seconds: float
    used: float
    remaining: float
    retry_after_seconds: float = 0.0


class SecurityError(RuntimeError):
    """Base error for auth failures."""


class AuthenticationError(SecurityError):
    """Raised when API key validation fails."""


class AuthorizationError(SecurityError):
    """Raised when a tenant is not allowed for a validated API key."""


class QuotaExceededError(SecurityError):
    """Raised when upload or index quotas are exceeded."""


class RateLimitExceededError(SecurityError):
    """Raised when the rate limiter denies a request."""


class SecurityPolicy:
    """Validate tenant-scoped API keys."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        allow_anonymous: bool = True,
        api_keys: Sequence[str] | None = None,
        tenant_api_keys: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.enabled = enabled
        self.allow_anonymous = allow_anonymous
        self._api_keys = tuple(str(key) for key in (api_keys or ()) if key is not None and str(key))
        self._tenant_api_keys = {
            str(tenant_key): tuple(str(key) for key in keys if key is not None and str(key))
            for tenant_key, keys in (tenant_api_keys or {}).items()
        }

    @classmethod
    def from_settings(cls, settings: object) -> SecurityPolicy:
        security = _read_attr(settings, "security", settings)
        return cls(
            enabled=bool(_read_attr(security, "enabled", False)),
            allow_anonymous=bool(_read_attr(security, "allow_anonymous", True)),
            api_keys=list(_read_attr(security, "api_keys", ()) or ()),
            tenant_api_keys=_read_attr(security, "tenant_api_keys", {}) or {},
        )

    def _matches(self, api_key: str, candidates: Sequence[str]) -> bool:
        return any(compare_digest(api_key, candidate) for candidate in candidates)

    def validate(
        self,
        api_key: str | None = None,
        *,
        tenant: object | Mapping[str, Any] | None = None,
    ) -> AuthDecision:
        tenant_identity = TenantIdentity.from_value(tenant)
        tenant_key = tenant_identity.canonical_key() if tenant_identity else None
        has_api_key = bool(api_key)

        if not self.enabled:
            return AuthDecision(
                allowed=True,
                reason="security disabled",
                tenant_key=tenant_key,
                api_key_present=has_api_key,
                matched_scope="disabled",
            )

        if tenant_key and tenant_key in self._tenant_api_keys:
            allowed = bool(api_key) and self._matches(api_key or "", self._tenant_api_keys[tenant_key])
            if allowed:
                return AuthDecision(
                    allowed=True,
                    reason="tenant api key matched",
                    tenant_key=tenant_key,
                    api_key_present=has_api_key,
                    matched_scope="tenant",
                )
            return AuthDecision(
                allowed=False,
                reason="api key not permitted for tenant",
                tenant_key=tenant_key,
                api_key_present=has_api_key,
                matched_scope="tenant",
            )

        if bool(api_key) and self._matches(api_key or "", self._api_keys):
            return AuthDecision(
                allowed=True,
                reason="global api key matched",
                tenant_key=tenant_key,
                api_key_present=True,
                matched_scope="global",
            )

        if not has_api_key:
            if self.allow_anonymous and not (tenant_key and tenant_key in self._tenant_api_keys):
                return AuthDecision(
                    allowed=True,
                    reason="anonymous access allowed",
                    tenant_key=tenant_key,
                    api_key_present=False,
                    matched_scope="anonymous",
                )
            return AuthDecision(
                allowed=False,
                reason="api key required",
                tenant_key=tenant_key,
                api_key_present=False,
                matched_scope="none",
            )

        return AuthDecision(
            allowed=False,
            reason="invalid api key",
            tenant_key=tenant_key,
            api_key_present=True,
            matched_scope="none",
        )

    def require(
        self,
        api_key: str | None = None,
        *,
        tenant: object | Mapping[str, Any] | None = None,
    ) -> AuthDecision:
        decision = self.validate(api_key, tenant=tenant)
        if decision.allowed:
            return decision
        if decision.reason == "api key required":
            raise AuthenticationError(decision.reason)
        raise AuthorizationError(decision.reason)


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Simple in-memory token bucket limiter."""

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: float = 60.0,
        burst: int = 0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if burst < 0:
            raise ValueError("burst must be non-negative")
        self.limit = limit
        self.window_seconds = float(window_seconds)
        self.burst = burst
        self._clock = clock
        self._rate_per_second = limit / self.window_seconds if limit else 0.0
        self._capacity = float(limit + burst)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = RLock()

    @classmethod
    def from_settings(cls, settings: object) -> TokenBucketRateLimiter:
        rate_limit = _read_attr(settings, "rate_limit", settings)
        return cls(
            limit=int(_read_attr(rate_limit, "requests_per_window", 0)),
            window_seconds=float(_read_attr(rate_limit, "window_seconds", 60)),
            burst=int(_read_attr(rate_limit, "burst", 0)),
        )

    def _get_bucket(self, subject: str, now: float) -> _Bucket:
        bucket = self._buckets.get(subject)
        if bucket is None:
            bucket = _Bucket(tokens=self._capacity, updated_at=now)
            self._buckets[subject] = bucket
            return bucket

        elapsed = max(0.0, now - bucket.updated_at)
        if self._rate_per_second > 0:
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate_per_second)
        else:
            bucket.tokens = min(self._capacity, bucket.tokens)
        bucket.updated_at = now
        return bucket

    def allow(self, subject: str, *, weight: int = 1) -> RateLimitDecision:
        if weight <= 0:
            raise ValueError("weight must be positive")

        now = self._clock()
        with self._lock:
            bucket = self._get_bucket(subject, now)
            if bucket.tokens >= weight:
                bucket.tokens -= weight
                return RateLimitDecision(
                    subject=subject,
                    allowed=True,
                    limit=self.limit,
                    window_seconds=self.window_seconds,
                    used=float(self.limit + self.burst) - bucket.tokens,
                    remaining=bucket.tokens,
                    retry_after_seconds=0.0,
                )

            missing = weight - bucket.tokens
            retry_after = missing / self._rate_per_second if self._rate_per_second > 0 else self.window_seconds
            return RateLimitDecision(
                subject=subject,
                allowed=False,
                limit=self.limit,
                window_seconds=self.window_seconds,
                used=float(self.limit + self.burst) - bucket.tokens,
                remaining=bucket.tokens,
                retry_after_seconds=retry_after,
            )

    def require(self, subject: str, *, weight: int = 1) -> RateLimitDecision:
        decision = self.allow(subject, weight=weight)
        if decision.allowed:
            return decision
        raise RateLimitExceededError(
            f"rate limit exceeded for {subject}: retry after {decision.retry_after_seconds:.2f}s"
        )

    def snapshot(self) -> dict[str, RateLimitDecision]:
        now = self._clock()
        with self._lock:
            return {
                subject: self._decision_for_bucket(subject, bucket, now)
                for subject, bucket in self._buckets.items()
            }

    def _decision_for_bucket(self, subject: str, bucket: _Bucket, now: float) -> RateLimitDecision:
        elapsed = max(0.0, now - bucket.updated_at)
        tokens = bucket.tokens
        if self._rate_per_second > 0:
            tokens = min(self._capacity, tokens + elapsed * self._rate_per_second)
        return RateLimitDecision(
            subject=subject,
            allowed=tokens >= 1,
            limit=self.limit,
            window_seconds=self.window_seconds,
            used=float(self.limit + self.burst) - tokens,
            remaining=tokens,
            retry_after_seconds=0.0 if tokens >= 1 or self._rate_per_second <= 0 else (1 - tokens) / self._rate_per_second,
        )


class UploadQuotaPolicy:
    """Check upload batch size and file-size quotas."""

    def __init__(
        self,
        *,
        max_files: int,
        max_total_bytes: int,
        max_file_bytes: int,
    ) -> None:
        for name, value in {
            "max_files": max_files,
            "max_total_bytes": max_total_bytes,
            "max_file_bytes": max_file_bytes,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self.max_file_bytes = max_file_bytes

    @classmethod
    def from_settings(cls, settings: object) -> UploadQuotaPolicy:
        quotas = _read_attr(settings, "quotas", settings)
        return cls(
            max_files=int(_read_attr(quotas, "max_upload_files", 0)),
            max_total_bytes=int(_read_attr(quotas, "max_upload_bytes", 0)),
            max_file_bytes=int(_read_attr(quotas, "max_upload_file_bytes", 0)),
        )

    def check(self, file_sizes: Sequence[int]) -> QuotaDecision:
        sizes = [int(size) for size in file_sizes]
        observed = {
            "files": len(sizes),
            "total_bytes": sum(sizes),
            "largest_file_bytes": max(sizes) if sizes else 0,
        }
        violations: list[str] = []
        if len(sizes) > self.max_files:
            violations.append(f"too many files ({len(sizes)} > {self.max_files})")
        if observed["total_bytes"] > self.max_total_bytes:
            violations.append(f"batch too large ({observed['total_bytes']} > {self.max_total_bytes})")
        if observed["largest_file_bytes"] > self.max_file_bytes:
            violations.append(
                f"file too large ({observed['largest_file_bytes']} > {self.max_file_bytes})"
            )

        allowed = not violations
        return QuotaDecision(
            kind="upload",
            allowed=allowed,
            reason="; ".join(violations) if violations else "upload quota ok",
            observed=observed,
            limits={
                "max_files": self.max_files,
                "max_total_bytes": self.max_total_bytes,
                "max_file_bytes": self.max_file_bytes,
            },
        )

    def require(self, file_sizes: Sequence[int]) -> QuotaDecision:
        decision = self.check(file_sizes)
        if decision.allowed:
            return decision
        raise QuotaExceededError(decision.reason)


class IndexQuotaPolicy:
    """Check indexing batch size and document-size quotas."""

    def __init__(
        self,
        *,
        max_documents: int,
        max_chunks: int,
        max_chars: int,
    ) -> None:
        for name, value in {
            "max_documents": max_documents,
            "max_chunks": max_chunks,
            "max_chars": max_chars,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.max_documents = max_documents
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    @classmethod
    def from_settings(cls, settings: object) -> IndexQuotaPolicy:
        quotas = _read_attr(settings, "quotas", settings)
        return cls(
            max_documents=int(_read_attr(quotas, "max_index_documents", 0)),
            max_chunks=int(_read_attr(quotas, "max_index_chunks", 0)),
            max_chars=int(_read_attr(quotas, "max_index_chars", 0)),
        )

    def check(self, *, document_count: int, chunk_count: int, total_chars: int) -> QuotaDecision:
        observed = {
            "documents": int(document_count),
            "chunks": int(chunk_count),
            "chars": int(total_chars),
        }
        violations: list[str] = []
        if observed["documents"] > self.max_documents:
            violations.append(f"too many documents ({observed['documents']} > {self.max_documents})")
        if observed["chunks"] > self.max_chunks:
            violations.append(f"too many chunks ({observed['chunks']} > {self.max_chunks})")
        if observed["chars"] > self.max_chars:
            violations.append(f"too many characters ({observed['chars']} > {self.max_chars})")

        allowed = not violations
        return QuotaDecision(
            kind="index",
            allowed=allowed,
            reason="; ".join(violations) if violations else "index quota ok",
            observed=observed,
            limits={
                "max_documents": self.max_documents,
                "max_chunks": self.max_chunks,
                "max_chars": self.max_chars,
            },
        )

    def require(self, *, document_count: int, chunk_count: int, total_chars: int) -> QuotaDecision:
        decision = self.check(document_count=document_count, chunk_count=chunk_count, total_chars=total_chars)
        if decision.allowed:
            return decision
        raise QuotaExceededError(decision.reason)
