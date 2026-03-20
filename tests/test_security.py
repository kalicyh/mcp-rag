from __future__ import annotations

import unittest

from mcp_rag.config import Settings
from mcp_rag.security import (
    AuthorizationError,
    SecurityPolicy,
    TenantIdentity,
    TokenBucketRateLimiter,
    UploadQuotaPolicy,
    IndexQuotaPolicy,
)


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SecurityPolicyTests(unittest.TestCase):
    def test_tenant_scoped_keys_take_precedence(self) -> None:
        policy = SecurityPolicy(
            enabled=True,
            allow_anonymous=False,
            api_keys=["global-secret"],
            tenant_api_keys={"u7_docs": ["tenant-secret"]},
        )
        tenant = TenantIdentity(base_collection="docs", user_id=7)

        denied = policy.validate("global-secret", tenant=tenant)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.matched_scope, "tenant")

        allowed = policy.validate("tenant-secret", tenant=tenant)
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.matched_scope, "tenant")

        other = policy.validate("global-secret", tenant=TenantIdentity(base_collection="docs", user_id=9))
        self.assertTrue(other.allowed)
        self.assertEqual(other.matched_scope, "global")

    def test_anonymous_access_can_be_enabled_without_keys(self) -> None:
        policy = SecurityPolicy(enabled=True, allow_anonymous=True)
        decision = policy.validate(None, tenant=TenantIdentity(base_collection="default"))
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.matched_scope, "anonymous")

    def test_require_raises_for_missing_or_wrong_keys(self) -> None:
        policy = SecurityPolicy(
            enabled=True,
            allow_anonymous=False,
            api_keys=["global-secret"],
            tenant_api_keys={"u7_docs": ["tenant-secret"]},
        )

        with self.assertRaises(AuthorizationError):
            policy.require("global-secret", tenant=TenantIdentity(base_collection="docs", user_id=7))


class RateLimiterTests(unittest.TestCase):
    def test_token_bucket_refills_over_time(self) -> None:
        clock = _Clock()
        limiter = TokenBucketRateLimiter(limit=2, window_seconds=10, burst=0, clock=clock)

        first = limiter.allow("tenant-a")
        second = limiter.allow("tenant-a")
        third = limiter.allow("tenant-a")

        self.assertTrue(first.allowed)
        self.assertTrue(second.allowed)
        self.assertFalse(third.allowed)
        self.assertGreater(third.retry_after_seconds, 0)

        clock.advance(5)
        fourth = limiter.allow("tenant-a")
        self.assertTrue(fourth.allowed)


class QuotaPolicyTests(unittest.TestCase):
    def test_upload_quota_checks_count_and_bytes(self) -> None:
        policy = UploadQuotaPolicy(max_files=2, max_total_bytes=10, max_file_bytes=8)

        allowed = policy.check([3, 4])
        self.assertTrue(allowed.allowed)

        too_many = policy.check([3, 4, 5])
        self.assertFalse(too_many.allowed)
        self.assertIn("too many files", too_many.reason)
        self.assertIn("batch too large", too_many.reason)

        too_large = policy.check([9])
        self.assertFalse(too_large.allowed)
        self.assertIn("file too large", too_large.reason)

    def test_index_quota_checks_documents_chunks_and_chars(self) -> None:
        policy = IndexQuotaPolicy(max_documents=3, max_chunks=5, max_chars=100)

        allowed = policy.check(document_count=3, chunk_count=5, total_chars=100)
        self.assertTrue(allowed.allowed)

        too_much = policy.check(document_count=4, chunk_count=6, total_chars=101)
        self.assertFalse(too_much.allowed)
        self.assertIn("too many documents", too_much.reason)
        self.assertIn("too many chunks", too_much.reason)
        self.assertIn("too many characters", too_much.reason)


class ConfigIntegrationTests(unittest.TestCase):
    def test_settings_include_security_and_guardrail_defaults(self) -> None:
        settings = Settings()
        self.assertFalse(settings.security.enabled)
        self.assertEqual(settings.rate_limit.requests_per_window, 120)
        self.assertEqual(settings.quotas.max_upload_files, 20)
        self.assertEqual(settings.observability.critical_error_rate, 0.2)


if __name__ == "__main__":
    unittest.main()
