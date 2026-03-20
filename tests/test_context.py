from __future__ import annotations

import unittest
from unittest.mock import Mock

from mcp_rag.context import RequestContext, normalize_request_context, normalize_tenant


class ContextTests(unittest.TestCase):
    def test_normalize_tenant_accepts_legacy_hidden_fields(self) -> None:
        tenant = normalize_tenant(
            {"collection": "docs", "_user_id": "7", "_agent_id": "2"},
            base_collection="fallback",
        )

        self.assertEqual(tenant.base_collection, "docs")
        self.assertEqual(tenant.user_id, 7)
        self.assertEqual(tenant.agent_id, 2)
        self.assertEqual(tenant.canonical_key(), "u7_a2_docs")

    def test_normalize_request_context_preserves_request_and_trace_identity(self) -> None:
        context = normalize_request_context(
            RequestContext(
                tenant=normalize_tenant(base_collection="docs", user_id=7),
                transport="http",
                api_key="secret",
                request_id="req-1",
                trace_id="trace-1",
                client_host="127.0.0.1",
            ),
            base_collection="docs",
        )

        self.assertEqual(context.request_id, "req-1")
        self.assertEqual(context.trace_id, "trace-1")
        self.assertEqual(context.subject_key(fallback="default"), "u7_docs")

    def test_from_http_uses_state_request_ids_when_present(self) -> None:
        request = Mock()
        request.state.request_id = "req-123"
        request.state.trace_id = "trace-123"
        request.headers = {}
        request.client = type("Client", (), {"host": "10.0.0.1"})()

        context = RequestContext.from_http(request, base_collection="docs", user_id=7)

        self.assertEqual(context.request_id, "req-123")
        self.assertEqual(context.trace_id, "trace-123")
        self.assertEqual(context.client_host, "10.0.0.1")
        self.assertEqual(context.tenant.base_collection, "docs")


if __name__ == "__main__":
    unittest.main()
