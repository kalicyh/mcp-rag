from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

from mcp_rag.config import ProviderBudgetRule, ProviderBudgetSettings, Settings
from mcp_rag.services.runtime import ServiceRuntime


@dataclass
class _FakeLLM:
    label: str

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"{self.label}:generate"

    async def summarize(self, content: str, query: str) -> str:
        return f"{self.label}:summarize"


@dataclass
class _FakeEmbeddingModel:
    label: str

    async def initialize(self) -> None:
        return None

    async def encode(self, texts):
        return [[float(len(text))] for text in texts]

    async def encode_single(self, text: str):
        return [float(len(text))]


class ProviderBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_budget_falls_back_after_limit(self) -> None:
        settings = Settings().model_copy(
            update={
                "llm_provider": "doubao",
                "llm_api_key": "primary-key",
                "llm_fallback_provider": "ollama",
                "provider_budget": ProviderBudgetSettings(
                    enabled=True,
                    llm=ProviderBudgetRule(
                        requests_per_window=1,
                        window_seconds=60,
                        burst=0,
                        failure_threshold=2,
                        cooldown_seconds=60,
                    ),
                ),
            }
        )
        primary = _FakeLLM("primary")
        fallback = _FakeLLM("fallback")

        with patch("mcp_rag.services.runtime.get_llm_model", AsyncMock(return_value=primary)):
            runtime = ServiceRuntime(settings_obj=settings)
            with patch.object(runtime, "_build_fallback_llm_model", AsyncMock(return_value=fallback)):
                model = await runtime.ensure_llm_model()
                first = await model.generate("hello")
                second = await model.generate("world")

        self.assertEqual(first, "primary:generate")
        self.assertEqual(second, "fallback:generate")
        self.assertTrue(runtime.readiness_snapshot()["provider_budget"]["llm"]["enabled"])

    async def test_embedding_failure_falls_back_to_secondary_provider(self) -> None:
        settings = Settings().model_copy(
            update={
                "embedding_provider": "zhipu",
                "embedding_fallback_provider": "m3e-small",
                "provider_budget": ProviderBudgetSettings(
                    enabled=True,
                    embeddings=ProviderBudgetRule(
                        requests_per_window=10,
                        window_seconds=60,
                        burst=0,
                        failure_threshold=1,
                        cooldown_seconds=60,
                    ),
                ),
            }
        )
        primary = _FakeEmbeddingModel("primary")
        fallback = _FakeEmbeddingModel("fallback")

        with patch.object(ServiceRuntime, "build_embedding_model", return_value=primary):
            runtime = ServiceRuntime(settings_obj=settings)
            with patch.object(runtime, "_build_fallback_embedding_model", AsyncMock(return_value=fallback)):
                model = await runtime.ensure_embedding_model()
                with patch.object(primary, "encode_single", AsyncMock(side_effect=RuntimeError("boom"))):
                    result = await model.encode_single("hello")

        self.assertEqual(result, [5.0])
        self.assertGreaterEqual(runtime.readiness_snapshot()["provider_budget"]["embeddings"]["consecutive_failures"], 1)

    async def test_reload_settings_resets_budget_state(self) -> None:
        settings = Settings().model_copy(
            update={
                "llm_provider": "doubao",
                "llm_api_key": "primary-key",
                "llm_fallback_provider": "ollama",
                "provider_budget": ProviderBudgetSettings(
                    enabled=True,
                    llm=ProviderBudgetRule(
                        requests_per_window=1,
                        window_seconds=60,
                        burst=0,
                        failure_threshold=1,
                        cooldown_seconds=60,
                    ),
                ),
            }
        )
        primary = _FakeLLM("primary")
        fallback = _FakeLLM("fallback")

        with patch("mcp_rag.services.runtime.get_llm_model", AsyncMock(return_value=primary)):
            runtime = ServiceRuntime(settings_obj=settings)
            with patch.object(runtime, "_build_fallback_llm_model", AsyncMock(return_value=fallback)):
                model = await runtime.ensure_llm_model()
                self.assertEqual(await model.generate("hello"), "primary:generate")
                self.assertEqual(await model.generate("world"), "fallback:generate")

                updated = settings.model_copy(
                    update={
                        "provider_budget": ProviderBudgetSettings(
                            enabled=True,
                            llm=ProviderBudgetRule(
                                requests_per_window=5,
                                window_seconds=60,
                                burst=0,
                                failure_threshold=1,
                                cooldown_seconds=60,
                            ),
                        )
                    }
                )
                await runtime.reload_settings(updated)
                refreshed = await runtime.ensure_llm_model()
                third = await refreshed.generate("again")

        self.assertEqual(third, "primary:generate")


if __name__ == "__main__":
    unittest.main()
