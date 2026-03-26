from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp_rag.config import ConfigManager


class ConfigManagerTests(unittest.TestCase):
    def test_missing_config_uses_defaults_without_eager_file_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            manager = ConfigManager(config_file=str(config_path))

            settings = manager.settings

            self.assertEqual(settings.http_port, 8060)
            self.assertFalse(config_path.exists())

            persisted = manager.ensure_config_file()
            self.assertEqual(persisted.http_port, 8060)
            self.assertTrue(config_path.exists())
    def test_nested_guardrail_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            manager = ConfigManager(config_file=str(config_path))

            self.assertTrue(manager.update_setting("security", {"enabled": True, "allow_anonymous": False, "api_keys": ["alpha"]}))
            self.assertTrue(manager.update_settings({"rate_limit": {"requests_per_window": 15, "window_seconds": 30, "burst": 5}}))
            self.assertTrue(
                manager.update_settings(
                    {
                        "observability": {
                            "warning_error_rate": 0.1,
                            "critical_error_rate": 0.4,
                            "slow_request_ms": 250.0,
                            "latency_window_size": 128,
                        },
                        "embedding_fallback_provider": "m3e-small",
                        "llm_fallback_provider": "ollama",
                    }
                )
            )

            reloaded = ConfigManager(config_file=str(config_path))
            settings = reloaded.settings
            self.assertTrue(settings.security.enabled)
            self.assertFalse(settings.security.allow_anonymous)
            self.assertEqual(settings.security.api_keys, ["alpha"])
            self.assertEqual(settings.rate_limit.requests_per_window, 15)
            self.assertEqual(settings.rate_limit.window_seconds, 30)
            self.assertEqual(settings.rate_limit.burst, 5)
            self.assertEqual(settings.observability.latency_window_size, 128)
            self.assertEqual(settings.embedding_fallback_provider, "m3e-small")
            self.assertEqual(settings.llm_fallback_provider, "ollama")

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["security"]["api_keys"], ["alpha"])
            self.assertEqual(payload["rate_limit"]["burst"], 5)
            self.assertEqual(payload["observability"]["latency_window_size"], 128)

    def test_reload_picks_up_external_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"http_port": 9001, "security": {"enabled": True, "allow_anonymous": False}}),
                encoding="utf-8",
            )
            manager = ConfigManager(config_file=str(config_path))

            self.assertEqual(manager.settings.http_port, 9001)

            config_path.write_text(
                json.dumps({"http_port": 9002, "security": {"enabled": False, "allow_anonymous": True}}),
                encoding="utf-8",
            )
            reloaded = manager.reload()

            self.assertEqual(reloaded.http_port, 9002)
            self.assertFalse(reloaded.security.enabled)

    def test_reload_if_changed_picks_up_external_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            manager = ConfigManager(config_file=str(config_path))
            manager.ensure_config_file()

            payload = manager.get_all_settings()
            payload["http_port"] = 9901
            config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            reloaded = manager.reload_if_changed()

            self.assertIsNotNone(reloaded)
            self.assertEqual(manager.settings.http_port, 9901)

    def test_legacy_qwen_provider_is_normalized_to_aliyun(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "embedding_provider": "qwen",
                        "llm_provider": "qwen",
                        "provider_configs": {
                            "qwen": {
                                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                "model": "text-embedding-v4",
                                "llm_model": "qwen-plus",
                                "embedding_model": "text-embedding-v4",
                                "api_key": "test-key",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            manager = ConfigManager(config_file=str(config_path))

            settings = manager.settings

            self.assertEqual(settings.embedding_provider, "aliyun")
            self.assertEqual(settings.llm_provider, "aliyun")
            self.assertIn("aliyun", settings.provider_configs)
            self.assertNotIn("qwen", settings.provider_configs)
            self.assertEqual(settings.provider_configs["aliyun"].llm_model, "qwen-plus")

    def test_missing_new_builtin_provider_is_merged_into_existing_provider_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider_configs": {
                            "doubao": {
                                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                                "model": "doubao-embedding-text-240715",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            manager = ConfigManager(config_file=str(config_path))

            settings = manager.settings

            self.assertIn("doubao", settings.provider_configs)
            self.assertIn("aliyun", settings.provider_configs)
            self.assertEqual(
                settings.provider_configs["aliyun"].base_url,
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    def test_empty_builtin_provider_fields_fall_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider_configs": {
                            "aliyun": {
                                "base_url": "",
                                "model": "",
                                "llm_model": "",
                                "embedding_model": "",
                                "api_key": "sk-test",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            manager = ConfigManager(config_file=str(config_path))

            aliyun = manager.settings.provider_configs["aliyun"]

            self.assertEqual(aliyun.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertEqual(aliyun.model, "")
            self.assertIsNone(aliyun.llm_model)
            self.assertIsNone(aliyun.embedding_model)
            self.assertEqual(aliyun.api_key, "sk-test")

if __name__ == "__main__":
    unittest.main()
