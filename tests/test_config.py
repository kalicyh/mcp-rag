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

            reloaded = ConfigManager(config_file=str(config_path))
            settings = reloaded.settings
            self.assertTrue(settings.security.enabled)
            self.assertFalse(settings.security.allow_anonymous)
            self.assertEqual(settings.security.api_keys, ["alpha"])
            self.assertEqual(settings.rate_limit.requests_per_window, 15)
            self.assertEqual(settings.rate_limit.window_seconds, 30)
            self.assertEqual(settings.rate_limit.burst, 5)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["security"]["api_keys"], ["alpha"])
            self.assertEqual(payload["rate_limit"]["burst"], 5)

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

if __name__ == "__main__":
    unittest.main()
