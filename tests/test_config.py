from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp_rag.config import ConfigManager


class ConfigManagerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
