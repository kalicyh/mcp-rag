from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class PackagingMetadataTests(unittest.TestCase):
    def test_prebuilt_spa_bundle_exists(self) -> None:
        bundle_dir = Path("src/mcp_rag/static/app")
        self.assertTrue((bundle_dir / "index.html").is_file())
        self.assertTrue(any((bundle_dir / "assets").glob("*.js")))
        self.assertTrue(any((bundle_dir / "assets").glob("*.css")))

    def test_setuptools_includes_static_package_data(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        setuptools_cfg = pyproject["tool"]["setuptools"]

        self.assertTrue(setuptools_cfg["include-package-data"])
        self.assertEqual(setuptools_cfg["package-dir"][""], "src")
        self.assertEqual(setuptools_cfg["packages"]["find"]["where"], ["src"])

        package_data = setuptools_cfg["package-data"]["mcp_rag"]
        self.assertIn("static/*", package_data)
        self.assertIn("static/**/*", package_data)

    def test_manifest_includes_static_assets(self) -> None:
        manifest = Path("MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("include README.md", manifest)
        self.assertIn("recursive-include src/mcp_rag/static *", manifest)

    def test_readme_explains_tool_install_and_pnpm_boundary(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("uv tool install mcp-rag", readme)
        self.assertIn("pnpm install", readme)
        self.assertIn("pnpm build", readme)
        self.assertIn("cd frontend", readme)
        self.assertIn("安装用户不需要 Node.js", readme)
        self.assertIn("src/mcp_rag/static/", readme)


if __name__ == "__main__":
    unittest.main()
