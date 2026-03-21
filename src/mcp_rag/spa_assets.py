"""Helpers for serving a prebuilt SPA shell from the backend."""

from __future__ import annotations

from html import escape
from pathlib import Path

SPA_ENTRY_CANDIDATES = (
    "app/index.html",
    "spa/index.html",
    "index.html",
)


def spa_entry_candidates(static_dir: Path) -> tuple[Path, ...]:
    """Return candidate SPA entry paths under the mounted static directory."""

    return tuple((static_dir / candidate).resolve() for candidate in SPA_ENTRY_CANDIDATES)


def resolve_spa_entry(static_dir: Path) -> Path | None:
    """Return the first existing SPA entry file, if any."""

    for candidate in spa_entry_candidates(static_dir):
        if candidate.is_file():
            return candidate
    return None


def render_missing_spa_html(*, static_dir: Path, request_path: str) -> str:
    """Build a clear fallback page when no prebuilt SPA bundle is available."""

    items = "\n".join(
        f"<li><code>{escape(str(candidate))}</code></li>"
        for candidate in spa_entry_candidates(static_dir)
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MCP-RAG SPA Unavailable</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
    }}
    main {{
      max-width: 760px;
      margin: 48px auto;
      padding: 32px;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid #334155;
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(15, 23, 42, 0.45);
    }}
    h1 {{
      margin-top: 0;
      font-size: 28px;
    }}
    p, li {{
      line-height: 1.6;
    }}
    code {{
      background: #111827;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>SPA assets are unavailable</h1>
    <p>The backend is configured to serve a prebuilt SPA entry for <code>{escape(request_path)}</code>, but no entry file was found.</p>
    <p>No frontend build is triggered at startup. Copy a prebuilt bundle into the static directory and expose one of these files:</p>
    <ul>
      {items}
    </ul>
    <p>The existing JSON APIs remain available.</p>
  </main>
</body>
</html>
"""
