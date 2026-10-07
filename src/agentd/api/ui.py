"""Serve the console UI from agentd, same-origin with the API.

Why same-origin matters concretely: the PTY stream is a WebSocket and the API
calls use relative paths. Serving the UI from the same origin avoids CORS, lets
sessionStorage scope the token to one tab, and removes any need for a separate
dev server or a CDN.

Path handling is the whole risk here, so:
  * the resolved path must stay inside the root (no ../ escape, no symlink out)
  * only a small extension allowlist is served
  * unknown paths fall through to index.html so the tabbed UI can own routing
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse

ALLOWED_SUFFIXES = {".html", ".js", ".css", ".json", ".svg", ".png", ".ico", ".woff", ".woff2", ".map"}
MAX_BYTES = 8 * 1024 * 1024


def make_ui_router(root: Path):
    from fastapi import APIRouter

    router = APIRouter()
    root = root.resolve()

    if not root.is_dir():
        return None

    @router.get("/", response_class=HTMLResponse)
    def index():
        page = root / "index.html"
        if not page.is_file():
            raise HTTPException(404, "console not installed")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @router.get("/{asset:path}")
    def asset(asset: str):
        # Reject traversal before touching the filesystem. resolve() alone is
        # not enough once symlinks are in play, hence the is_relative_to check.
        candidate = (root / asset).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise HTTPException(400, "path escapes the console root")
        if candidate.suffix and candidate.suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(403, f"{candidate.suffix} is not served")
        if not candidate.is_file():
            # Single-page app: let the shell handle unknown routes.
            page = root / "index.html"
            if page.is_file():
                return HTMLResponse(page.read_text(encoding="utf-8"))
            raise HTTPException(404, "not found")
        if candidate.stat().st_size > MAX_BYTES:
            raise HTTPException(413, "asset too large")
        ctype = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        return FileResponse(candidate, media_type=ctype,
                            headers={"Cache-Control": "no-cache"})

    return router

