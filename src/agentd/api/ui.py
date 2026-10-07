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

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

ALLOWED_SUFFIXES = {".html", ".js", ".css", ".json", ".svg", ".png", ".ico", ".woff", ".woff2", ".map"}
MAX_BYTES = 8 * 1024 * 1024


def _weak_etag(st: object) -> str:
    """Weak validator from (size, mtime): same file content changes -> new tag.

    Weak ("W/") on purpose: two servers or two writes within filesystem
    timestamp granularity are not guaranteed byte-identical, so we only claim
    semantic equivalence, which is all a console asset needs.
    """
    return f'W/"{st.st_size:x}-{st.st_mtime_ns:x}"'


def _etag_matches(if_none_match: str, etag: str) -> bool:
    # If-None-Match can be "*", a single tag, or a comma-separated list.
    if_none_match = if_none_match.strip()
    if if_none_match == "*":
        return True
    return any(candidate.strip() == etag
               for candidate in if_none_match.split(","))

# The API surface must never fall back to the SPA shell: an unknown /api/*
# path has to answer 404 JSON, because a client that asked for an API resource
# and received HTTP 200 with the console's HTML will parse a web page as JSON
# and report a nonsense error far from the cause. Keep this list aligned with
# PROTECTED_PREFIXES / PUBLIC_PATHS in server.py -- anything the API owns gets
# a JSON answer, and only true console routes get the shell.
API_OWNED_PREFIXES = ("api", "events", "healthz")


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
    def asset(asset: str, request: Request):
        for prefix in API_OWNED_PREFIXES:
            if asset == prefix or asset.startswith(prefix + "/"):
                # 404 before the filesystem: these are not console routes even
                # if a file with a coincidental name exists under the root.
                raise HTTPException(404, f"no such API route: /{asset}")
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
        st = candidate.stat()
        if st.st_size > MAX_BYTES:
            raise HTTPException(413, "asset too large")
        # Conditional request: with Cache-Control: no-cache the client still
        # revalidates every time, but an unchanged asset costs a 304 with no
        # body instead of a full re-download.
        etag = _weak_etag(st)
        inm = request.headers.get("if-none-match")
        if inm and _etag_matches(inm, etag):
            return Response(status_code=304,
                            headers={"ETag": etag, "Cache-Control": "no-cache"})
        ctype = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        return FileResponse(candidate, media_type=ctype,
                            headers={"Cache-Control": "no-cache", "ETag": etag})

    return router

