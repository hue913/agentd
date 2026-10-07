"""Route-shape regressions that cost real debugging time.

Both cases here returned HTTP 200 with the console's HTML instead of the JSON
or 404 the caller expected, because the console's SPA catch-all
(`/{asset:path}`) was mounted before the ops router and answered anything it did
not recognise.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from agentd.api import create_app
from agentd.api.ops import router as ops_router
from agentd.api.ui import make_ui_router
from agentd.runtime import Runtime

TEST_TOKEN = "route-shape-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def served(tmp_path):
    os.environ["AGENTD_API_TOKEN"] = TEST_TOKEN
    os.environ["AGENTD_UI_DIR"] = str(tmp_path / "ui")
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text("<!doctype html><title>console</title><script src=app.js>")
    (ui_dir / "app.js").write_text("// console")
    (ui_dir / "secret.py").write_text("# must never be served")

    runtime = Runtime({"db": str(tmp_path / "r.db")})
    port = _free_port()
    app = create_app(runtime)
    # The console must be able to shadow the API if mounted in the wrong order.
    # Mounting it here on purpose reproduces the original bug.
    app.include_router(make_ui_router(ui_dir))
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                       log_level="error", lifespan="off"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    for _ in range(80):
        if uv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    uv.should_exit = True
    thread.join(timeout=5)
    runtime.close()
    os.environ.pop("AGENTD_UI_DIR", None)


def _get(url: str, token: str | None = TEST_TOKEN):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode("utf-8", "replace")


def test_console_assets_are_served(served):
    for path in ("/", "/app.js"):
        code, ctype, body = _get(served + path)
        assert code == 200, (path, code)
        assert "console" in body or "// console" in body


def test_spa_fallback_still_works_for_console_routes(served):
    code, ctype, body = _get(served + "/council")
    assert code == 200
    assert "console" in body, "an unknown console route should fall back to the shell"


def test_api_path_is_never_answered_with_html(served):
    """The regression: the catch-all must not swallow the API surface."""
    for path in ("/api/state", "/api/does-not-exist", "/api/credit"):
        code, ctype, body = _get(served + path)
        assert "<!doctype html" not in body.lower(), (path, "console HTML leaked into the API")
        if code != 200:
            assert "json" in ctype, (path, ctype)


def test_unknown_api_path_is_a_json_404(served):
    code, ctype, body = _get(served + "/api/does-not-exist")
    assert code == 404
    assert "json" in ctype
    assert json.loads(body)["detail"]


def test_healthz_is_not_treated_as_a_console_asset(served):
    code, ctype, _ = _get(served + "/healthz", token=None)
    assert code == 200
    assert "json" in ctype


def test_console_never_serves_a_non_allowlisted_extension(served):
    code, _, _ = _get(served + "/secret.py")
    assert code == 403, "a .py file inside the console root must not be served"


def test_traversal_is_refused(served):
    for path in ("/../etc/passwd", "/%2e%2e/%2e%2e/etc/passwd"):
        code, _, body = _get(served + path)
        assert "root:x:0:0" not in body, path
