"""HTTP API against a live uvicorn server on an ephemeral loopback port."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from agentd.api import create_app
from agentd.runtime import Runtime


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server(tmp_path):
    runtime = Runtime({"db": str(tmp_path / "api.db"), "kernel": {"gamma": 0.5}})
    port = free_port()
    uv = uvicorn.Server(uvicorn.Config(create_app(runtime), host="127.0.0.1", port=port,
                                       log_level="error", lifespan="off"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    for _ in range(80):
        if uv.started:
            break
        time.sleep(0.05)
    else:
        uv.should_exit = True
        pytest.skip("uvicorn did not start in time")
    yield f"http://127.0.0.1:{port}", runtime
    uv.should_exit = True
    thread.join(timeout=5)
    runtime.close()


def get(url: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body[:1] in "{[" else body)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


def post(url: str, payload: dict) -> tuple[int, object]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body[:1] in "{[" else body)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


def test_health_and_state(server):
    base, runtime = server
    code, body = get(f"{base}/healthz")
    assert code == 200 and body["ok"] is True
    assert body["kernel"]["gamma"] == 0.5

    code, state = get(f"{base}/api/state")
    assert code == 200
    assert "builtin" in state["tools"]["by_source"]
    assert state["tasks"] and "providers" in state


def test_tools_listing_and_expand(server):
    base, _ = server
    code, body = get(f"{base}/api/tools")
    assert code == 200 and "ssh.exec" in body["names"]
    assert body["catalog_tokens"] > 0
    code, one = get(f"{base}/api/tools?expand=ssh.exec")
    assert code == 200 and "host" in one["parameters"]["properties"]


def test_session_without_provider_is_a_clean_400(server):
    base, _ = server
    code, body = post(f"{base}/api/session", {"task": "nginx-down"})
    assert code == 400
    assert "no model configured" in str(body)


def test_unknown_task_is_404(server):
    base, _ = server
    code, body = post(f"{base}/api/session", {"task": "nope"})
    assert code == 404


def test_demo_session_runs_and_learns(server):
    base, runtime = server
    code, body = post(f"{base}/api/session", {"task": "nginx-down", "provider": "demo",
                                             "learning": True, "max_steps": 6})
    assert code == 200, body
    session_id = body["session"]

    report = None
    for _ in range(120):
        code, state = get(f"{base}/api/session/{session_id}")
        assert code == 200
        if state["status"] in ("success", "failed", "error"):
            report = state
            break
        time.sleep(0.1)
    assert report and report["status"] != "error", report
    assert report["report"]["steps"] >= 1
    assert runtime.store.count_steps() > 0, "an executed episode must land in memory"


def test_memory_endpoints(server):
    base, runtime = server
    code, text = get(f"{base}/api/memory?fmt=md")
    assert code == 200 and "# agentd memory pack" in text

    code, pack = get(f"{base}/api/memory")
    assert code == 200 and pack["format"] == "agentd-memory/1"

    code, report = post(f"{base}/api/memory/import", {"pack": pack})
    assert code == 200 and report["checksum_ok"] is True

    code, risks = get(f"{base}/api/risks")
    assert code == 200 and "risks" in risks

    code, audit = get(f"{base}/api/audit")
    assert code == 200 and "entries" in audit


def test_ssh_unknown_host_is_a_typed_502(server):
    base, _ = server
    code, body = post(f"{base}/api/ssh/exec", {"host": "ghost", "command": "ls"})
    assert code == 502
    assert "unknown host" in str(body)


def test_compact_endpoint_reports_savings(server):
    base, _ = server
    noisy = "\n".join(["button Add to cart"] * 40 + ["data:image/png;base64,AAAA" + "A" * 80])
    code, body = post(f"{base}/api/util/compact", {"text": noisy, "budget_chars": 600})
    assert code == 200
    assert body["kept_chars"] < body["orig_chars"] and body["saved_pct"] > 50


def test_approve_unknown_token_is_404(server):
    base, _ = server
    code, body = post(f"{base}/api/approve", {"token": "nope", "approved": True})
    assert code == 404


def test_sse_stream_opens_with_hello(server):
    base, _ = server
    with urllib.request.urlopen(f"{base}/events", timeout=15) as resp:
        first = resp.readline().decode()
    assert first.startswith("data:")
    assert json.loads(first[5:].strip())["type"] == "hello"
