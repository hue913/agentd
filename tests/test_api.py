"""HTTP API against a live uvicorn server on an ephemeral loopback port."""

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
from agentd.runtime import Runtime


TEST_TOKEN = "test-token-not-a-real-secret"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server(tmp_path):
    # The API is fail-closed: with no AGENTD_API_TOKEN it returns 503 for every
    # protected route. The suite therefore has to arm a token, exactly as the
    # deployed systemd unit does.
    previous = os.environ.get("AGENTD_API_TOKEN")
    os.environ["AGENTD_API_TOKEN"] = TEST_TOKEN
    # Provider management writes the config file: point it at the tmp dir so a
    # test can never touch the operator's real ~/.config/agentd/agentd.json.
    previous_cfg = os.environ.get("AGENTD_CONFIG")
    os.environ["AGENTD_CONFIG"] = str(tmp_path / "agentd.json")
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
    if previous is None:
        os.environ.pop("AGENTD_API_TOKEN", None)
    else:
        os.environ["AGENTD_API_TOKEN"] = previous
    if previous_cfg is None:
        os.environ.pop("AGENTD_CONFIG", None)
    else:
        os.environ["AGENTD_CONFIG"] = previous_cfg


def get(url: str) -> tuple[int, object]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
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
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {TEST_TOKEN}"})
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


def delete(url: str) -> tuple[int, object]:
    req = urllib.request.Request(url, method="DELETE",
                                 headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
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
    with urllib.request.urlopen(f"{base}/events?token={TEST_TOKEN}", timeout=15) as resp:
        first = resp.readline().decode()
    assert first.startswith("data:")
    assert json.loads(first[5:].strip())["type"] == "hello"


def test_healthz_stays_public(server):
    """Liveness must not require a token, or a misconfigured token wedges systemd."""
    base, _ = server
    code, body = get(f"{base}/healthz")
    assert code == 200 and body["ok"] is True


def test_protected_routes_reject_missing_and_wrong_tokens(server):
    base, _ = server
    for url in (f"{base}/api/state", f"{base}/api/tools", f"{base}/api/risks",
                f"{base}/api/audit", f"{base}/api/memory"):
        req = urllib.request.Request(url)
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError(f"{url} served without a token")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, (url, exc.code)

        bad = urllib.request.Request(url, headers={"Authorization": "Bearer wrong"})
        try:
            urllib.request.urlopen(bad, timeout=10)
            raise AssertionError(f"{url} served a wrong token")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, (url, exc.code)


def test_ssh_exec_is_not_reachable_without_a_token(server):
    base, _ = server
    req = urllib.request.Request(
        f"{base}/api/ssh/exec", data=json.dumps({"host": "x", "command": "id"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("ssh exec reached without a token")
    except urllib.error.HTTPError as exc:
        assert exc.code == 401


def test_missing_server_token_fails_closed(tmp_path):
    """No configured token must NOT mean open access -- it must mean 503."""
    previous = os.environ.pop("AGENTD_API_TOKEN", None)
    try:
        port = free_port()
        uv = uvicorn.Server(uvicorn.Config(create_app(Runtime({"db": str(tmp_path / "nc.db")})),
                                           host="127.0.0.1", port=port,
                                           log_level="error", lifespan="off"))
        thread = threading.Thread(target=uv.run, daemon=True)
        thread.start()
        for _ in range(80):
            if uv.started:
                break
            time.sleep(0.05)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=10) as resp:
                assert resp.status == 200
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10)
                raise AssertionError("served unauthenticated with no token configured")
            except urllib.error.HTTPError as exc:
                assert exc.code == 503
        finally:
            uv.should_exit = True
            thread.join(timeout=5)
    finally:
        if previous is not None:
            os.environ["AGENTD_API_TOKEN"] = previous


# --- model seats -----------------------------------------------------------

def test_provider_seats_crud_and_config_persistence(server, tmp_path):
    base, _ = server
    code, body = get(f"{base}/api/providers")
    assert code == 200 and body["providers"] == []

    code, body = post(f"{base}/api/providers", {
        "name": "lab", "kind": "openai_compat", "model": "qwen3-8b",
        "base_url": "http://127.0.0.1:9/v1", "tier": "cheap",
        "api_key_env": "AGENTD_LAB_KEY"})
    assert code == 200, body
    assert body["name"] == "lab" and "api_key" not in body

    code, body = get(f"{base}/api/providers")
    assert [p["name"] for p in body["providers"]] == ["lab"]
    seat = body["providers"][0]
    assert seat["key_env"] == "AGENTD_LAB_KEY"
    assert seat["key_present"] is False, "env var is not set, the seat must say so"
    assert body["default_provider"] == "lab"

    cfg = json.loads((tmp_path / "agentd.json").read_text())
    assert cfg["providers"]["lab"]["model"] == "qwen3-8b"
    assert cfg["default_provider"] == "lab"

    code, body = delete(f"{base}/api/providers/lab")
    assert code == 200 and body["removed"] == "lab"
    code, body = get(f"{base}/api/providers")
    assert body["providers"] == []


def test_provider_upsert_validates_input(server):
    base, _ = server
    for payload, needle in (
        ({"name": "x", "kind": "weird", "model": "m"}, "unknown kind"),
        ({"name": "bad name!", "model": "m"}, "name must start"),
        ({"name": "ok", "kind": "openai_compat", "model": "m"}, "base_url"),
        ({"name": "ok", "kind": "openai_compat", "model": "", "base_url": "https://x/v1"}, "model"),
    ):
        code, body = post(f"{base}/api/providers", payload)
        assert code == 400, (payload, code, body)
        assert needle in str(body), (payload, body)


def test_provider_delete_unknown_is_404(server):
    base, _ = server
    code, _ = delete(f"{base}/api/providers/ghost")
    assert code == 404


def test_ops_session_needs_a_model_and_refuses_an_empty_goal(server):
    base, _ = server
    code, body = post(f"{base}/api/session", {"task": "check the disk space", "kind": "ops"})
    assert code == 400 and "no model configured" in str(body)

    code, body = post(f"{base}/api/session", {"task": "   ", "kind": "ops", "provider": "demo"})
    assert code == 400 and "goal" in str(body)


def test_ops_session_runs_and_fails_honestly_without_a_usable_model(server):
    """`demo` cannot propose tool calls; the session must say so, not fake success."""
    base, _ = server
    code, body = post(f"{base}/api/session", {"task": "check nginx and report",
                                             "kind": "ops", "provider": "demo",
                                             "max_steps": 3})
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
    assert report is not None
    assert report["status"] == "failed"
    assert "provider" in str(report["report"].get("stopped_by", ""))
