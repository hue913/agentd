"""Approval flow regressions: _ask_human runs on worker threads, not the loop.

The old implementation reached for asyncio.get_running_loop() inside a worker
thread, got RuntimeError, and returned False — silently denying every pending
command. These tests exercise the threading.Event path directly, no HTTP.
"""

from __future__ import annotations

import asyncio
import threading
import time
import types

import pytest

from agentd.runtime import Runtime


@pytest.fixture()
def runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTD_AUTO_APPROVE", raising=False)
    # Safety net only: every test below resolves or times out in well under a
    # second, so a regression can cost 5s, never the 300s default.
    monkeypatch.setenv("AGENTD_APPROVAL_TIMEOUT", "5")
    rt = Runtime({"db": str(tmp_path / "approval.db")})
    yield rt
    rt.close()


def _wait_for_record(pending: dict, timeout_s: float = 5.0) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pending:
            return next(iter(pending))
        time.sleep(0.01)
    raise AssertionError("the approval record never registered")


def _session_scoped_worker(runtime, pending: dict, outcome: dict):
    """What a real session worker thread looks like: run_session binds the
    thread-local current session, then the agent loop calls _ask_human."""

    def worker():
        runtime._local.session = types.SimpleNamespace(pending_approvals=pending)
        outcome["granted"] = runtime._ask_human(
            {"tool": "fs.write", "args": {"path": "/tmp/x"}, "risk": "dangerous"})

    return worker


def test_ask_human_blocks_until_resolved(runtime):
    pending: dict = {}
    outcome: dict = {}
    t = threading.Thread(target=_session_scoped_worker(runtime, pending, outcome))
    t.start()
    token = _wait_for_record(pending)
    assert runtime.resolve_approval(token, True) is True
    t.join(timeout=5)
    assert not t.is_alive(), "_ask_human never returned after approval"
    assert outcome["granted"] is True
    assert token not in pending, "a resolved approval must be cleared from the pending table"


def test_ask_human_returns_false_when_declined(runtime):
    pending: dict = {}
    outcome: dict = {}
    t = threading.Thread(target=_session_scoped_worker(runtime, pending, outcome))
    t.start()
    token = _wait_for_record(pending)
    runtime.resolve_approval(token, False)
    t.join(timeout=5)
    assert outcome["granted"] is False
    assert token not in pending


def _timeout_worker(runtime, pending: dict, outcome: dict):
    def worker():
        runtime._local.session = types.SimpleNamespace(pending_approvals=pending)
        outcome["granted"] = runtime._ask_human({"tool": "t"})

    return worker


def test_orphaned_approval_is_still_resolvable(runtime):
    """An approval raised with NO session bound on the calling thread (e.g. the
    HTTP ssh.exec route) must still register in the process-level pending table
    and be resolvable — the old session-only table made these unresolvable."""
    queue = runtime.subscribe()
    outcome: dict = {}
    t = threading.Thread(target=lambda: outcome.update(granted=runtime._ask_human({"tool": "t"})))
    t.start()
    deadline = time.time() + 5
    event = None
    while time.time() < deadline:
        try:
            event = queue.get_nowait()
            break
        except asyncio.QueueEmpty:
            time.sleep(0.01)
    runtime.unsubscribe(queue)
    assert event is not None and event["type"] == "approval_required", "no approval announced"
    assert runtime.resolve_approval(event["token"], True) is True
    t.join(timeout=5)
    assert outcome["granted"] is True
    assert event["token"] not in runtime.pending_approvals


def test_ask_human_times_out_and_cleans_up(runtime, monkeypatch):
    monkeypatch.setenv("AGENTD_APPROVAL_TIMEOUT", "0.2")
    pending: dict = {}
    outcome: dict = {}
    t = threading.Thread(target=_timeout_worker(runtime, pending, outcome))
    t.start()
    _wait_for_record(pending)
    t.join(timeout=5)
    assert not t.is_alive()
    assert outcome["granted"] is False
    assert not pending, "a timed-out approval must not linger as a phantom card"
    assert not runtime.pending_approvals, "process-level pending table must be cleaned too"


def test_resolve_after_timeout_is_a_clean_miss(runtime, monkeypatch):
    monkeypatch.setenv("AGENTD_APPROVAL_TIMEOUT", "0.2")
    pending: dict = {}
    t = threading.Thread(target=_timeout_worker(runtime, pending, {}))
    t.start()
    _wait_for_record(pending)
    t.join(timeout=5)
    assert not pending
    # a late client approval must land on nothing, not on some other request
    assert runtime.resolve_approval("stale-token", True) is False


def test_auto_approve_env_short_circuits(runtime, monkeypatch):
    monkeypatch.setenv("AGENTD_AUTO_APPROVE", "1")
    assert runtime._ask_human({"tool": "t"}) is True
