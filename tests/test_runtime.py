"""Runtime concurrency and lifecycle: publish safety, session caps, eviction."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agentd.runtime import Runtime, Session


@pytest.fixture()
def runtime(tmp_path):
    rt = Runtime({"db": str(tmp_path / "runtime.db")})
    yield rt
    rt.close()


def test_publish_from_worker_threads_does_not_lose_events(runtime):
    """Two worker threads burst 100 events each; the subscriber must see all 200.

    The subscriber list is mutated under a lock and (with no event loop in this
    test) events are put directly; either way nothing may vanish between
    subscribe, publish and unsubscribe.
    """
    queue = runtime.subscribe()
    errors: list[Exception] = []

    def burst():
        try:
            for i in range(100):
                runtime.publish({"type": "tick", "i": i})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=burst) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    received = []
    while True:
        try:
            received.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    assert len(received) == 200, f"lost {200 - len(received)} events"
    assert runtime.state()["events"]["dropped"] == 0
    runtime.unsubscribe(queue)
    assert queue not in runtime.subscribers


def test_subscribe_captures_the_running_loop(runtime):
    async def on_loop():
        runtime.subscribe()

    asyncio.run(on_loop())
    assert runtime.main_loop is not None


def test_finished_sessions_are_evicted_oldest_first(runtime):
    runtime.max_sessions = 2
    for sid in ("old-1", "old-2"):
        runtime.sessions[sid] = Session(id=sid, task="t", loop=None, status="success")
    runtime.sessions["fresh"] = Session(id="fresh", task="t", loop=None, status="success")
    runtime._prune_sessions()
    # the oldest finished session goes first; the cap holds afterwards
    assert "old-1" not in runtime.sessions
    assert len(runtime.sessions) == 2
    assert "fresh" in runtime.sessions


def test_stale_finished_sessions_expire_by_ttl(runtime):
    stale = Session(id="stale", task="t", loop=None, status="failed")
    stale.created_at = time.time() - (runtime.session_ttl_s + 10)
    runtime.sessions["stale"] = stale
    runtime._prune_sessions()
    assert "stale" not in runtime.sessions


def test_running_sessions_are_never_evicted(runtime):
    runtime.max_sessions = 1
    runtime.sessions["busy"] = Session(id="busy", task="t", loop=None, status="running")
    runtime.sessions["done"] = Session(id="done", task="t", loop=None, status="success")
    runtime._prune_sessions()
    assert "busy" in runtime.sessions, "evicting a running session orphans a live episode"
    # growing past the cap is the documented fallback while slots are busy


def test_session_events_dead_field_is_gone():
    import dataclasses

    names = {f.name for f in dataclasses.fields(Session)}
    assert "events" not in names


def test_run_session_keeps_current_session_thread_local(runtime):
    """Two concurrent run_session calls must each see their own session in
    _ask_human's context lookup, not whichever session set the pointer last."""

    class FakeReport:
        success = True
        score = 1.0

        def as_dict(self):
            return {"ok": True}

    observed = {}

    def fake_loop_for(session):
        class FakeLoop:
            def run(self, task):
                observed[session.id] = runtime._current_session()
                return FakeReport()
        return FakeLoop()

    for sid in ("s1", "s2"):
        session = Session(id=sid, task="nginx-down", loop=None)
        session.loop = fake_loop_for(session)
        runtime.sessions[sid] = session

    def run(sid):
        runtime.run_session(runtime.sessions[sid])

    threads = [threading.Thread(target=run, args=(sid,)) for sid in ("s1", "s2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert observed["s1"] is runtime.sessions["s1"]
    assert observed["s2"] is runtime.sessions["s2"]
    assert runtime._current_session() is None
