"""Tests for the P1/P2 hardening round.

Covers: SQLite cross-thread serialisation, MCP read timeout and stderr
draining, capability probe caching (TTL), audit log concurrency/rotation/tail,
the background PTY reaper, the /api/memory/diff path confinement, the mkstemp
askpass, the shared provider post_json, and the log token redaction.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentd.envs.ssh_env import HostSpec, SSHHub, cleanup_askpass
from agentd.kernel.store import Step, Store
from agentd.log import TokenRedactingFilter, get_logger
from agentd.providers.capability import clear_cache, probe_capabilities
from agentd.providers.http import ProviderHTTPStatus, post_json
from agentd.safety.audit import AuditLog, AuditRecord
from agentd.toolbus.mcp_client import MCPError, MCPServerStdio


# --- store: cross-thread safety (M2) -----------------------------------------

def test_store_survives_concurrent_writers():
    """Two threads committing steps on one shared Store must not interleave
    transactions or raise — the RLock makes one public method one transaction."""
    store = Store(":memory:")
    ep1 = store.start_episode("t1", "t1")
    ep2 = store.start_episode("t2", "t2")

    errors: list[Exception] = []

    def writer(episode_id: int, n: int) -> None:
        try:
            for i in range(n):
                store.add_step(Step(episode_id=episode_id, t=i, state=f"s{i}",
                                    state_fp=f"fp{i}", action=f"a{i}", action_fp=f"a{i}"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(ep1, 50)),
               threading.Thread(target=writer, args=(ep2, 50))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert store.count_steps() == 100
    assert store.steps_for_episode(ep1) and store.steps_for_episode(ep2)


def test_store_opens_an_old_database_and_adds_indexes(tmp_path):
    """An existing database written before the indexes/FK/credit round must
    upgrade in place: no crash, indexes present, credit accounting works."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    # Schema as it looked before idx_episodes_task / idx_risks_episode and the
    # credit columns existed.
    conn.executescript("""
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL, goal TEXT,
            started_at REAL, finished_at REAL, success INTEGER, score REAL,
            analysis TEXT, meta TEXT);
        CREATE TABLE steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER REFERENCES episodes(id) ON DELETE CASCADE,
            t INTEGER NOT NULL, state TEXT NOT NULL, state_fp TEXT NOT NULL,
            action TEXT NOT NULL, action_fp TEXT NOT NULL, scope TEXT, z REAL,
            adv REAL, z_prime REAL, chosen INTEGER, reward REAL, ret REAL);
    """)
    conn.execute("INSERT INTO episodes(task) VALUES('legacy')")
    conn.commit()
    conn.close()

    store = Store(db)  # opens, migrates, creates indexes
    names = {row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    assert "idx_episodes_task" in names and "idx_risks_episode" in names

    episode_id = store.start_episode("legacy", "continue")
    store.add_step(Step(episode_id=episode_id, t=0, state="s", state_fp="fp",
                        action="a", action_fp="a"))
    store.note_outcome(store.max_step_id(), adopted=True)
    assert store.credit_totals()["adopted"] >= 1


# --- MCP client (M5) ----------------------------------------------------------

def test_mcp_request_times_out_instead_of_blocking():
    """A server that never answers must raise MCPError at the deadline, not
    block the calling thread forever on readline()."""
    server = MCPServerStdio(
        "silent", sys.executable,
        ["-c", "import sys, time; sys.stdin.read(); time.sleep(60)"],
        timeout=1,
    )
    t0 = time.monotonic()
    with pytest.raises(MCPError):
        server.start()  # start() itself issues the initialize request
    assert time.monotonic() - t0 < 10, "read timeout must actually fire"


def test_mcp_stderr_is_drained_and_attached_to_errors():
    """stderr=PIPE with no reader deadlocks once the pipe fills; the drain
    thread keeps the last 2KB available for error messages."""
    marker = "AGENTD_STDERR_MARKER"
    payload = "x" * 6000 + marker + "\n"
    server = MCPServerStdio(
        "chatty", sys.executable,
        ["-c", f"import sys; sys.stderr.write({payload!r}); sys.stderr.flush(); "
               f"sys.stdin.read(); sys.exit(1)"],
        timeout=2,
    )
    with pytest.raises(MCPError) as excinfo:
        server.start()
    message = str(excinfo.value)
    assert "closed the pipe" in message or "timed out" in message
    assert marker in message, "the stderr tail must reach the error message"
    assert len(server._stderr_buf) <= 2048 + 4096  # ring is bounded


# --- capability cache (M6) ----------------------------------------------------

@pytest.fixture()
def probe_counter(monkeypatch):
    """Replace the HTTP layer with a counting fake and reset the cache."""
    calls = {"n": 0}

    def fake_post_json(url, *, headers, payload, timeout, max_retries=0):
        calls["n"] += 1
        return {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    from agentd.providers import capability

    monkeypatch.setattr(capability, "post_json", fake_post_json)
    clear_cache()
    return calls


def _spec(**kw):
    from agentd.providers import ProviderSpec

    defaults = dict(label="t", model="m", base_url="http://127.0.0.1:9/v1", api_key="k")
    defaults.update(kw)
    return ProviderSpec(**defaults)


def test_capability_probe_is_cached_within_ttl(probe_counter):
    first = probe_capabilities(_spec(), timeout=1)
    assert first.reachable
    second = probe_capabilities(_spec(), timeout=1)
    assert second.cached is True, "a TTL hit must be reported as cached"
    assert probe_counter["n"] == 1, "no HTTP may happen inside the TTL"


def test_capability_cache_key_includes_api_key(probe_counter):
    probe_capabilities(_spec(api_key="k1"), timeout=1)
    probe_capabilities(_spec(api_key="k2"), timeout=1)
    assert probe_counter["n"] == 2, "a different key is a different endpoint"


def test_capability_cache_respects_clear_and_ttl_expiry(probe_counter, monkeypatch):
    probe_capabilities(_spec(), timeout=1)
    clear_cache()
    probe_capabilities(_spec(), timeout=1)
    assert probe_counter["n"] == 2

    # A fresh probe is not marked cached; an immediate repeat is.
    spec = _spec(api_key="stale")
    assert probe_capabilities(spec, timeout=1).cached is False
    assert probe_counter["n"] == 3
    assert probe_capabilities(spec, timeout=1).cached is True
    assert probe_counter["n"] == 3, "still no HTTP inside the TTL"
    monkeypatch.setattr("agentd.providers.capability.PROBE_TTL_S", 0)
    assert probe_capabilities(spec, timeout=1).cached is False
    assert probe_counter["n"] == 4, "an expired entry is probed again"


# --- audit log (M9) -----------------------------------------------------------

def test_audit_concurrent_appends_keep_jsonl_intact(tmp_path):
    """100+ appends from many threads: every line must parse as JSON."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    def writer(n):
        for i in range(n):
            audit.write(AuditRecord(host="h", command=f"cmd-{i}"))
    threads = [threading.Thread(target=writer, args=(25,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_audit_rotates_to_one_archive_generation(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl", max_bytes=500)
    for i in range(40):
        audit.write(AuditRecord(host="h", command=f"filler-{i}-" + "y" * 20))
    backup = tmp_path / "audit.jsonl.1"
    assert backup.exists(), "crossing max_bytes must rotate to .1"
    assert audit.path.exists()
    # The archive holds the earlier records, all still valid JSONL.
    archived = backup.read_text(encoding="utf-8").splitlines()
    assert len(archived) >= 1
    for line in archived:
        assert json.loads(line)["command"].startswith("filler-")
    # The tail reads the live file and stays chronological.
    rows = audit.tail(limit=5)
    assert 1 <= len(rows) <= 5
    assert all(r["command"].startswith("filler-") for r in rows)
    assert rows == sorted(rows, key=lambda r: r["ts"])


def test_audit_tail_returns_last_n_in_chronological_order(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    for i in range(300):
        audit.write(AuditRecord(host="h" if i % 2 else "other", command=f"c{i}"))
    rows = audit.tail(limit=10)
    assert [r["command"] for r in rows] == [f"c{i}" for i in range(290, 300)]
    rows = audit.tail(limit=10, host="h")
    # newest 10 records for host "h" are the odd i in [281, 299]
    assert [r["command"] for r in rows] == [f"c{i}" for i in range(281, 300, 2)]
    assert audit.blocked_count() == 0


# --- pty reaper (M12) ---------------------------------------------------------

def test_pty_reaper_runs_in_background_and_stops_cleanly():
    from agentd.envs.pty_env import PTYHub

    hub = PTYHub(SSHHub())
    calls = {"n": 0}
    original = hub.reap

    def counting_reap(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    hub.reap = counting_reap
    hub.start_reaper(interval=1)
    assert hub._reaper is not None and hub._reaper.daemon
    # force an immediate pass through the stop path to prove wake-ability
    hub.stop_reaper()
    assert hub._reaper is None
    # a second start/stop cycle works (idempotent start, clean stop)
    hub.start_reaper(interval=1)
    hub.start_reaper(interval=1)  # must not spawn a second thread
    first = hub._reaper
    hub.start_reaper(interval=1)
    assert hub._reaper is first
    hub.stop_reaper()


# --- /api/memory/diff path confinement (M3) -----------------------------------

def test_memory_diff_path_guard():
    from agentd.api.server import _diff_path_allowed

    db = "/var/lib/agentd/memory.db"
    assert _diff_path_allowed(db, "/var/lib/agentd/pack.agentdmem")
    assert _diff_path_allowed(db, "/var/lib/agentd/../agentd/pack.agentdmem")
    assert not _diff_path_allowed(db, "/etc/hostname")
    assert not _diff_path_allowed(db, "/var/lib/agentd")            # the dir itself
    assert not _diff_path_allowed(db, "/var/lib/agentd-secret/x")   # prefix games


# --- askpass (M4) --------------------------------------------------------------

def test_askpass_uses_mkstemp_and_cleans_up(monkeypatch):
    monkeypatch.setenv("AGENTD_SSH_PW_TEST", "pw")
    hub = SSHHub()
    spec = HostSpec(label="h", host="127.0.0.1", password_env="AGENTD_SSH_PW_TEST")
    env = hub._env(spec)
    path = env["SSH_ASKPASS"]
    try:
        assert "agentd-askpass-" in path and path.endswith(".sh")
        assert os.path.exists(path)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o700
        content = open(path, encoding="utf-8").read()
        assert "AGENTD_SSH_PASSWORD" in content
        # concurrent invocations each get their own file
        env2 = hub._env(spec)
        assert env2["SSH_ASKPASS"] != path
        cleanup_askpass(env2["SSH_ASKPASS"])
        assert not os.path.exists(env2["SSH_ASKPASS"])
    finally:
        cleanup_askpass(path)
    assert not os.path.exists(path)
    assert not os.path.exists(os.path.join(os.path.dirname(path), "agentd-askpass.sh"))


# --- shared post_json (L2) -----------------------------------------------------

class _FlakyHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        STATE["n"] += 1
        if STATE["n"] in STATE["fail_on"]:
            body = b'{"error": "transient"}'
            self.send_response(503)
        else:
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


STATE = {"n": 0, "fail_on": set()}


@pytest.fixture()
def flaky_server():
    STATE.update(n=0, fail_on=set())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FlakyHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/x"
    yield url
    httpd.shutdown()


def test_post_json_retries_transient_failures(flaky_server, monkeypatch):
    STATE["fail_on"] = {1, 2}
    monkeypatch.setattr("agentd.providers.http.time.sleep", lambda s: None)
    data = post_json(flaky_server, headers={}, payload={"a": 1}, timeout=5, max_retries=2)
    assert data == {"ok": True}
    assert STATE["n"] == 3


def test_post_json_does_not_retry_client_errors(flaky_server, monkeypatch):
    STATE["fail_on"] = set()
    monkeypatch.setattr("agentd.providers.http.time.sleep", lambda s: None)
    import io
    import urllib.error
    import urllib.request

    from agentd.providers import http as http_mod

    def url_open_400(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {},
                                     io.BytesIO(b"bad key"))

    monkeypatch.setattr(http_mod.urllib.request, "urlopen", url_open_400)
    with pytest.raises(ProviderHTTPStatus) as excinfo:
        post_json(flaky_server, headers={}, payload={}, timeout=5)
    assert excinfo.value.code == 401
    assert STATE["n"] == 0  # no request even reached the (clean) server


# --- log redaction (P2) --------------------------------------------------------

def test_token_redacting_filter_masks_query_and_bearer():
    filt = TokenRedactingFilter()
    record = logging.LogRecord("uvicorn.access", logging.INFO, "x", 1,
                               'GET /events?token=super-secret-123 HTTP/1.1 "Bearer abc.def-ghi"',
                               None, None)
    assert filt.filter(record) is True
    text = record.getMessage()
    assert "super-secret-123" not in text and "abc.def-ghi" not in text
    assert "token=***" in text and "Bearer ***" in text


def test_get_logger_is_idempotent_and_level_is_configurable(monkeypatch):
    monkeypatch.setenv("AGENTD_LOG_LEVEL", "debug")
    logger = get_logger("agentd.test-idem")
    handlers = [h for h in logger.handlers if getattr(h, "_agentd_handler", False)]
    assert len(handlers) == 1
    again = get_logger("agentd.test-idem")
    assert again is logger
    assert logger.level == logging.DEBUG
