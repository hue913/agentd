"""Interactive PTY sessions and the ops metrics collector.

The behaviours worth locking down, in priority order:

1. Admission control actually refuses a destructive *purpose*. A PTY is a byte
   stream, so the gate can only judge the declared intent -- which is exactly
   why that check has to be real rather than decorative.
2. Audit records never contain typed bytes (the whole point of the digest).
3. A disconnected socket leaves no live shell behind.
4. sysinfo never lets caller data reach a shell: unit names are validated and
   every argv is a module constant.
5. The ring buffer drops oldest bytes instead of growing without bound.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from agentd import sysinfo
from agentd.envs.pty_env import _Ring, PTYHub, PTYRefused, PTYSession
from agentd.envs.ssh_env import HostSpec, SSHHub
from agentd.safety.audit import AuditLog

# Interactive-session tests need a live `ssh root@127.0.0.1`: without a key the
# spawned ssh exits at once, so any assertion on liveness would fail on a dev
# laptop -- and pass for the wrong reason if we tolerated it. The server (the
# deployment target) carries /root/.ssh/agentd_self_ed25519 and runs these.
_SSH_SELF_KEY = "/root/.ssh/agentd_self_ed25519"
requires_ssh_self = pytest.mark.skipif(
    not os.path.exists(_SSH_SELF_KEY),
    reason=f"no live ssh-to-self (missing {_SSH_SELF_KEY}); run on the server or install the key",
)

# sysinfo reads /proc, systemctl and journalctl by contract -- the ops panel is
# for the Linux server, not for the host that happens to be running the tests.
requires_linux = pytest.mark.skipif(
    sys.platform != "linux",
    reason="reads /proc + systemd + journalctl; the ops panel targets the Linux server",
)


# --- ring buffer -----------------------------------------------------------

def test_ring_drops_oldest_instead_of_growing():
    ring = _Ring(cap=10)
    ring.put(b"0123456789")
    ring.put(b"abc")
    out = ring.drain()
    assert out == b"3456789abc", out
    assert ring.dropped == 3
    assert len(ring) == 0, "drain must empty the buffer"


def test_ring_under_cap_keeps_everything():
    ring = _Ring(cap=100)
    ring.put(b"hello")
    assert ring.drain() == b"hello"
    assert ring.dropped == 0


# --- admission -------------------------------------------------------------

@pytest.fixture()
def hub(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    ssh = SSHHub(audit=audit)
    # The host must authenticate for real: a keyless HostSpec still spawns ssh,
    # which then fails with "Permission denied (publickey)". The PTY exists
    # either way, so a fixture without a key would look like it works.
    key = "/root/.ssh/agentd_self_ed25519"
    spec = (HostSpec(label="self", host="127.0.0.1", user="root", key_path=key)
            if os.path.exists(key) else
            HostSpec(label="self", host="127.0.0.1", user="root"))
    ssh.add_host(spec)
    return PTYHub(ssh, audit=audit, idle_ttl=2), ssh, audit


def test_destructive_purpose_is_refused_before_any_shell_exists(hub):
    pty_hub, _, audit = hub
    with pytest.raises(PTYRefused):
        pty_hub.open("self", purpose="rm -rf / --no-preserve-root")
    assert pty_hub.sessions == {}, "a refused purpose must not leave a session behind"
    levels = [rec.level for rec in _read_audit(audit)]
    assert "block" in levels


@requires_ssh_self
def test_benign_purpose_is_admitted(hub):
    pty_hub, _, _ = hub
    session = pty_hub.open("self", purpose="read the nginx status page")
    try:
        assert session.alive
        assert session.id in pty_hub.sessions
    finally:
        pty_hub.close(session.id)


def _read_audit(audit):
    from agentd.safety.audit import AuditRecord

    return [AuditRecord(**rec) for rec in audit.tail(limit=100)]


# --- lifecycle -------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(30, 100), (4, 20), (999, 9999)])
def test_geometry_is_clamped_to_something_a_terminal_accepts(hub, rows, cols):
    pty_hub, _, _ = hub
    session = pty_hub.open("self", purpose="geometry check", rows=rows, cols=cols)
    try:
        assert 4 <= session.rows <= 500
        assert 20 <= session.cols <= 1000
    finally:
        pty_hub.close(session.id)


@requires_ssh_self
def test_session_survives_a_real_command_and_reports_activity(hub):
    pty_hub, _, _ = hub
    session = pty_hub.open("self", purpose="run one command", rows=30, cols=100)
    try:
        session.write(b"echo AGENTD_TEST_MARKER_$((6*7))\n")
        deadline = time.time() + 15
        seen = b""
        while time.time() < deadline and b"AGENTD_TEST_MARKER_42" not in seen:
            time.sleep(0.1)
            seen += session.read()
        assert b"AGENTD_TEST_MARKER_42" in seen, seen[-300:]
        digest = session.digest()
        assert digest["input_bytes"] > 0 and digest["output_bytes"] > 0
    finally:
        pty_hub.close(session.id)


@requires_ssh_self
def test_digest_carries_no_typed_bytes(hub):
    """Audit must not become a place secrets end up."""
    pty_hub, _, _ = hub
    session = pty_hub.open("self", purpose="digest check")
    try:
        session.write(b"export SUPER_SECRET_TOKEN=hunter2\n")
        time.sleep(0.8)
        session.read()
        blob = repr(session.digest())
        assert "hunter2" not in blob
        assert "SUPER_SECRET_TOKEN" not in blob
        assert session.digest()["input_sha256"]
    finally:
        pty_hub.close(session.id)


def test_closing_a_session_stops_the_child(hub):
    pty_hub, _, _ = hub
    session = pty_hub.open("self", purpose="lifecycle check")
    pid = session.proc.pid
    pty_hub.close(session.id)
    time.sleep(0.6)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert session.id not in pty_hub.sessions


@requires_ssh_self
def test_idle_sessions_are_reaped(hub):
    """A client that vanishes mid-session must not leave a root shell on the box."""
    pty_hub, _, _ = hub
    session = pty_hub.open("self", purpose="idle reaper check")
    session.last_activity = time.time() - 3600
    reaped = pty_hub.reap()
    assert [r["session"] for r in reaped] == [session.id]
    assert "idle" in reaped[0]["exit_reason"]
    assert pty_hub.sessions == {}


@requires_ssh_self
def test_reaper_leaves_live_sessions_alone(hub):
    pty_hub, _, _ = hub
    session = pty_hub.open("self", purpose="still in use")
    try:
        session.read()
        assert pty_hub.reap() == []
        assert session.id in pty_hub.sessions
    finally:
        pty_hub.close(session.id)


def test_get_on_unknown_session_is_an_error(hub):
    pty_hub, _, _ = hub
    with pytest.raises(Exception):
        pty_hub.get("pty-does-not-exist")


# --- sysinfo ---------------------------------------------------------------

@requires_linux
def test_overview_reports_the_running_host():
    data = sysinfo.overview()
    assert data["uptime_s"] > 0
    assert data["mem_total_kb"] > 0
    assert data["disk"], "at least the root filesystem should be reported"
    root = [d for d in data["disk"] if d["mount"] == "/"]
    assert root and root[0]["size"] > 0


@requires_linux
def test_overview_matches_proc_meminfo():
    """The panel must not drift from the kernel's own numbers."""
    data = sysinfo.overview()
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            expected = int(line.split()[1])
            break
    assert data["mem_total_kb"] == expected


@pytest.mark.parametrize("sort", ["rss", "cpu", "etime", "bogus"])
def test_processes_sort_and_clamp(sort):
    data = sysinfo.processes(limit=5, sort=sort)
    rows = data["processes"]
    assert len(rows) <= 5
    assert all(r["pid"] > 0 and r["comm"] for r in rows)


@requires_linux
def test_services_reports_real_units():
    data = sysinfo.services()
    units = {s["unit"]: s for s in data["services"]}
    # The default unit list must be reported with its shape, everywhere.
    assert "agentd.service" in units
    assert all({"unit", "active", "active_state", "main_pid"} <= set(s) for s in units.values())
    # Where the unit is actually deployed (the agentd box), it must read as
    # active. A bare CI runner legitimately reports it inactive -- asserting the
    # deployment fact there would fail for the wrong reason.
    if os.path.exists("/etc/systemd/system/agentd.service"):
        assert units["agentd.service"]["active"] is True


@pytest.mark.parametrize("hostile", ["x -o cat", "a;rm -rf /", "$(id)", "a b", "../etc", ""])
def test_logs_refuses_a_unit_name_that_could_become_an_option(hostile):
    result = sysinfo.logs(hostile, lines=5)
    assert result["entries"] == []
    assert result["error"] == "invalid unit name"


@requires_linux
def test_logs_clamps_the_line_count():
    assert sysinfo.logs("agentd.service", lines=999999)["lines"] == 2000
    assert sysinfo.logs("agentd.service", lines=0)["lines"] == 1


@requires_linux
def test_snapshot_has_all_three_sections():
    snap = sysinfo.snapshot()
    assert set(snap) == {"overview", "processes", "services"}
    assert snap["overview"]["mem_total_kb"] > 0
