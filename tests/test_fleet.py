"""Fleet: host CRUD, per-host policy tiers, metrics cache, batch exec, health.

Baseline discipline: these tests only ADD coverage — the 346 passing tests
upstream stay untouched. Everything runs against fakes (no real ssh, no real
agentd.json), so the suite is hermetic.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

import agentd.envs.ssh_env as ssh_env
from agentd.envs.ssh_env import ApprovalRequired, ExecResult, HostSpec, SSHError, SSHHub
from agentd.fleet import Fleet, parse_metrics
from agentd.runtime import Runtime
from agentd.safety.audit import AuditLog, AuditRecord

GOOD_PROBE_OUT = ("arch=x86_64\nmem_total_kb=16000000\nmem_avail_kb=6000000\n"
                  "disk_pct=42\nload1=0.53\nprocs=217\n")


# --- fakes -------------------------------------------------------------------

class FakeHub:
    """SSHHub stand-in: real HostSpec handling, scripted exec replies."""

    def __init__(self):
        self._hosts: dict[str, HostSpec] = {}
        self.calls: list[dict] = []
        self.reply = lambda label, command, kw: (0, GOOD_PROBE_OUT, "")

    def add(self, label: str, **fields) -> None:
        spec = HostSpec(label=label, host="10.0.0.9", **fields)
        self._hosts[label] = spec

    def hosts(self) -> list[str]:
        return sorted(self._hosts)

    def get_host(self, label: str) -> HostSpec:
        if label not in self._hosts:
            raise SSHError(f"unknown host '{label}'")
        return self._hosts[label]

    def remove_host(self, label: str) -> None:
        self._hosts.pop(label, None)

    def exec(self, label: str, command: str, timeout: int = 60, approved: bool = False,
             **kw) -> ExecResult:
        self.calls.append({"label": label, "command": command, "approved": approved, **kw})
        rc, out, err = self.reply(label, command, kw)
        return ExecResult(host=label, command=command, rc=rc, stdout=out, stderr=err,
                          level="allow")


@pytest.fixture()
def fleet():
    hub = FakeHub()
    hub.add("web-1")
    hub.add("web-2")
    hub.add("db-1", tags=["db"])
    return Fleet(hub, metrics_ttl=60, health_interval=60)


# --- parse_metrics ------------------------------------------------------------

def test_parse_metrics_extracts_all_fields():
    metrics = parse_metrics(GOOD_PROBE_OUT)
    assert metrics == {"arch": "x86_64", "mem_total": 16000000, "mem_used": 10000000,
                       "disk_pct": 42, "load1": 0.53, "procs": 217}


def test_parse_metrics_returns_none_without_usable_data():
    assert parse_metrics("") is None
    assert parse_metrics("Last login: Tue ... \nWelcome.") is None


def test_parse_metrics_tolerates_partial_output():
    # a busybox host without /proc/loadavg still reports what it has
    metrics = parse_metrics("arch=aarch64\nmem_total_kb=100\n")
    assert metrics["arch"] == "aarch64" and metrics["mem_total"] == 100
    assert metrics["disk_pct"] == -1 and metrics["procs"] == 0


# --- metrics cache (F2+F3) ------------------------------------------------------

def test_metrics_cached_within_ttl(fleet):
    assert fleet.metrics("web-1")["metrics"]["arch"] == "x86_64"
    assert fleet.metrics("web-1")["online"] is True
    assert len(fleet.hub.calls) == 1, "second read inside the TTL must not re-probe"
    fleet.metrics("web-1", force=True)
    assert len(fleet.hub.calls) == 2, "force must bypass the cache"


def test_metrics_failure_is_honest(fleet):
    fleet.hub.reply = lambda *a: (255, "", "ssh: connect to host 10.0.0.9 port 22: Connection refused")
    report = fleet.metrics_report("web-1")
    assert report["online"] is False
    assert report["metrics"] is None
    assert "Connection refused" in report["unreachable_reason"]
    assert report["ts"] > 0


def test_metrics_timeout_reason_is_short(fleet):
    fleet.hub.reply = lambda *a: (124, "", "timed out after 8s")
    report = fleet.metrics_report("web-1")
    assert report["online"] is False and report["unreachable_reason"] == "timed out"


def test_metrics_unparseable_output_reports_no_data(fleet):
    fleet.hub.reply = lambda *a: (0, "Welcome to Ubuntu 22.04\n", "")
    report = fleet.metrics_report("web-1")
    assert report["unreachable_reason"] == "probe returned no usable data"


def test_unknown_host_is_offline_not_crash(fleet):
    report = fleet.metrics_report("ghost")
    assert report["online"] is False and report["unreachable_reason"] == "unknown host"


def test_builtin_probe_runs_approved_and_audited(tmp_path, fleet):
    audit = AuditLog(tmp_path / "audit.jsonl")
    fleet.audit = audit
    fleet.metrics("web-1")
    call = fleet.hub.calls[0]
    assert call["approved"] is True, "the constant probe must bypass the gate"
    assert call["action"] == "readonly-probe", "probes must be marked in the audit trail"


def test_custom_metrics_cmd_needs_approval_when_not_readonly(tmp_path, fleet):
    audit = AuditLog(tmp_path / "audit.jsonl")
    fleet.audit = audit
    fleet.hub.get_host("web-1").metrics_cmd = "ps aux > /tmp/x"   # write: needs approval
    report = fleet.metrics_report("web-1")
    assert report["online"] is False
    assert "requires approval" in report["unreachable_reason"]
    assert fleet.hub.calls == [], "a denied collector must never reach the host"
    rows = audit.tail(limit=10)
    assert any(r["host"] == "web-1" and "approval" in (r.get("error") or "") for r in rows)


def test_custom_metrics_cmd_runs_after_one_shot_approval(fleet):
    seen = []

    def approver(payload):
        seen.append(payload)
        return True

    fleet.approver = approver
    fleet.hub.get_host("web-1").metrics_cmd = "cat /proc/meminfo > /tmp/mem.snapshot"
    report = fleet.metrics_report("web-1")
    assert report["online"] is True and report["metrics"]["arch"] == "x86_64"
    assert seen and seen[0]["host"] == "web-1" \
        and "meminfo" in seen[0]["command"]
    call = fleet.hub.calls[0]
    assert call["approved"] is True and call["action"] == "metrics-cmd"


def test_custom_metrics_cmd_blocked_level_stays_blocked(fleet):
    fleet.hub.get_host("web-1").metrics_cmd = "rm -rf /tmp/data"
    report = fleet.metrics_report("web-1")
    assert report["unreachable_reason"] == "custom metrics command blocked by policy"
    assert fleet.hub.calls == []


def test_custom_metrics_cmd_readonly_runs_without_approval(fleet):
    fleet.hub.get_host("web-1").metrics_cmd = "df -P /"   # provably read-only
    report = fleet.metrics_report("web-1")
    assert report["online"] is True
    assert fleet.hub.calls[0]["approved"] is True


# --- health polling (F9) --------------------------------------------------------

def _listener():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    return srv, srv.getsockname()[1]


def test_health_check_tcp_online_and_offline(fleet):
    srv, port = _listener()
    try:
        fleet.hub.get_host("web-1").host = "127.0.0.1"
        fleet.hub.get_host("web-1").port = port
        assert fleet.health_check("web-1")["online"] is True
        fleet.hub.get_host("web-1").port = 1   # nothing listens here
        report = fleet.health_check("web-1")
        assert report["online"] is False and report["unreachable_reason"]
    finally:
        srv.close()


def test_polling_thread_updates_cache_and_stops(fleet):
    srv, port = _listener()
    try:
        fleet.hub.get_host("web-1").host = "127.0.0.1"
        fleet.hub.get_host("web-1").port = port
        fleet.start_polling(interval=0.05)
        deadline = time.time() + 5
        while time.time() < deadline and not fleet.online("web-1"):
            time.sleep(0.02)
        assert fleet.online("web-1"), "poller never marked the host online"
    finally:
        fleet.stop_polling()
        srv.close()
    assert not fleet._poller or not fleet._poller.is_alive()


def test_host_summaries_shape_and_tag_filter(tmp_path, fleet):
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.write(AuditRecord(host="web-1", command="df -h /", level="allow", rc=0))
    fleet.audit = audit
    rows = fleet.host_summaries()
    assert [r["label"] for r in rows] == ["db-1", "web-1", "web-2"]
    row = next(r for r in rows if r["label"] == "web-1")
    for key in ("label", "host", "port", "user", "jump", "policy", "tags", "online",
                "unreachable_reason", "metrics", "metrics_ts", "last_action"):
        assert key in row, f"contract field '{key}' missing"
    assert row["policy"] == "standard" and row["last_action"]["command"] == "df -h /"
    assert [r["label"] for r in fleet.host_summaries(tag="db")] == ["db-1"]
    assert fleet.host_summaries(tag="nope") == []


# --- per-host policy tiers (F4) --------------------------------------------------

@pytest.fixture()
def hub(tmp_path):
    hub = SSHHub(audit=AuditLog(tmp_path / "audit.jsonl"))
    hub.add_host(HostSpec(label="web-1", host="10.0.0.1"))
    hub.add_host(HostSpec(label="web-2", host="10.0.0.2"))
    hub.add_host(HostSpec(label="strict-1", host="10.0.0.3", policy="strict"))
    hub.add_host(HostSpec(label="perm-1", host="10.0.0.4", policy="permissive"))
    return hub


class _Proc:
    def __init__(self, rc, out, err):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _patch_ssh(monkeypatch, behavior):
    """Replace subprocess.run inside ssh_env: behavior(argv) -> (rc, out, err)."""
    monkeypatch.setattr(ssh_env.subprocess, "run",
                        lambda argv, **kw: _Proc(*behavior(argv)))


CONFIRM_CMD = "chmod -R 0777 /tmp/share"        # gate: confirm
BLOCK_CMD = "rm -rf /tmp/data"                  # gate: block
WRITE_UNPROVEN = "touch /tmp/marker"            # gate: allow, not provably read-only
READONLY_CMD = "ls /"                           # gate: allow + provably read-only


def test_standard_confirm_needs_approval_and_audits(hub, monkeypatch):
    _patch_ssh(monkeypatch, lambda argv: (0, "ok", ""))
    with pytest.raises(ApprovalRequired):
        hub.exec("web-1", CONFIRM_CMD)
    rows = hub.audit.tail(limit=10)
    assert rows and rows[-1]["level"] == "confirm" and rows[-1]["error"] == "awaiting approval"


def test_standard_confirm_runs_once_granted(hub, monkeypatch):
    _patch_ssh(monkeypatch, lambda argv: (0, "ran", ""))
    payloads = []
    hub.approver = lambda payload: payloads.append(payload) or True
    result = hub.exec("web-1", CONFIRM_CMD)
    assert result.ok and result.stdout == "ran"
    assert payloads[0]["host"] == "web-1"
    assert payloads[0]["policy"] == "standard", "approval payload must carry the host policy"


def test_strict_forces_approval_even_when_the_gate_would_wave_it_through(hub, monkeypatch):
    _patch_ssh(monkeypatch, lambda argv: (0, "ran", ""))
    # touch is gate-allow, and an inline approver would grant it — strict must
    # still raise so approval only happens through the explicit two-phase flow.
    hub.approver = lambda payload: True
    with pytest.raises(ApprovalRequired):
        hub.exec("strict-1", WRITE_UNPROVEN, approved=False)
    with pytest.raises(ApprovalRequired):
        hub.exec("strict-1", CONFIRM_CMD, approved=False)
    assert hub.audit.tail(limit=10), "strict refusals must be audited"
    # provably read-only commands run without ceremony
    assert hub.exec("strict-1", READONLY_CMD).ok
    # after the human approves (approved=True re-run), the command executes
    assert hub.exec("strict-1", WRITE_UNPROVEN, approved=True).ok


def test_permissive_autoruns_confirm_but_audits_it(hub, monkeypatch):
    _patch_ssh(monkeypatch, lambda argv: (0, "ran", ""))
    result = hub.exec("perm-1", CONFIRM_CMD)   # no approver configured at all
    assert result.ok
    rows = hub.audit.tail(limit=10)
    assert any("policy=permissive auto-approved" in (r.get("action") or "") for r in rows)


def test_permissive_autoruns_unproven_write_but_audits_it(hub, monkeypatch):
    _patch_ssh(monkeypatch, lambda argv: (0, "ran", ""))
    assert hub.exec("perm-1", WRITE_UNPROVEN).ok
    assert any("policy=permissive auto-approved" in (r.get("action") or "")
               for r in hub.audit.tail(limit=10))


def test_permissive_still_blocks_block_level(hub, monkeypatch):
    _patch_ssh(monkeypatch, lambda argv: (0, "ran", ""))
    with pytest.raises(ApprovalRequired):
        hub.exec("perm-1", BLOCK_CMD)
    rows = hub.audit.tail(limit=10)
    assert rows[-1]["level"] == "block" and rows[-1]["error"] == "blocked by gate"


def test_unknown_policy_value_falls_back_to_standard(tmp_path, monkeypatch):
    hub = SSHHub(audit=AuditLog(tmp_path / "a.jsonl"))
    hub.add_host(HostSpec(label="weird", host="10.0.0.5", policy="yolo"))
    _patch_ssh(monkeypatch, lambda argv: (0, "ran", ""))
    with pytest.raises(ApprovalRequired):
        hub.exec("weird", CONFIRM_CMD)   # standard behaviour, not permissive


# --- batch execution (F6+F7+F8) ---------------------------------------------------

@pytest.fixture()
def batch_fleet():
    hub = FakeHub()
    hub.add("web-1")
    hub.add("web-2")
    hub.add("db-1", tags=["db"])
    return Fleet(hub, metrics_ttl=60)


def test_exec_many_requires_approval_unconditionally(batch_fleet):
    report = batch_fleet.exec_many(["web-1", "web-2"], "uptime")
    assert report["blocked"] is True
    assert batch_fleet.hub.calls == [], "a refused batch must not touch any host"


def test_exec_many_parallel_all_hosts(batch_fleet):
    batch_fleet.approver = lambda payload: True
    report = batch_fleet.exec_many(["web-1", "web-2"], "uptime")
    assert report["stopped_at"] is None
    assert [r["label"] for r in report["results"]] == ["web-1", "web-2"]
    assert all(r["ok"] and r["rc"] == 0 for r in report["results"])
    # every per-host exec rides the approved path (the batch approval already
    # happened once) and its audit lands through the normal exec route
    assert all(c["approved"] is True for c in batch_fleet.hub.calls)


def test_exec_many_parallel_survives_a_dead_host(batch_fleet):
    batch_fleet.approver = lambda payload: True

    def reply(label, command, kw):
        if label == "web-2":
            raise SSHError("ssh died")
        return (0, GOOD_PROBE_OUT, "")

    batch_fleet.hub.reply = reply
    report = batch_fleet.exec_many(["web-1", "web-2"], "uptime")
    assert len(report["results"]) == 2
    dead = next(r for r in report["results"] if r["label"] == "web-2")
    assert dead["ok"] is False and dead["error"]
    assert next(r for r in report["results"] if r["label"] == "web-1")["ok"] is True


def test_exec_many_stdout_tail_is_capped(batch_fleet):
    batch_fleet.approver = lambda payload: True
    batch_fleet.hub.reply = lambda *a: (0, "x" * 9000, "")
    row = batch_fleet.exec_many(["web-1"], "bigcat")["results"][0]
    assert len(row["stdout_tail"]) == 2048


def test_exec_many_rolling_stops_at_first_failure(batch_fleet):
    batch_fleet.approver = lambda payload: True

    def reply(label, command, kw):
        return (3, "", "nope") if label == "web-2" else (0, "ok", "")

    batch_fleet.hub.reply = reply
    report = batch_fleet.exec_many(["web-1", "web-2", "db-1"], "uptime", mode="rolling")
    assert report["stopped_at"] == "web-2"
    assert [r["label"] for r in report["results"]] == ["web-1", "web-2"]


def test_exec_many_rolling_honours_order(batch_fleet):
    batch_fleet.approver = lambda payload: True
    report = batch_fleet.exec_many(["web-1", "web-2", "db-1"], "uptime",
                                   mode="rolling", order=["db-1", "web-1", "web-2"])
    assert [c["label"] for c in batch_fleet.hub.calls] == ["db-1", "web-1", "web-2"]


def test_exec_many_tag_expansion(batch_fleet):
    batch_fleet.approver = lambda payload: True
    report = batch_fleet.exec_many(["tag:db"], "uptime")
    assert [r["label"] for r in report["results"]] == ["db-1"]
    with pytest.raises(ValueError):
        batch_fleet.exec_many(["tag:ghost"], "uptime")
    with pytest.raises(ValueError):
        batch_fleet.exec_many(["no-such-host"], "uptime")


def test_exec_many_rejects_bad_input(batch_fleet):
    batch_fleet.approver = lambda payload: True
    with pytest.raises(ValueError):
        batch_fleet.exec_many(["web-1"], "")
    with pytest.raises(ValueError):
        batch_fleet.exec_many(["web-1"], "uptime", mode="swarm")
    with pytest.raises(ValueError):
        batch_fleet.exec_many([], "uptime")


def test_exec_many_denial_is_audited(batch_fleet, tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    batch_fleet.audit = audit
    batch_fleet.exec_many(["web-1"], "uptime")
    rows = audit.tail(limit=10)
    assert any(r.get("action") == "fleet.exec-many" for r in rows)


def test_exec_many_duplicate_retry_shares_one_approval_and_result(batch_fleet):
    """The console re-POSTs while a batch parks in the approval queue; a
    duplicate must join the in-flight request, not stack a second approval
    card (approving that stale card would execute the batch twice)."""
    approvals: list[dict] = []
    human_decided = threading.Event()

    def approver(payload):
        approvals.append(payload)
        human_decided.wait(5)
        return True

    batch_fleet.approver = approver
    outcomes: list[dict] = []

    def caller():
        outcomes.append(batch_fleet.exec_many(["web-1", "web-2"], "uptime"))

    first = threading.Thread(target=caller)
    first.start()
    time.sleep(0.2)                       # let the owner park in the approver
    second = threading.Thread(target=caller)
    second.start()
    time.sleep(0.3)                       # let the joiner reach the shared wait
    human_decided.set()
    first.join(10)
    second.join(10)

    assert len(approvals) == 1, "duplicate retry must not raise a second approval"
    assert len(outcomes) == 2 and outcomes[0] == outcomes[1]
    assert outcomes[0]["stopped_at"] is None
    # one execution per host, not one per request
    assert sorted(c["label"] for c in batch_fleet.hub.calls) == ["web-1", "web-2"]


def test_exec_many_joiner_receives_declined_outcome(batch_fleet):
    batch_fleet.approver = lambda payload: False
    owner = batch_fleet.exec_many(["web-1"], "uptime")
    assert owner["blocked"] is True
    # after the owner finished, the record is popped: a later retry starts a
    # fresh approval cycle instead of replaying the stale refusal forever
    later = batch_fleet.exec_many(["web-1"], "uptime")
    assert later["blocked"] is True
    assert batch_fleet.hub.calls == []


# --- Runtime host CRUD (F1) --------------------------------------------------------

@pytest.fixture()
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTD_CONFIG", str(tmp_path / "agentd.json"))
    cfg = {
        "db": str(tmp_path / "fleet.db"),
        "default_provider": "x",
        "providers": {"x": {"kind": "openai_compat", "model": "m", "base_url": "http://x/v1"}},
        "ssh_hosts": [{"label": "self", "host": "127.0.0.1"}],
    }
    (tmp_path / "agentd.json").write_text(json.dumps(cfg), encoding="utf-8")
    runtime = Runtime(dict(cfg))
    yield runtime
    runtime.close()


def test_add_host_hot_and_persisted(rt, tmp_path):
    out = rt.add_host({"label": "web-1", "host": "10.1.1.1", "port": 2222,
                       "policy": "strict", "tags": ["web", "prod"]})
    assert out["policy"] == "strict" and out["tags"] == ["web", "prod"]
    assert rt.ssh.get_host("web-1").port == 2222, "must be hot-usable without a restart"

    cfg = json.loads((tmp_path / "agentd.json").read_text())
    assert cfg["providers"]["x"]["model"] == "m", "other top-level fields must survive"
    assert [h["label"] for h in cfg["ssh_hosts"]] == ["self", "web-1"]
    assert (tmp_path / "agentd.json.bak").exists(), "save_config keeps one backup"


def test_add_host_validates(rt):
    for body, needle in (
        ({"label": "", "host": "h"}, "label"),
        ({"label": "1bad", "host": "h"}, "label"),
        ({"label": "a b", "host": "h"}, "label"),
        ({"label": "self", "host": "h"}, "reserved"),
        ({"label": "ok1", "host": ""}, "host"),
        ({"label": "ok1", "host": "h", "port": 0}, "port"),
        ({"label": "ok1", "host": "h", "port": "x"}, "port"),
        ({"label": "ok1", "host": "h", "policy": "yolo"}, "policy"),
        ({"label": "ok1", "host": "h", "tags": "web"}, "tags"),
        ({"label": "ok1", "host": "h", "jump": "ghost"}, "jump"),
    ):
        with pytest.raises(ValueError, match=needle):
            rt.add_host(body)


def test_add_host_duplicate_label(rt):
    rt.add_host({"label": "web-1", "host": "10.0.0.1"})
    with pytest.raises(ValueError, match="already exists"):
        rt.add_host({"label": "web-1", "host": "10.0.0.2"})


def test_update_host_partial_and_hot(rt):
    rt.add_host({"label": "web-1", "host": "10.0.0.1", "tags": ["web"]})
    out = rt.update_host("web-1", {"policy": "strict", "port": 2200})
    assert out["policy"] == "strict" and out["tags"] == ["web"], "untouched fields survive"
    assert rt.ssh.get_host("web-1").policy == "strict", "policy must be hot-effective"
    with pytest.raises(SSHError):
        rt.update_host("ghost", {"policy": "strict"})
    with pytest.raises(ValueError, match="unknown field"):
        rt.update_host("web-1", {"nonsense": 1})


def test_remove_host_rules(rt):
    rt.add_host({"label": "web-1", "host": "10.0.0.1"})
    with pytest.raises(ValueError, match="self"):
        rt.remove_host("self")
    with pytest.raises(KeyError):
        rt.remove_host("ghost")
    assert rt.remove_host("web-1")["ok"] is True
    assert "web-1" not in rt.ssh.hosts()
    # last-host guard: a fleet reduced to a single (non-self) host keeps it
    rt.config["ssh_hosts"] = [{"label": "solo", "host": "10.9.9.9"}]
    rt.ssh.add_host(HostSpec(label="solo", host="10.9.9.9"))
    with pytest.raises(ValueError, match="last host"):
        rt.remove_host("solo")


def test_runtime_state_hosts_carry_policy_and_tags(rt):
    rt.add_host({"label": "web-1", "host": "10.0.0.1", "policy": "strict", "tags": ["web"]})
    hosts = {h["label"]: h for h in rt.state()["hosts"]}
    assert hosts["web-1"]["policy"] == "strict"
    assert hosts["web-1"]["tags"] == ["web"]
    assert "online" in hosts["web-1"]


# --- HTTP API (F1 contract) ----------------------------------------------------------

API_TOKEN = "fleet-test-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server(tmp_path, monkeypatch):
    import uvicorn
    from agentd.api import create_app

    monkeypatch.setenv("AGENTD_API_TOKEN", API_TOKEN)
    monkeypatch.setenv("AGENTD_CONFIG", str(tmp_path / "agentd.json"))
    monkeypatch.setenv("AGENTD_APPROVAL_TIMEOUT", "3")
    monkeypatch.delenv("AGENTD_AUTO_APPROVE", raising=False)
    # A reserved "self" entry ships by default, so the delete-guards are
    # exercisable over HTTP without touching the real machine.
    runtime = Runtime({"db": str(tmp_path / "api.db"),
                       "ssh_hosts": [{"label": "self", "host": "127.0.0.1"}]})
    port = _free_port()
    uv = uvicorn.Server(uvicorn.Config(create_app(runtime), host="127.0.0.1", port=port,
                                       log_level="error", lifespan="off"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    for _ in range(80):
        if uv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", runtime
    uv.should_exit = True
    thread.join(timeout=5)
    runtime.close()


def _get(url: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _send(url: str, payload: dict, method: str = "POST"):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method=method,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {API_TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _delete(url: str):
    req = urllib.request.Request(url, method="DELETE",
                                 headers={"Authorization": f"Bearer {API_TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def test_api_host_crud_roundtrip(server):
    base, _ = server
    code, body = _send(f"{base}/api/hosts",
                       {"label": "web-1", "host": "10.1.1.1", "tags": ["web"],
                        "policy": "strict"})
    assert code == 201 and body["host"]["label"] == "web-1"
    code, _ = _send(f"{base}/api/hosts", {"label": "web-2", "host": "10.1.1.2"})

    code, body = _get(f"{base}/api/hosts")
    assert code == 200
    row = next(h for h in body["hosts"] if h["label"] == "web-1")
    for key in ("label", "host", "port", "user", "jump", "policy", "tags", "online",
                "unreachable_reason", "metrics", "metrics_ts", "last_action"):
        assert key in row
    assert row["policy"] == "strict" and row["tags"] == ["web"] and row["online"] is False

    code, body = _send(f"{base}/api/hosts/web-1", {"policy": "permissive"}, method="PATCH")
    assert code == 200 and body["host"]["policy"] == "permissive"

    code, body = _get(f"{base}/api/hosts?tag=web")
    assert code == 200 and len(body["hosts"]) == 1
    code, body = _get(f"{base}/api/hosts?tag=absent")
    assert code == 200 and body["hosts"] == []

    code, body = _send(f"{base}/api/hosts", {"label": "web-1", "host": "x"})
    assert code == 400 and "already exists" in str(body)

    code, body = _delete(f"{base}/api/hosts/web-1")
    assert code == 200 and body["ok"] is True
    code, body = _delete(f"{base}/api/hosts/self")
    assert code == 400
    code, body = _delete(f"{base}/api/hosts/ghost")
    assert code == 404


def test_api_patch_unknown_host_is_404(server):
    base, _ = server
    code, _ = _send(f"{base}/api/hosts/ghost", {"policy": "strict"}, method="PATCH")
    assert code == 404


def test_api_metrics_endpoint(server, monkeypatch):
    base, rt = server
    rt.add_host({"label": "web-1", "host": "10.1.1.1"})
    monkeypatch.setattr(rt.ssh, "exec", lambda *a, **kw: ExecResult(
        host=a[0], command=a[1], rc=0, stdout=GOOD_PROBE_OUT, stderr="", level="allow"))
    code, body = _get(f"{base}/api/hosts/web-1/metrics")
    assert code == 200
    assert body["online"] is True and body["metrics"]["arch"] == "x86_64"
    assert body["unreachable_reason"] is None and body["ts"] > 0
    code, _ = _get(f"{base}/api/hosts/ghost/metrics")
    assert code == 404


def test_api_probe_endpoint_uses_hub(server, monkeypatch):
    base, rt = server
    rt.add_host({"label": "web-1", "host": "10.1.1.1"})
    monkeypatch.setattr(rt.ssh, "probe", lambda label, timeout=25: {"host": label, "rc": 0})
    code, body = _send(f"{base}/api/hosts/web-1/probe", {})
    assert code == 200 and body["host"] == "web-1"
    code, _ = _send(f"{base}/api/hosts/ghost/probe", {})
    assert code == 404


def test_api_exec_many_blocked_then_approved_by_human(server, monkeypatch):
    base, rt = server
    rt.add_host({"label": "web-1", "host": "10.1.1.1"})
    rt.add_host({"label": "web-2", "host": "10.1.1.2"})
    monkeypatch.setattr(
        rt.ssh, "exec",
        lambda label, command, timeout=60, approved=False, **kw: ExecResult(
            host=label, command=command, rc=0, stdout="up", stderr="", level="allow"))

    # the request thread parks in the human-approval queue until we resolve it
    outcome: dict = {}

    def caller():
        outcome["reply"] = _send(f"{base}/api/hosts/exec-many",
                                 {"labels": ["web-1", "web-2"], "command": "uptime",
                                  "mode": "parallel"})

    thread = threading.Thread(target=caller)
    thread.start()
    deadline = time.time() + 5
    token = ""
    while time.time() < deadline:
        pending = list(rt.pending_approvals)
        if pending:
            token = pending[0]
            break
        time.sleep(0.02)
    assert token, "exec-many must always raise an approval request"
    payload = rt.pending_approvals[token]["payload"]
    assert payload["labels"] == ["web-1", "web-2"] and payload["command"] == "uptime"

    _send(f"{base}/api/approve", {"token": token, "approved": False})
    thread.join(timeout=10)
    assert outcome["reply"][0] == 202 and outcome["reply"][1]["blocked"] is True

    # second attempt, approved this time
    outcome2: dict = {}

    def caller2():
        outcome2["reply"] = _send(f"{base}/api/hosts/exec-many",
                                  {"labels": ["web-1", "web-2"], "command": "uptime"})

    thread = threading.Thread(target=caller2)
    thread.start()
    deadline = time.time() + 5
    token = ""
    while time.time() < deadline:
        pending = list(rt.pending_approvals)
        if pending:
            token = pending[0]
            break
        time.sleep(0.02)
    _send(f"{base}/api/approve", {"token": token, "approved": True})
    thread.join(timeout=10)
    code, body = outcome2["reply"]
    assert code == 200 and body["stopped_at"] is None
    assert [r["label"] for r in body["results"]] == ["web-1", "web-2"]
    assert all(r["ok"] for r in body["results"])


def test_api_exec_many_validates_input(server):
    base, _ = server
    code, body = _send(f"{base}/api/hosts/exec-many", {"labels": [], "command": "ls"})
    assert code == 400
    code, body = _send(f"{base}/api/hosts/exec-many", {"labels": ["x"], "command": ""})
    assert code == 400
    code, body = _send(f"{base}/api/hosts/exec-many",
                       {"labels": ["ghost"], "command": "ls", "mode": "parallel"})
    assert code == 400 and "unknown host" in str(body)
