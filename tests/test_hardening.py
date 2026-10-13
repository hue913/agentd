"""Regressions for the 2026-10-13 review: no self-grant, no finish-as-success,
path confinement, long-option rm, plugin allowlist, council reuse.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from agentd.api import create_app
from agentd.council import Council, TriggerPolicy
from agentd.envs.tool_task import ToolTask
from agentd.kernel import JitRLKernel, Store
from agentd.loop import AgentLoop, LoopConfig
from agentd.providers import MockProvider, ProviderSpec
from agentd.runtime import Runtime
from agentd.safety.gate import classify, initial_readonly
from agentd.toolbus import CallContext, ToolBus, load_plugin_dir, register_builtins
from agentd.toolbus.paths import PathRefused, resolve_allowed


def test_rm_long_options_are_blocked():
    verdict = classify("rm --recursive --force /var/www")
    assert verdict.level == "block"
    assert not initial_readonly("rm --recursive --force /var/www")


def test_interpreter_script_file_is_not_read_only():
    assert not initial_readonly("python3 /tmp/wipe.py")
    assert classify("python3 /tmp/wipe.py").level in ("confirm", "block")
    assert classify("bash -c 'rm -rf /tmp/x'").level in ("confirm", "block")


def test_secret_paths_are_refused(tmp_path):
    with pytest.raises(PathRefused):
        resolve_allowed(str(tmp_path / ".ssh" / "id_ed25519"))
    with pytest.raises(PathRefused):
        resolve_allowed("/etc/shadow")
    with pytest.raises(PathRefused):
        resolve_allowed(str(tmp_path / "agentd.json"), write=True)
    ok = resolve_allowed(str(tmp_path / "notes.txt"))
    assert ok.name == "notes.txt"


def test_fs_read_refuses_ssh_keys(tmp_path):
    bus = ToolBus()
    register_builtins(bus)
    key = tmp_path / ".ssh" / "id_ed25519"
    key.parent.mkdir()
    key.write_text("SECRET", encoding="utf-8")
    # resolve() of a relative .ssh path under tmp still has a .ssh part
    result = bus.call("fs.read", {"path": str(key)}, CallContext())
    assert not result.ok
    assert "refused" in result.error.lower() or "secret" in result.error.lower()


def test_local_exec_asks_the_approver_and_does_not_stick(tmp_path):
    """A granted write must not leave ctx.approved True for the next call."""
    bus = ToolBus()
    register_builtins(bus)
    seen: list[dict] = []
    ctx = CallContext(approver=lambda payload: seen.append(payload) or True)
    target = tmp_path / "once.txt"
    first = bus.call("local.exec", {"command": f"echo once > {target}"}, ctx)
    assert first.ok and target.read_text(encoding="utf-8").strip() == "once"
    assert ctx.approved is False, "approval must not leak onto the episode context"
    assert seen and seen[0]["tool"] == "local.exec"

    # A later dangerous call with the same ctx still has to ask again.
    seen.clear()
    ctx.approver = lambda payload: seen.append(payload) or False
    second = bus.call("local.exec", {"command": f"echo twice >> {target}"}, ctx)
    assert not second.ok
    assert seen, "the second write must re-prompt"


def test_finish_does_not_teach_the_kernel_a_success():
    task = ToolTask(goal_text="do real work", hosts=[])
    store = Store()
    kernel = JitRLKernel(store=store, enabled=True, seed=1)
    bus = ToolBus()
    register_builtins(bus)
    provider = MockProvider(
        ProviderSpec(label="m", model="m", kind="mock"),
        weights={"finish": 0.9},
        critiques=['{"options": ["finish {\\"answer\\": \\"skipped the work\\"}"]}'],
    )
    loop = AgentLoop(kernel, provider, bus, config=LoopConfig(max_steps=2, reflect=False))
    report = loop.run(task)
    assert report.success is False
    assert report.score == 0.0
    steps = store.steps_for_episode(report.episode_id)
    assert steps
    assert all(s.reward <= 0 for s in steps), "finish must not land a +1 in memory"


def test_http_ssh_exec_ignores_client_approved(monkeypatch, tmp_path):
    previous = os.environ.get("AGENTD_API_TOKEN")
    os.environ["AGENTD_API_TOKEN"] = "test-token-not-a-real-secret"
    os.environ["AGENTD_CONFIG"] = str(tmp_path / "agentd.json")
    rt = Runtime({"db": str(tmp_path / "h.db")})
    seen = {}

    def spy(host, command, timeout=60, approved=False, **kw):
        seen["approved"] = approved
        seen["command"] = command
        from agentd.envs.ssh_env import ApprovalRequired
        from agentd.safety.gate import classify
        raise ApprovalRequired(host, command, classify(command))

    monkeypatch.setattr(rt.ssh, "exec", spy)
    import uvicorn, socket, threading, time, json, urllib.request, urllib.error

    def free_port():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port = free_port()
    uv = uvicorn.Server(uvicorn.Config(create_app(rt), host="127.0.0.1", port=port,
                                       log_level="error", lifespan="off"))
    t = threading.Thread(target=uv.run, daemon=True)
    t.start()
    for _ in range(80):
        if uv.started:
            break
        time.sleep(0.05)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ssh/exec",
            data=json.dumps({"host": "ghost", "command": "rm -rf /", "approved": True}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer test-token-not-a-real-secret"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                code = resp.status
        except urllib.error.HTTPError as exc:
            code = exc.code
        assert seen.get("approved") is False, "HTTP must never forward client approved=true"
        assert code in (202, 502)
    finally:
        uv.should_exit = True
        t.join(timeout=5)
        rt.close()
        if previous is None:
            os.environ.pop("AGENTD_API_TOKEN", None)
        else:
            os.environ["AGENTD_API_TOKEN"] = previous


def test_plugin_allowlist_rejects_unknown(tmp_path):
    bus = ToolBus()
    plugin = tmp_path / "good.py"
    plugin.write_text(
        "from agentd.toolbus.spec import ToolSpec, ToolResult\n"
        "TOOLS = [ToolSpec(name='x.ping', description='p',\n"
        "  parameters={'type':'object','properties':{}}, handler=lambda **k: ToolResult(ok=True))]\n",
        encoding="utf-8",
    )
    report = load_plugin_dir(bus, tmp_path, allow_hashes={"nope.py": "0" * 64})
    assert "good.py" in report["errors"]
    assert bus.get("x.ping") is None

    digest = hashlib.sha256(plugin.read_bytes()).hexdigest()
    report = load_plugin_dir(bus, tmp_path, allow_hashes={"good.py": digest})
    assert report["loaded"]
    assert bus.get("x.ping") is not None


def test_council_reuses_the_primary_choose():
    a = MockProvider(ProviderSpec(label="a", model="a", kind="mock"),
                     weights={"restart": 0.51, "reboot": 0.49})
    b = MockProvider(ProviderSpec(label="b", model="b", kind="mock"),
                     weights={"restart": 0.4, "reboot": 0.6})
    council = Council(JitRLKernel(store=Store(), seed=1), [a, b],
                      trigger=TriggerPolicy(margin_threshold=0.12))
    outcome = council.decide("s", ["restart", "reboot"])
    assert outcome.used
    # primary choose once (trigger); critique may still call text(). primary
    # is not asked to choose again.
    assert sum(1 for c in a.calls if "candidates" in c) == 1
    assert sum(1 for c in b.calls if "candidates" in c) >= 1
