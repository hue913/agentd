"""The tool task: an open goal whose actions are real bus calls.

What must hold:
1. The action grammar parses the shapes a model actually emits (JSON args,
   bare-command shorthand, finish), and a malformed option is a recoverable
   observation rather than a crash.
2. `apply` dispatches through the bus -- approval gating and safety
   classification included -- instead of inventing its own execution path.
3. An ssh.* call without an explicit host falls back to the default host.
4. The loop drives the task to `finish` end to end and the answer lands in
   the report.
"""

from __future__ import annotations

import pytest

from agentd.context import ContextBuilder, Ledger
from agentd.envs.tool_task import ActionFormatError, ToolTask, parse_action
from agentd.kernel import JitRLKernel, Store
from agentd.loop import AgentLoop, LoopConfig
from agentd.providers import MockProvider, ProviderSpec
from agentd.toolbus import CallContext, ToolBus, register_builtins
from agentd.toolbus.spec import RISK_READ, ToolResult, ToolSpec


# --- grammar ---------------------------------------------------------------

def test_json_arguments_roundtrip():
    kind, name, args = parse_action('ssh.exec {"host": "self", "command": "df -h"}')
    assert (kind, name) == ("call", "ssh.exec")
    assert args == {"host": "self", "command": "df -h"}


def test_exec_shorthand_becomes_command():
    kind, name, args = parse_action("local.exec echo hello world")
    assert (kind, name, args) == ("call", "local.exec", {"command": "echo hello world"})


def test_zero_argument_tool():
    assert parse_action("memory.stats") == ("call", "memory.stats", {})


def test_finish_with_json_and_bare_text():
    assert parse_action('finish {"answer": "nginx is up"}')[2]["answer"] == "nginx is up"
    assert parse_action("finish all good")[2]["answer"] == "all good"


def test_non_exec_tool_without_json_is_refused():
    with pytest.raises(ActionFormatError):
        parse_action("fs.read /etc/hosts")


def test_garbage_is_refused():
    with pytest.raises(ActionFormatError):
        parse_action("   ")
    with pytest.raises(ActionFormatError):
        parse_action('ssh.exec {"broken": ')


# --- execution through the bus --------------------------------------------

def _bus_with_echo(calls: list) -> ToolBus:
    bus = ToolBus()

    def echo(text: str = "") -> ToolResult:
        calls.append({"text": text})
        return ToolResult(ok=True, output=f"echo: {text}")

    def ssh_echo(command: str = "", host: str = "") -> ToolResult:
        calls.append({"command": command, "host": host})
        return ToolResult(ok=True, output=f"{host}$ {command}")

    bus.register(ToolSpec(name="demo.echo", description="echo", parameters={
        "type": "object", "properties": {"text": {"type": "string"}}, "required": []},
        source="test", risk=RISK_READ, handler=echo))
    bus.register(ToolSpec(name="ssh.exec", description="ssh echo", parameters={
        "type": "object", "properties": {"command": {"type": "string"}, "host": {"type": "string"}},
        "required": ["command"]}, source="test", risk=RISK_READ, handler=ssh_echo))
    return bus


def test_apply_dispatches_to_the_bus():
    calls: list = []
    task = ToolTask(goal_text="say hi", hosts=["self"])
    ctx = CallContext(session="t", extra={"bus": _bus_with_echo(calls)})
    state, done, reward = task.apply('demo.echo {"text": "hi"}', ctx)
    assert not done and reward == 0.0
    assert "demo.echo -> ok" in state and "echo: hi" in state
    assert calls == [{"text": "hi"}]


def test_ssh_call_defaults_to_the_task_host():
    calls: list = []
    task = ToolTask(goal_text="check disk", hosts=["prod-1", "prod-2"])
    ctx = CallContext(session="t", extra={"bus": _bus_with_echo(calls)})
    task.apply("ssh.exec df -h", ctx)
    assert calls == [{"command": "df -h", "host": "prod-1"}]


def test_malformed_action_is_an_observation_not_an_exception():
    task = ToolTask(goal_text="x", hosts=[])
    state, done, _ = task.apply("not a valid call !!!", CallContext())
    assert not done and "unusable action" in state


def test_finish_ends_the_episode_with_the_answer():
    task = ToolTask(goal_text="x", hosts=[])
    state, done, reward = task.apply('finish {"answer": "all clear"}', CallContext())
    assert done and reward == 1.0
    assert "all clear" in state


def test_dangerous_command_is_blocked_before_it_runs():
    """The gate decides, not the task: rm -rf must come back blocked."""
    bus = ToolBus()
    register_builtins(bus)
    task = ToolTask(goal_text="pretend to clean", hosts=[])
    ctx = CallContext(session="t", extra={"bus": bus})  # no approver attached
    state, done, _ = task.apply("local.exec rm -rf / --no-preserve-root", ctx)
    assert not done
    assert "FAILED" in state
    assert "blocked" in state.lower() or "approval" in state.lower()


# --- the loop drives it to the end ----------------------------------------

def test_loop_runs_a_tool_goal_to_finish(tmp_path):
    task = ToolTask(goal_text="echo hello and report", hosts=["self"])
    store = Store(str(tmp_path / "m.db"))
    kernel = JitRLKernel(store=store, enabled=False, seed=5)
    bus = _bus_with_echo([])
    provider = MockProvider(
        ProviderSpec(label="m", model="m", kind="mock"),
        weights={"demo.echo": 0.9, "finish": 0.3},
        critiques=[
            '{"options": ["demo.echo {\\"text\\": \\"hello\\"}"]}',
            '{"options": ["finish {\\"answer\\": \\"hello was echoed\\"}"]}',
        ],
    )
    loop = AgentLoop(kernel, provider, bus, context=ContextBuilder(), ledger=Ledger(),
                     config=LoopConfig(max_steps=4, reflect=False))
    report = loop.run(task)
    assert report.success is True
    assert "hello was echoed" in report.final
    assert len(report.steps) == 2
    assert "demo.echo" in report.steps[0].chosen_action
    assert report.steps[0].ok is True
