"""Tool bus: four sources behind one registry, plus the token-budget catalogue."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agentd.toolbus import (
    CallContext, MCPError, MCPServerStdio, RISK_DANGEROUS, RISK_READ, ToolBus, ToolResult,
    ToolSpec, load_plugin_dir, load_skill_dir, parse_frontmatter, register_builtins,
    register_mcp_server,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


@pytest.fixture()
def bus():
    b = ToolBus()
    register_builtins(b)
    return b


# -- registry semantics ---------------------------------------------------
def test_builtins_register_and_list(bus):
    names = bus.names()
    assert {"ssh.exec", "fs.read", "local.exec", "memory.recall"} <= set(names)
    assert "tools.expand" not in names            # hidden until expanded
    assert bus.by_source()["builtin"] == sorted(bus.by_source()["builtin"])


def test_names_without_namespace_are_rejected(bus):
    with pytest.raises(ValueError):
        bus.register(ToolSpec(name="oops", description="d", handler=lambda **k: ToolResult(ok=True)))
    with pytest.raises(ValueError):
        bus.register(ToolSpec(name="x.y", description="d"))


def test_unknown_tool_lists_what_is_available(bus):
    result = bus.call("nope.nope", {})
    assert not result.ok
    assert "available:" in result.error and "fs.read" in result.error


def test_argument_validation_reports_to_the_model_not_the_operator(bus):
    result = bus.call("fs.ls", {"path": 12})
    assert not result.ok and "must be string" in result.error
    result = bus.call("fs.ls", {"nope": 1})
    assert not result.ok and "unknown argument" in result.error
    assert bus.call("local.exec", {}).ok is False and "missing required" in bus.call("local.exec", {}).error


def test_disabled_tool_is_hidden_and_refused(bus):
    bus.disable("fs.ls")
    assert "fs.ls" not in bus.names()
    assert "disabled" in bus.call("fs.ls", {}).error
    bus.enable("fs.ls")
    assert bus.call("fs.ls", {"path": "."}).ok


def test_dangerous_tool_needs_approval(bus):
    ctx = CallContext(approved=False)
    result = bus.call("local.exec", {"command": "rm -rf /tmp/definitely-not-here"}, ctx)
    assert not result.ok and "safety gate" in result.error
    allowed = bus.call("local.exec", {"command": "rm -rf /tmp/definitely-not-here", "approved": True}, ctx)
    assert allowed.ok or allowed.data.get("rc") in (0, 1)


def test_readonly_builtin_runs(bus):
    result = bus.call("local.exec", {"command": "echo hello"}, CallContext())
    assert result.ok and "hello" in result.output
    assert bus.call("fs.ls", {"path": str(Path(__file__).parent)}, CallContext()).ok


# -- catalogue budget (token thrift) --------------------------------------
def test_catalog_respects_the_token_budget(bus):
    full = bus.catalog(token_budget=100_000)
    assert full.count("input_schema") == len(bus.names())
    assert len(full) // 4 <= 100_000

    tiny = bus.catalog(token_budget=40)
    assert len(tiny) // 4 <= 40 or "name" in tiny
    assert "input_schema" not in tiny

    mid = bus.catalog(token_budget=600)
    assert "input_schema" not in mid          # brief mode drops schemas
    assert mid.count("ssh.exec") >= 1


def test_expand_returns_the_full_schema(bus):
    info = bus.expand("ssh.exec")
    assert info["risk"] and "host" in info["parameters"]["properties"]
    assert "error" in bus.expand("nope.nope")


# -- plugins --------------------------------------------------------------
def test_plugin_dir_loads_good_and_isolates_bad(tmp_path, bus):
    (tmp_path / "good.py").write_text(
        "from agentd.toolbus.spec import ToolSpec, ToolResult\n"
        "def _ping(host: str, ctx=None):\n"
        "    return ToolResult(ok=True, output=f'pong {host}')\n"
        "TOOLS = [ToolSpec(name='ping', description='ping a thing',\n"
        "                   parameters={'type':'object','properties':{'host':{'type':'string'}},\n"
        "                               'required':['host']}, handler=_ping)]\n",
        encoding="utf-8",
    )
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom on import')\n", encoding="utf-8")
    (tmp_path / "_ignored.py").write_text("x = 1\n", encoding="utf-8")

    report = load_plugin_dir(bus, tmp_path)
    assert any("plugin.good.ping" in str(v) for v in report["loaded"])
    assert "broken.py" in report["errors"] and "boom on import" in report["errors"]["broken.py"]
    assert "_ignored.py" not in json.dumps(report["loaded"])

    result = bus.call("plugin.good.ping", {"host": "app-01"}, CallContext())
    assert result.ok and result.output == "pong app-01"


def test_plugin_missing_directory_is_skipped_not_fatal(tmp_path, bus):
    report = load_plugin_dir(bus, tmp_path / "nope")
    assert report["loaded"] == [] and report["errors"] == {} and report["skipped"]


# -- skills ---------------------------------------------------------------
def test_frontmatter_parser():
    meta, body = parse_frontmatter(
        "---\nname: nginx-logs\ndescription: Read nginx errors\nargs: host, lines\n"
        "run: ssh {host} tail -n {lines}\n---\n\nStep 1: look for upstream timeouts.\n"
    )
    assert meta["name"] == "nginx-logs"
    assert meta["args"] == ["host", "lines"]
    assert body.startswith("Step 1")


def test_skill_dir_becomes_tools(tmp_path, bus):
    skill = tmp_path / "explain-bus"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: explain-bus\ndescription: Explain the local machine\n"
        "when: the user asks what this box is\n---\n\nReport uname then uptime.\n",
        encoding="utf-8",
    )
    report = load_skill_dir(bus, tmp_path)
    assert "skill.explain-bus" in report["loaded"]
    result = bus.call("skill.explain-bus", {}, CallContext())
    assert result.ok and "Report uname then uptime" in result.output


def test_skill_with_run_command_passes_gate(tmp_path, bus):
    skill = tmp_path / "hi"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: hi\ndescription: say hi\nrun: echo hello-from-skill\n---\n\nbody\n",
        encoding="utf-8",
    )
    load_skill_dir(bus, tmp_path)
    assert "hello-from-skill" in bus.call("skill.hi", {}, CallContext()).output

    danger = tmp_path / "wipe"
    danger.mkdir()
    (danger / "SKILL.md").write_text(
        "---\nname: wipe\ndescription: destructive on purpose\nrun: rm -rf /tmp/agentd-skill-test\n---\n\nbody\n",
        encoding="utf-8",
    )
    load_skill_dir(bus, tmp_path)
    result = bus.call("skill.wipe", {}, CallContext())
    assert not result.ok and "safety gate" in result.error


# -- MCP ------------------------------------------------------------------
def test_mcp_handshake_discovery_and_call(bus):
    server = MCPServerStdio("fixture", sys.executable, [str(FIXTURE)])
    report = register_mcp_server(bus, server)
    try:
        assert report["error"] is None, report
        assert "instructions" in report and "Fixture server" in report["instructions"]
        assert "mcp.fixture.read_status" in report["tools"]
        assert "mcp.fixture.drop_table" in report["tools"]

        result = bus.call("mcp.fixture.read_status", {"service": "nginx"}, CallContext())
        assert result.ok and "nginx: active" in result.output

        risky = bus.get("mcp.fixture.drop_table")
        assert risky.risk == RISK_DANGEROUS
        blocked = bus.call("mcp.fixture.drop_table", {"name": "users"}, CallContext())
        assert not blocked.ok

        failing = bus.call("mcp.fixture.explode", {}, CallContext())
        assert not failing.ok and "boom" in failing.output
    finally:
        server.stop()


def test_mcp_missing_command_reports_instead_of_crashing(bus):
    before = bus.names(include_hidden=True)
    server = MCPServerStdio("ghost", str(FIXTURE) + ".does.not.exist")
    report = register_mcp_server(bus, server)
    assert report["error"] and "not found" in report["error"]
    assert bus.names(include_hidden=True) == before
    assert not [n for n in bus.names() if n.startswith("mcp.ghost")]


def test_mcp_error_on_unknown_method():
    server = MCPServerStdio("fixture", sys.executable, [str(FIXTURE)])
    server.start()
    try:
        with pytest.raises(MCPError):
            server.request("tools/nope", {})
    finally:
        server.stop()
