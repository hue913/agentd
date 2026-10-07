"""MCP client: any Model Context Protocol server becomes agentd tools.

Newline-delimited JSON-RPC 2.0 over stdio, per the 2025-06-18 specification:
initialize -> notifications/initialized -> tools/list -> tools/call. Tool risk
comes from the MCP annotations (readOnlyHint / destructiveHint) so a server that
declares a destructive tool still passes the approval path.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

from .spec import RISK_DANGEROUS, RISK_READ, RISK_WRITE, ToolResult, ToolSpec

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TIMEOUT = 30


class MCPError(RuntimeError):
    pass


class MCPServerStdio:
    def __init__(self, name: str, command: str | list[str], args: list[str] | None = None,
                 env: dict | None = None, cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        self.name = _safe(name)
        self.argv = [command] + list(args or []) if isinstance(command, str) else list(command)
        self.env = {**os.environ, **(env or {})}
        self.cwd = str(Path(cwd).expanduser()) if cwd else None
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        self.server_info: dict = {}

    # -- lifecycle --------------------------------------------------------
    def start(self) -> dict:
        if self._proc and self._proc.poll() is None:
            return self.server_info
        try:
            self._proc = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=self.env, cwd=self.cwd,
            )
        except FileNotFoundError:
            raise MCPError(f"MCP server '{self.name}': command not found: {self.argv[0]}") from None

        info = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "clientInfo": {"name": "agentd", "version": "0.1.0"},
        })
        self.notify("notifications/initialized", {})
        self.server_info = info
        return info

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # -- json-rpc ---------------------------------------------------------
    def _send(self, payload: dict) -> None:
        if not self._proc or not self._proc.stdin:
            raise MCPError(f"MCP server '{self.name}' is not running")
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _read(self) -> dict:
        assert self._proc and self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            err = (self._proc.stderr.read() if self._proc.stderr else "") or ""
            raise MCPError(f"MCP server '{self.name}' closed the pipe. stderr: {err[-400:]}")
        try:
            return json.loads(line)
        except ValueError:
            raise MCPError(f"MCP server '{self.name}' sent non-JSON: {line[:200]!r}") from None

    def request(self, method: str, params: dict) -> dict:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            deadline = self.timeout
            while True:
                message = self._read()
                if message.get("id") != request_id:
                    if "method" in message:      # a server-initiated request we do not use
                        continue
                    continue
                if "error" in message:
                    error = message["error"]
                    raise MCPError(f"{method} failed: {error.get('code')} {error.get('message')}")
                return message.get("result", {})

    def notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    # -- tools ------------------------------------------------------------
    def list_tools(self) -> list[dict]:
        self.start()
        return self.request("tools/list", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        try:
            result = self.request("tools/call", {"name": tool_name, "arguments": arguments})
        except MCPError as exc:
            return ToolResult(ok=False, error=str(exc))
        content = result.get("content") or []
        text = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        if not text:
            structured = result.get("structuredContent")
            text = json.dumps(structured, ensure_ascii=False) if structured else "(no text content)"
        return ToolResult(ok=not result.get("isError", False), output=text[:20_000],
                          data={"server": self.name, "tool": tool_name, "raw_keys": sorted(result)})


def _risk_for(tool_def: dict) -> str:
    annotations = tool_def.get("annotations") or {}
    if annotations.get("destructiveHint"):
        return RISK_DANGEROUS
    if annotations.get("readOnlyHint"):
        return RISK_READ
    if annotations.get("openWorldHint") or annotations.get("idempotentHint") is False:
        return RISK_WRITE
    return RISK_WRITE


def register_mcp_server(bus, server: MCPServerStdio) -> dict:
    """Discover a server's tools and attach them as `mcp.<server>.<tool>`."""
    report = {"server": server.name, "tools": [], "error": None, "instructions": ""}
    try:
        info = server.start()
        report["instructions"] = (info.get("instructions") or "")[:1000]
        defs = server.list_tools()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    for definition in defs:
        tool_name = definition.get("name")
        if not tool_name:
            continue
        schema = definition.get("inputSchema") or {"type": "object", "properties": {}}
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": schema.get("properties", {})}

        def handler(server=server, tool_name=tool_name, ctx=None, **kwargs) -> ToolResult:
            result = server.call_tool(tool_name, kwargs)
            return result

        bus.register(
            ToolSpec(
                name=f"mcp.{server.name}.{_safe(tool_name)}",
                description=(definition.get("description") or f"MCP tool {tool_name}")[:300],
                parameters=schema,
                source="mcp",
                risk=_risk_for(definition),
                handler=handler,
            ),
            replace=True,
        )
        report["tools"].append(f"mcp.{server.name}.{_safe(tool_name)}")
    return report


def _safe(value: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(value).strip())
    return out or "unnamed"
