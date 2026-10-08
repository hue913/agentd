"""MCP client: any Model Context Protocol server becomes agentd tools.

Newline-delimited JSON-RPC 2.0 over stdio, per the 2025-06-18 specification:
initialize -> notifications/initialized -> tools/list -> tools/call. Tool risk
comes from the MCP annotations (readOnlyHint / destructiveHint) so a server that
declares a destructive tool still passes the approval path.

Transport notes (each one earned by a real failure mode):
* reads go through select() against a deadline — the old BufferedReader.readline()
  had no timeout, so a hung MCP server blocked the calling thread forever;
* stderr is drained by a daemon thread into a 2 KB ring — stderr=PIPE with no
  reader deadlocks the child once the OS pipe buffer fills;
* errors that are swallowed for resilience are recorded on the instance
  (last_error), never passed silently.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path

from ..log import get_logger
from .spec import RISK_DANGEROUS, RISK_READ, RISK_WRITE, ToolResult, ToolSpec

PROTOCOL_VERSION = "2025-06-18"
# Override with AGENTD_MCP_TIMEOUT (seconds) without touching code.
DEFAULT_TIMEOUT = int(os.environ.get("AGENTD_MCP_TIMEOUT", "30") or "30")
STDERR_TAIL_BYTES = 2048
READ_CHUNK = 65536

log = get_logger("agentd.mcp")


class MCPError(RuntimeError):
    pass


class MCPServerStdio:
    def __init__(self, name: str, command: str | list[str], args: list[str] | None = None,
                 env: dict | None = None, cwd: str | None = None,
                 timeout: int | None = None):
        self.name = _safe(name)
        self.argv = [command] + list(args or []) if isinstance(command, str) else list(command)
        self.env = {**os.environ, **(env or {})}
        self.cwd = str(Path(cwd).expanduser()) if cwd else None
        self.timeout = DEFAULT_TIMEOUT if timeout is None else timeout
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        self.server_info: dict = {}
        # Diagnostics for suppressed errors — see stop() and _readline_until().
        self.last_error: str = ""
        self._rbuf = bytearray()
        self._stderr_buf = bytearray()
        self._stderr_lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------
    def start(self) -> dict:
        if self._proc and self._proc.poll() is None:
            return self.server_info
        try:
            # Binary, unbuffered pipes: _readline_until() selects on the raw
            # fd, and a text-mode reader would prefetch bytes select cannot see.
            self._proc = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, env=self.env, cwd=self.cwd,
            )
        except FileNotFoundError:
            raise MCPError(f"MCP server '{self.name}': command not found: {self.argv[0]}") from None

        threading.Thread(target=self._drain_stderr, daemon=True,
                         name=f"agentd-mcp-{self.name}-stderr").start()

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
            except Exception as exc:
                # Closing stdin of a dying server can fail legitimately; the
                # terminate/kill below still reaps it. Recorded, not swallowed.
                self.last_error = f"stdin close failed: {exc}"
                log.debug("mcp '%s': %s", self.name, self.last_error)
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

    # -- stderr -----------------------------------------------------------
    def _drain_stderr(self) -> None:
        """Tail the server's stderr into a small ring buffer.

        Without a reader, a chatty server fills the pipe, blocks on write and
        freezes — usually taking the tool call with it. The tail (not the
        whole log) is kept because its only consumer is an error message.
        """
        stream = self._proc.stderr if self._proc else None
        if stream is None:
            return
        while True:
            try:
                chunk = stream.read(4096)
            except (OSError, ValueError):
                return  # pipe closed during shutdown; nothing left to drain
            if not chunk:
                return
            with self._stderr_lock:
                self._stderr_buf.extend(chunk)
                excess = len(self._stderr_buf) - STDERR_TAIL_BYTES
                if excess > 0:
                    del self._stderr_buf[:excess]

    def _stderr_tail(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr_buf).decode("utf-8", "replace")

    # -- json-rpc ---------------------------------------------------------
    def _send(self, payload: dict) -> None:
        if not self._proc or not self._proc.stdin:
            raise MCPError(f"MCP server '{self.name}' is not running")
        self._proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _readline_until(self, deadline: float | None) -> str:
        """One newline-terminated line, honoring the deadline.

        select() with the remaining time gives a real read timeout on the raw
        fd; a small buffer carries partial lines across polls.
        """
        if not self._proc or not self._proc.stdout:
            raise MCPError(f"MCP server '{self.name}' is not running")
        fd = self._proc.stdout.fileno()
        while b"\n" not in self._rbuf:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise MCPError(
                    f"MCP server '{self.name}' timed out after {self.timeout}s waiting for a "
                    f"response. stderr tail: {self._stderr_tail()}")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise MCPError(
                    f"MCP server '{self.name}' timed out after {self.timeout}s waiting for a "
                    f"response. stderr tail: {self._stderr_tail()}")
            try:
                chunk = os.read(fd, READ_CHUNK)
            except OSError as exc:
                self.last_error = f"stdout read failed: {exc}"
                raise MCPError(f"MCP server '{self.name}' pipe error: {exc}") from None
            if not chunk:
                raise MCPError(
                    f"MCP server '{self.name}' closed the pipe. "
                    f"stderr tail: {self._stderr_tail()}")
            self._rbuf.extend(chunk)
        line, _, rest = self._rbuf.partition(b"\n")
        self._rbuf = bytearray(rest)
        return line.decode("utf-8", "replace")

    def _read(self, deadline: float | None) -> dict:
        line = self._readline_until(deadline)
        try:
            return json.loads(line)
        except ValueError:
            raise MCPError(f"MCP server '{self.name}' sent non-JSON: {line[:200]!r}") from None

    def request(self, method: str, params: dict) -> dict:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            # The deadline finally does its job: every read below is bounded
            # by it, so one hung request cannot wedge a worker thread.
            deadline = time.monotonic() + self.timeout
            while True:
                message = self._read(deadline)
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
