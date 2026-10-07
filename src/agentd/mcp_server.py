"""agentd as an MCP *server*: other clients (Claude Desktop, Qoder, any MCP host) can
call the same gated tools, read the same memory, and use the same learning kernel.

Newline-delimited JSON-RPC 2.0 over stdio, per MCP 2025-06-18. Nothing here is a
second implementation of tool execution: every call goes through the same ToolBus,
so the safety gate, the audit log and the approval path stay in force.
"""

from __future__ import annotations

import json
import sys
from typing import TextIO

from .toolbus.spec import RISK_DANGEROUS

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "agentd", "version": "0.1.0"}

RESOURCES = {
    "agentd://memory": "The full experience pack as JSON (state, action, return).",
    "agentd://memory.md": "The same pack rendered as human-readable Markdown.",
    "agentd://risks": "Unresolved disagreements raised by the council.",
    "agentd://tools": "Registered tools grouped by source.",
}


def _tool_definitions(bus) -> list[dict]:
    out = []
    for name in bus.names(include_hidden=True):
        spec = bus.get(name)
        annotations = {
            "readOnlyHint": spec.risk == "read",
            "destructiveHint": spec.risk == RISK_DANGEROUS,
            "idempotentHint": False,
            "openWorldHint": spec.namespace in ("ssh", "mcp", "local"),
        }
        out.append({"name": name, "description": spec.description,
                    "inputSchema": spec.parameters, "annotations": annotations})
    return out


def _read_resource(runtime, uri: str) -> str:
    from .kernel.pack import export_pack, to_markdown

    if uri in ("agentd://memory", "agentd://memory.md"):
        pack = export_pack(runtime.store, source={"providers": list(runtime.providers)})
        return to_markdown(pack) if uri.endswith(".md") else json.dumps(pack, ensure_ascii=False)
    if uri == "agentd://risks":
        return json.dumps(runtime.store.risks_for(limit=100), ensure_ascii=False)
    if uri == "agentd://tools":
        return json.dumps({"names": runtime.bus.names(include_hidden=True),
                           "by_source": runtime.bus.by_source()}, ensure_ascii=False)
    raise KeyError(f"unknown resource {uri}")


def handle(message: dict, runtime, bus) -> dict | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    def result(payload):
        return {"jsonrpc": "2.0", "id": request_id, "result": payload}

    def error(code, text):
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": text}}

    if request_id is None:                       # notification
        return None
    if method == "initialize":
        return result({"protocolVersion": PROTOCOL_VERSION,
                       "capabilities": {"tools": {"listChanged": False}, "resources": {}},
                       "serverInfo": SERVER_INFO,
                       "instructions": "agentd: memory-backed agent tools. Destructive actions "
                                       "require approval=true and will be refused otherwise."})
    if method == "ping":
        return result({})
    if method == "tools/list":
        return result({"tools": _tool_definitions(bus)})
    if method == "resources/list":
        return result({"resources": [{"uri": uri, "name": uri.rsplit("/", 1)[-1], "description": desc}
                                     for uri, desc in RESOURCES.items()]})
    if method == "resources/read":
        try:
            text = _read_resource(runtime, params.get("uri", ""))
        except Exception as exc:
            return error(-32602, str(exc))
        return result({"contents": [{"uri": params.get("uri"), "mimeType": "text/plain", "text": text}]})
    if method == "tools/call":
        name = params.get("name", "")
        arguments = dict(params.get("arguments") or {})
        approved = bool(arguments.pop("approved", False))
        from .toolbus import CallContext

        ctx = CallContext(session="mcp", approved=approved, approver=None,
                          ssh=runtime.ssh, kernel=runtime.kernel, audit=runtime.audit,
                          extra={"bus": bus})
        outcome = bus.call(name, arguments, ctx)
        payload = {"content": [{"type": "text", "text": outcome.output or (outcome.error or "")}],
                   "isError": not outcome.ok}
        if outcome.error:
            payload["content"].append({"type": "text", "text": f"error: {outcome.error}"})
        return result(payload)
    if method in ("prompts/list",):
        return result({"prompts": []})
    return error(-32601, f"method not found: {method}")


def serve(runtime=None, bus=None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    if runtime is None or bus is None:
        from .runtime import Runtime

        runtime = runtime or Runtime()
        bus = bus or runtime.bus
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        try:
            reply = handle(message, runtime, bus)
        except Exception as exc:
            reply = {"jsonrpc": "2.0", "id": message.get("id"),
                     "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False, default=str) + "\n")
            stdout.flush()
    return 0
