#!/usr/bin/env python3
"""A minimal MCP server speaking newline-delimited JSON-RPC over stdio.

Used only by the test suite: it proves agentd's client does the handshake,
discovers tools, honours risk annotations and survives a tool that errors.
"""

import json
import sys

TOOLS = [
    {
        "name": "read_status",
        "description": "Return the status of a named service",
        "inputSchema": {"type": "object", "properties": {"service": {"type": "string"}},
                        "required": ["service"]},
        "annotations": {"readOnlyHint": True, "destructiveHint": False},
    },
    {
        "name": "drop table",
        "description": "Destructive example used to assert risk mapping",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        "annotations": {"readOnlyHint": False, "destructiveHint": True},
    },
    {
        "name": "explode",
        "description": "Always returns isError",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            continue

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "fixture-mcp", "version": "0.0.1"},
                "instructions": "Fixture server for agentd tests."}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "read_status":
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "content": [{"type": "text", "text": f"{arguments.get('service')}: active"}]}})
            elif name == "drop table":
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "content": [{"type": "text", "text": f"dropped {arguments.get('name')}"}]}})
            else:
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "content": [{"type": "text", "text": "boom"}], "isError": True}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32601, "message": f"method not found: {method}"}})


if __name__ == "__main__":
    main()
