"""Ops surface: interactive PTY over WebSocket, and host metrics.

Routes:
    POST   /api/pty/open          create a session (admission-checked)
    GET    /api/pty               list live sessions
    POST   /api/pty/{id}/close    close one
    WS     /api/pty/{id}/stream   bidirectional shell
    GET    /api/sys/overview      uptime / load / memory / disk
    GET    /api/sys/services      systemd unit states
    GET    /api/sys/procs         top processes
    GET    /api/sys/logs/{unit}   journal tail
    GET    /api/sys/snapshot      all three at once (15s SSE tick uses this)

WebSocket auth note: a browser WebSocket handshake cannot carry an
Authorization header, so the PTY stream authenticates by the same
`?token=` query parameter the SSE stream already uses. The value is compared
with the same constant-time helper, so this is a transport quirk and not a
weaker check.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time

from fastapi import APIRouter, Body, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .. import sysinfo
from ..envs.pty_env import PTYError, PTYRefused

router = APIRouter()

# Populated by api/server.py so the router does not have to reach into Runtime.
STATE: dict = {}


def _hub():
    hub = STATE.get("pty")
    if hub is None:
        raise HTTPException(503, "pty hub not initialised")
    return hub


def _token_ok(presented: str) -> bool:
    import hmac

    expected = STATE.get("token", "")
    return bool(expected) and hmac.compare_digest(presented, expected)


def _control_frame(text: str) -> dict | None:
    """Return the frame if `text` is a control JSON object, else None.

    Only small objects whose first key is `type` qualify, so ordinary shell
    input that happens to contain a brace is never swallowed.
    """
    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        payload = json.loads(stripped)
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("type"), str):
        return payload
    return None


# --- pty --------------------------------------------------------------------

class OpenBody(BaseModel):
    host: str
    purpose: str = ""
    rows: int = 24
    cols: int = 80
    approved: bool = False


@router.post("/api/pty/open")
async def pty_open(body: OpenBody):
    try:
        session = _hub().open(body.host, purpose=body.purpose, rows=body.rows,
                              cols=body.cols, approved=body.approved)
    except PTYRefused as exc:
        raise HTTPException(403, str(exc))
    except PTYError as exc:
        raise HTTPException(400, str(exc))
    return {**session.digest(), "alive": session.alive}


@router.get("/api/pty")
async def pty_list():
    return {"sessions": _hub().listing(), "reaped": _hub().reap()}


@router.post("/api/pty/{session_id}/close")
async def pty_close(session_id: str):
    try:
        return _hub().close(session_id)
    except PTYError as exc:
        raise HTTPException(404, str(exc))


@router.websocket("/api/pty/{session_id}/stream")
async def pty_stream(ws: WebSocket, session_id: str, token: str = Query(default="")):
    if not _token_ok(token):
        await ws.close(code=4401, reason="unauthorized")
        return
    try:
        hub = _hub()
        session = hub.get(session_id)
    except PTYError:
        await ws.close(code=4404, reason="no such session")
        return

    await ws.accept()
    stop = asyncio.Event()

    async def pump() -> None:
        """Server -> client. Polls the PTY ring; the read thread owns the fd."""
        last_ping = time.time()
        while not stop.is_set():
            data = await asyncio.to_thread(session.read)
            if data:
                await ws.send_bytes(data)
                last_ping = time.time()
            elif not session.alive:
                with contextlib.suppress(Exception):
                    await ws.send_json({"type": "exit", "reason": session.exit_reason})
                break
            elif time.time() - last_ping > 20:
                with contextlib.suppress(Exception):
                    await ws.send_json({"type": "ping"})
                last_ping = time.time()
            await asyncio.sleep(0.05)

    async def sink() -> None:
        """Client -> server. Control frames are JSON; everything else is keys.

        Control frames are detected by parsing, not by prefix matching. A
        prefix check silently misroutes a resize into the shell as literal
        text the moment a client emits `{"type": "resize", ...}` with a space
        after the colon -- which is exactly what json.dumps does.
        """
        while not stop.is_set():
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("text")
            if data is not None:
                control = _control_frame(data)
                if control and control.get("type") == "resize":
                    try:
                        session.resize(int(control.get("rows", 24)), int(control.get("cols", 80)))
                    except (TypeError, ValueError):
                        pass  # a malformed resize is not worth dropping the shell over
                    continue
                await asyncio.to_thread(session.write, data.encode())
            elif (raw := msg.get("bytes")) is not None:
                await asyncio.to_thread(session.write, raw)

    pump_task = asyncio.create_task(pump())
    sink_task = asyncio.create_task(sink())
    try:
        await asyncio.wait({pump_task, sink_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.set()
        for task in (pump_task, sink_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # The client going away must not leave a root shell running.
        with contextlib.suppress(Exception):
            hub.close(session_id, reason="websocket closed")


# --- sys --------------------------------------------------------------------

@router.get("/api/sys/overview")
async def sys_overview():
    return await asyncio.to_thread(sysinfo.overview)


@router.get("/api/sys/services")
async def sys_services(units: str = ""):
    wanted = [u for u in units.split(",") if u] or None
    return await asyncio.to_thread(sysinfo.services, wanted)


@router.get("/api/sys/procs")
async def sys_procs(limit: int = 15, sort: str = "rss"):
    return await asyncio.to_thread(sysinfo.processes, limit, sort)


@router.get("/api/sys/logs/{unit}")
async def sys_logs(unit: str, lines: int = 200):
    result = await asyncio.to_thread(sysinfo.logs, unit, lines)
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@router.get("/api/sys/snapshot")
async def sys_snapshot(sort: str = "rss"):
    return await asyncio.to_thread(sysinfo.snapshot, sort)
