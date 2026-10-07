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
import time

from fastapi import APIRouter, Body, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .. import sysinfo
from ..envs.pty_env import PTYError, PTYRefused
from ..screen.actions import ScreenActionError
from pathlib import Path

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
        """Client -> server. Resize arrives as JSON, keystrokes as raw text."""
        while not stop.is_set():
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("text")) is not None:
                if data.startswith('{"type":"resize"'):
                    try:
                        import json

                        payload = json.loads(data)
                        session.resize(int(payload.get("rows", 24)), int(payload.get("cols", 80)))
                    except (ValueError, TypeError):
                        pass  # a malformed resize is not worth dropping the shell over
                else:
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

# --- screen ----------------------------------------------------------------


def _screen():
    s = STATE.get("screen")
    if s is None:
        raise HTTPException(503, "screen capture not initialised")
    return s


def _actions():
    a = STATE.get("screen_actions")
    if a is None:
        raise HTTPException(503, "screen actions not initialised")
    return a


@router.get("/api/screen/frame")
async def screen_frame(image: bool = True):
    """One frame. `image=false` is the cheap change-detector-only path."""
    capture, perceiver = _screen()
    frame = await asyncio.to_thread(capture.grab, image)
    return frame.as_dict()


@router.get("/api/screen/frame.jpg")
async def screen_jpeg():
    from fastapi.responses import FileResponse

    capture, _ = _screen()
    frame = await asyncio.to_thread(capture.grab, True)
    if not frame.jpeg_path or not Path(frame.jpeg_path).exists():
        raise HTTPException(503, "no image produced")
    return FileResponse(frame.jpeg_path, media_type="image/jpeg",
                        headers={"X-Frame-Sha": frame.sha[:16],
                                 "X-Frame-Changed": "1" if frame.changed else "0"})


@router.get("/api/screen/observe")
async def screen_observe(question: str = ""):
    capture, perceiver = _screen()
    frame = await asyncio.to_thread(capture.grab, True)
    obs = await asyncio.to_thread(
        perceiver.observe, frame,
        question=question or "Describe what is on this screen and what a user should do next.",
    )
    return obs.as_dict()


@router.get("/api/screen/windows")
async def screen_windows():
    return await asyncio.to_thread(_actions().window_list)


class ClickBody(BaseModel):
    x: int
    y: int
    button: int = 1
    times: int = 1


class TypeBody(BaseModel):
    text: str
    delay_ms: int = 12


class KeyBody(BaseModel):
    key: str


@router.post("/api/screen/click")
async def screen_click(body: ClickBody):
    try:
        result = await asyncio.to_thread(_actions().click, body.x, body.y, body.button, body.times)
    except ScreenActionError as exc:
        raise HTTPException(400, str(exc))
    return result.as_dict()


@router.post("/api/screen/type")
async def screen_type(body: TypeBody):
    try:
        result = await asyncio.to_thread(_actions().type_text, body.text, body.delay_ms)
    except ScreenActionError as exc:
        raise HTTPException(400, str(exc))
    return result.as_dict()


@router.post("/api/screen/key")
async def screen_key(body: KeyBody):
    try:
        result = await asyncio.to_thread(_actions().key, body.key)
    except ScreenActionError as exc:
        raise HTTPException(400, str(exc))
    return result.as_dict()


@router.get("/api/screen/vision")
async def screen_vision():
    """Is a vision model actually attached? The agent should ask, not assume."""
    _, perceiver = _screen()
    return {"has_vision": perceiver.has_vision, "model": perceiver.model_name}


@router.get("/api/credit")
async def credit(limit: int = 50):
    """Per-memory credit, plus the totals.

    This is the observable that answers "is memory actually helping" with a
    number per memory rather than only an aggregate bench score.
    """
    store = STATE.get("store")
    if store is None:
        raise HTTPException(503, "store not initialised")
    return {"entries": await asyncio.to_thread(store.credit_report, limit),
            "totals": await asyncio.to_thread(store.credit_totals)}
