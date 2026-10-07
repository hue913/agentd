"""HTTP API: JSON + SSE. Binds to loopback by design; remote reach is via SSH tunnel."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time

from ..context import compact_observation
from ..kernel.pack import diff_packs, export_pack, import_pack, load_pack, to_markdown
from ..runtime import Runtime

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install the api extra: pip install 'agentd[api]'") from exc


# --- auth -------------------------------------------------------------------
# The service binds loopback, so loopback alone is NOT a security boundary: any
# `ssh -L 8765:127.0.0.1:8765` turns every route into a remotely reachable one,
# including /api/ssh/exec. A bearer token closes that hole.
#
# Fail-closed: if the server has no token configured, protected routes return
# 503 instead of serving unauthenticated. Silently running open was the previous
# behaviour and it is the bug this block exists to fix.
#
# /healthz is deliberately left open: it is a liveness probe, exposes no
# credentials or commands, and keeping it open means a misconfigured token
# cannot wedge systemd's supervision.
API_TOKEN_ENV = "AGENTD_API_TOKEN"
PUBLIC_PATHS = {"/healthz"}


class AuthNotConfigured(RuntimeError):
    pass


def server_token() -> str:
    return os.environ.get(API_TOKEN_ENV, "").strip()


def _presented(request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    alt = request.headers.get("x-agentd-token", "")
    if alt.strip():
        return alt.strip()
    # EventSource cannot set headers, so the SSE stream has to carry the token
    # in the query string. Only honoured for that reason; browsers cannot attach
    # headers to EventSource at all.
    return request.query_params.get("token", "").strip()


def require_auth(request: Request) -> None:
    path = request.url.path
    if path in PUBLIC_PATHS:
        return
    expected = server_token()
    if not expected:
        raise HTTPException(
            503,
            f"{API_TOKEN_ENV} is not set on the server; refusing to serve protected "
            "routes unauthenticated. Set it in the systemd unit / EnvironmentFile.",
        )
    got = _presented(request)
    if not got or not hmac.compare_digest(got, expected):
        raise HTTPException(401, "missing or invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


def create_app(runtime: Runtime | None = None) -> "FastAPI":
    rt = runtime or Runtime()
    app = FastAPI(title="agentd", version="0.1.0",
                  description="Test-time-RL agent kernel with pluggable models, SSH tools, "
                              "plugins/skills/MCP and a live server view.",
                  dependencies=[Depends(require_auth)])

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "ts": time.time(), "kernel": rt.kernel.stats()}

    @app.get("/api/state")
    def state():
        return rt.state()

    @app.get("/api/tools")
    def tools(expand: str = ""):
        if expand:
            return rt.bus.expand(expand)
        return {"names": rt.bus.names(), "by_source": rt.bus.by_source(),
                "catalog_tokens": rt.bus.signature_tokens()}

    @app.post("/api/session")
    async def start_session(body: dict):
        task = body.get("task")
        if not task:
            raise HTTPException(400, "body needs 'task'")
        try:
            session = rt.create_session(task, provider=body.get("provider", ""),
                                        learning=bool(body.get("learning", True)),
                                        max_steps=int(body.get("max_steps", 12)),
                                        council=bool(body.get("council", False)))
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        asyncio.create_task(asyncio.to_thread(rt.run_session, session))
        return {"session": session.id, "task": session.task}

    @app.get("/api/session/{session_id}")
    def get_session(session_id: str):
        session = rt.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "no such session")
        return {"id": session.id, "task": session.task, "status": session.status,
                "report": session.report,
                "pending_approvals": list(session.pending_approvals)}

    @app.post("/api/approve")
    def approve(body: dict):
        token, approved = body.get("token"), bool(body.get("approved"))
        if not token:
            raise HTTPException(400, "body needs 'token'")
        if not rt.resolve_approval(token, approved):
            raise HTTPException(404, "that approval request is no longer pending")
        return {"token": token, "approved": approved}

    @app.get("/api/memory")
    def memory(fmt: str = "json"):
        pack = export_pack(rt.store, source={"providers": list(rt.providers)})
        if fmt == "md":
            return HTMLResponse(to_markdown(pack))
        return pack

    @app.post("/api/memory/import")
    async def memory_import(body: dict):
        raw = body.get("pack")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raise HTTPException(400, "pack must be a JSON object or JSON string")
        report = import_pack(rt.store, raw, skip_tasks=body.get("skip_tasks", []),
                             current_source={"providers": list(rt.providers)})
        return report

    @app.get("/api/memory/diff")
    def memory_diff(other_path: str):
        try:
            other = load_pack(other_path)
        except FileNotFoundError:
            raise HTTPException(404, f"no such pack: {other_path}")
        return diff_packs(export_pack(rt.store), other)

    @app.get("/api/risks")
    def risks(limit: int = 50):
        return {"risks": rt.store.risks_for(limit=limit)}

    @app.get("/api/audit")
    def audit(limit: int = 50, host: str = ""):
        return {"entries": rt.audit.tail(limit=limit, host=host or None)}

    @app.post("/api/ssh/exec")
    async def ssh_exec(body: dict):
        host, command = body.get("host"), body.get("command")
        if not host or not command:
            raise HTTPException(400, "body needs 'host' and 'command'")
        from ..envs.ssh_env import ApprovalRequired

        try:
            result = rt.ssh.exec(host, command, approved=bool(body.get("approved")),
                                 session=body.get("session"))
        except ApprovalRequired as exc:
            return JSONResponse({"blocked": True, "reason": str(exc)}, status_code=202)
        except Exception as exc:
            raise HTTPException(502, f"{type(exc).__name__}: {exc}")
        return result.as_dict()

    @app.post("/api/ssh/probe")
    async def ssh_probe(body: dict):
        host = body.get("host")
        if not host:
            raise HTTPException(400, "body needs 'host'")
        try:
            return rt.ssh.probe(host)
        except Exception as exc:
            raise HTTPException(502, str(exc))

    @app.post("/api/util/compact")
    async def compact(body: dict):
        text = body.get("text", "")
        budget = int(body.get("budget_chars", 4000))
        compacted, stats = compact_observation(text, budget)
        return {"text": compacted, **stats}

    @app.get("/api/viewer")
    def viewer():
        from ..viewer import viewer_status

        return viewer_status(rt)

    @app.get("/events")
    async def events(request: Request):
        queue = rt.subscribe()

        async def stream():
            try:
                yield f"data: {json.dumps({'type': 'hello', 'ts': time.time()})}\n\n"
                while await request.is_disconnected() is False:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=25)
                        yield f"data: {json.dumps(event, default=str)}\n\n"
                    except (asyncio.TimeoutError, TimeoutError):
                        yield ": keepalive\n\n"
            finally:
                rt.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _PLACEHOLDER_PAGE

    return app


_PLACEHOLDER_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>agentd</title>
<style>body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;max-width:60rem}
code{background:#f4f4f5;padding:2px 5px;border-radius:4px}</style></head><body>
<h1>agentd</h1>
<p>The control UI ships in the desktop shell. This page exists so a browser can reach the
API after you open the tunnel:</p>
<pre>ssh -N -L 8765:127.0.0.1:8765 -L 6080:127.0.0.1:6080 user@your-server</pre>
<p>Endpoints: <code>/api/state</code> · <code>/api/memory</code> · <code>/api/risks</code>
· <code>/api/audit</code> · <code>/events</code> (SSE) · <code>/healthz</code></p>
</body></html>"""
