"""Built-in tools: local shell/files, SSH operations, memory inspection, human handoff.

Every one of these is a thin wrapper the kernel can call; each declares its risk
so the approval rule lives in one place instead of scattered through handlers.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from ..safety.gate import classify, initial_readonly
from .spec import RISK_DANGEROUS, RISK_READ, RISK_WRITE, ToolResult, ToolSpec

MAX_OUTPUT = 40_000
MAX_FILE_BYTES = 200_000


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    head = text[: MAX_OUTPUT // 2]
    tail = text[-MAX_OUTPUT // 2 :]
    return f"{head}\n... [{len(text) - MAX_OUTPUT} chars elided] ...\n{tail}"


def local_exec(command: str, timeout: int = 60, approved: bool = False, ctx=None) -> ToolResult:
    """Run a shell command locally.

    shell=True stays because an ops agent needs pipes and globs, so the safety
    boundary is not the API choice but this rule: anything the gate cannot prove
    read-only requires a human, and everything is classified before it runs.
    """
    verdict = classify(command)
    human_ok = approved or bool(ctx and ctx.approved)
    if verdict.hard_blocked and not human_ok:
        return ToolResult(ok=False, risk=RISK_DANGEROUS,
                          error=f"blocked by safety gate: {'; '.join(verdict.reasons)}")
    if not human_ok and not initial_readonly(command):
        return ToolResult(
            ok=False, risk=RISK_WRITE,
            error=(f"'{command}' is not provably read-only"
                   + (f" ({'; '.join(verdict.reasons)})" if verdict.reasons else "")
                   + "; re-request with approved=true after the user confirms"),
        )
    t0 = time.time()
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, risk=RISK_WRITE, error=f"timed out after {timeout}s",
                          duration_ms=int((time.time() - t0) * 1000))
    except OSError as exc:
        return ToolResult(ok=False, error=f"could not run: {exc}")
    return ToolResult(
        ok=proc.returncode == 0,
        output=_clip((proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")),
        data={"rc": proc.returncode, "level": verdict.level, "reasons": verdict.reasons},
        risk=RISK_READ if verdict.level == RISK_READ or verdict.level == "allow" else RISK_WRITE,
        duration_ms=int((time.time() - t0) * 1000),
    )


def fs_read(path: str, max_bytes: int = MAX_FILE_BYTES) -> ToolResult:
    target = Path(path).expanduser()
    if not target.exists():
        return ToolResult(ok=False, error=f"no such file: {path}")
    if target.is_dir():
        return ToolResult(ok=False, error=f"{path} is a directory; use fs.ls")
    data = target.read_bytes()[: int(max_bytes)]
    return ToolResult(ok=True, output=_clip(data.decode("utf-8", "replace")),
                      data={"bytes": len(data), "truncated": target.stat().st_size > len(data)})


def fs_write(path: str, content: str, append: bool = False) -> ToolResult:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if append:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(content)
    else:
        target.write_text(content, encoding="utf-8")
    return ToolResult(ok=True, output=f"wrote {len(content)} chars to {path}", risk=RISK_WRITE)


def fs_ls(path: str = ".") -> ToolResult:
    target = Path(path).expanduser()
    if not target.exists():
        return ToolResult(ok=False, error=f"no such path: {path}")
    if target.is_file():
        return ToolResult(ok=True, output=f"{target.name}\t{target.stat().st_size} bytes")
    rows = []
    for entry in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))[:400]:
        rows.append(f"{'d' if entry.is_dir() else '-'} {entry.name}\t{entry.stat().st_size}")
    return ToolResult(ok=True, output="\n".join(rows))


def _hub(ctx):
    if ctx is None or getattr(ctx, "ssh", None) is None:
        return None
    return ctx.ssh


def ssh_exec(host: str, command: str, timeout: int = 60, approved: bool = False, ctx=None) -> ToolResult:
    hub = _hub(ctx)
    if hub is None:
        return ToolResult(ok=False, error="no SSH hub configured (add a host with agentd host add)")
    human_ok = approved or bool(ctx and ctx.approved)
    try:
        result = hub.exec(host, command, timeout=timeout, approved=human_ok,
                          session=getattr(ctx, "session", None), episode_id=getattr(ctx, "episode_id", None))
    except Exception as exc:
        return ToolResult(ok=False, risk=RISK_DANGEROUS, error=str(exc))
    return ToolResult(
        ok=result.rc == 0, output=_clip(result.stdout + (("\n[stderr]\n" + result.stderr) if result.stderr else "")),
        data={"rc": result.rc, "level": result.level, "reasons": result.reasons,
              "duration_ms": result.duration_ms},
        risk=RISK_READ if result.level == "allow" else RISK_WRITE,
        duration_ms=result.duration_ms,
    )


def ssh_tail_log(host: str, path: str, lines: int = 200, ctx=None) -> ToolResult:
    hub = _hub(ctx)
    if hub is None:
        return ToolResult(ok=False, error="no SSH hub configured")
    try:
        out = hub.tail_log(host, path, lines=lines)
    except Exception as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, output=_clip(out))


def ssh_ls(host: str, path: str = ".", ctx=None) -> ToolResult:
    hub = _hub(ctx)
    if hub is None:
        return ToolResult(ok=False, error="no SSH hub configured")
    try:
        return ToolResult(ok=True, output="\n".join(hub.ls(host, path)))
    except Exception as exc:
        return ToolResult(ok=False, error=str(exc))


def ssh_port_check(host: str, port: int, ctx=None) -> ToolResult:
    hub = _hub(ctx)
    if hub is None:
        return ToolResult(ok=False, error="no SSH hub configured")
    try:
        listening = hub.port_check(host, port)
    except Exception as exc:
        return ToolResult(ok=False, error=str(exc))
    return ToolResult(ok=True, output=f"{host}:{port} " + ("LISTENING" if listening else "closed"),
                      data={"listening": listening})


def ssh_upload(host: str, local: str, remote: str, ctx=None) -> ToolResult:
    hub = _hub(ctx)
    if hub is None:
        return ToolResult(ok=False, error="no SSH hub configured")
    if not os.path.exists(os.path.expanduser(local)):
        return ToolResult(ok=False, error=f"no such local file: {local}")
    try:
        report = hub.transfer_resumable(host, local, remote)
    except Exception as exc:
        return ToolResult(ok=False, risk=RISK_WRITE, error=str(exc))
    return ToolResult(ok=bool(report.get("ok")), output=str(report), data=report, risk=RISK_WRITE)


def memory_stats(ctx=None) -> ToolResult:
    kernel = getattr(ctx, "kernel", None) if ctx else None
    if kernel is None:
        return ToolResult(ok=False, error="no memory kernel attached")
    return ToolResult(ok=True, output=str(kernel.stats()), data=kernel.stats())


def memory_recall(state: str, limit: int = 8, ctx=None) -> ToolResult:
    kernel = getattr(ctx, "kernel", None) if ctx else None
    if kernel is None:
        return ToolResult(ok=False, error="no memory kernel attached")
    neighbors = kernel.retriever.neighbors(state)[:limit]
    lines = [f"sim={n.sim:.3f} ret={n.ret:+.3f} action={n.action}" for n in neighbors]
    return ToolResult(ok=True, output="\n".join(lines) or "no similar past states",
                      data={"count": len(neighbors)})


def tools_expand(name: str, ctx=None) -> ToolResult:
    bus = getattr(ctx, "extra", {}).get("bus") if ctx else None
    if bus is None:
        return ToolResult(ok=False, error="tool registry not attached")
    info = bus.expand(name)
    return ToolResult(ok="error" not in info, output=str(info), data=info)


def human_request(reason: str, ctx=None) -> ToolResult:
    """Ask for a takeover — login, 2FA, a captcha, or an irreversible click."""
    approver = getattr(ctx, "approver", None) if ctx else None
    if approver is None:
        return ToolResult(ok=False, error="no human channel attached to this session")
    granted = bool(approver({"tool": "human.request", "args": {"reason": reason}, "risk": "handoff"}))
    return ToolResult(ok=granted, output="human took over" if granted else "human declined or absent",
                      risk=RISK_READ)


_S = ToolSpec

BUILTIN_TOOLS: list[ToolSpec] = [
    _S("local.exec", "Run a shell command on the machine hosting agentd.",
       {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": 60},
                                         "approved": {"type": "boolean", "default": False}},
        "required": ["command"]}, "builtin", RISK_WRITE, local_exec),
    _S("fs.read", "Read a local text file.",
       {"type": "object", "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer", "default": MAX_FILE_BYTES}},
        "required": ["path"]}, "builtin", RISK_READ, fs_read),
    _S("fs.write", "Write or overwrite a local text file.",
       {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                         "append": {"type": "boolean", "default": False}},
        "required": ["path", "content"]}, "builtin", RISK_WRITE, fs_write),
    _S("fs.ls", "List a local directory.",
       {"type": "object", "properties": {"path": {"type": "string", "default": "."}}, "required": []},
       "builtin", RISK_READ, fs_ls),

    _S("ssh.exec", "Run a command on a configured SSH host through the safety gate.",
       {"type": "object", "properties": {"host": {"type": "string"}, "command": {"type": "string"},
                                         "timeout": {"type": "integer", "default": 60},
                                         "approved": {"type": "boolean", "default": False}},
        "required": ["host", "command"]}, "builtin", RISK_WRITE, ssh_exec,
       examples=["ssh.exec(host='app-01', command='systemctl status nginx')"]),
    _S("ssh.tail_log", "Read the tail of a remote log file.",
       {"type": "object", "properties": {"host": {"type": "string"}, "path": {"type": "string"},
                                         "lines": {"type": "integer", "default": 200}},
        "required": ["host", "path"]}, "builtin", RISK_READ, ssh_tail_log),
    _S("ssh.ls", "List a remote directory.",
       {"type": "object", "properties": {"host": {"type": "string"}, "path": {"type": "string", "default": "."}},
        "required": ["host"]}, "builtin", RISK_READ, ssh_ls),
    _S("ssh.port_check", "Check whether a port is listening on a remote host.",
       {"type": "object", "properties": {"host": {"type": "string"}, "port": {"type": "integer"}},
        "required": ["host", "port"]}, "builtin", RISK_READ, ssh_port_check),
    _S("ssh.upload", "Upload a local file to a remote host, chunked with md5 verification.",
       {"type": "object", "properties": {"host": {"type": "string"}, "local": {"type": "string"},
                                         "remote": {"type": "string"}},
        "required": ["host", "local", "remote"]}, "builtin", RISK_WRITE, ssh_upload),

    _S("memory.stats", "Report how much experience the kernel has accumulated.",
       {"type": "object", "properties": {}, "required": []}, "builtin", RISK_READ, memory_stats),
    _S("memory.recall", "Show the past states most similar to this one and what worked there.",
       {"type": "object", "properties": {"state": {"type": "string"}, "limit": {"type": "integer", "default": 8}},
        "required": ["state"]}, "builtin", RISK_READ, memory_recall),
    _S("tools.expand", "Fetch the full schema of one tool by name.",
       {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
       "builtin", RISK_READ, tools_expand, hidden=True),
    _S("human.request", "Request a human takeover (login, 2FA, captcha, irreversible step).",
       {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
       "builtin", RISK_READ, human_request),
]


def register_builtins(bus) -> int:
    for spec in BUILTIN_TOOLS:
        bus.register(spec, replace=True)
    return len(BUILTIN_TOOLS)
