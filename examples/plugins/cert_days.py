"""Example plugin: drop this file in `plugins/` and agentd registers the tools.

Two contracts are supported: `TOOLS = [...]`, or a `register(bus)` function when you
need to build tools from runtime state.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from agentd.toolbus.spec import RISK_READ, ToolResult, ToolSpec

_ENDATE = re.compile(r"notAfter=(.+)")


def cert_days_left(path: str = "/etc/ssl/certs/server.crt", host: str = "", ctx=None) -> ToolResult:
    """Days left on a certificate file, read locally or over SSH."""
    argv = ["openssl", "x509", "-in", path, "-noout", "-enddate"]
    if ctx is not None and getattr(ctx, "ssh", None) is not None and host:
        # the SSH hub quotes each argv itself, so `path` cannot become a shell word
        result = ctx.ssh.exec_argv(host, argv, timeout=30)
        text, err = result.stdout, result.stderr
    else:
        text, err = _local(argv)
    if not text:
        return ToolResult(ok=False, output=f"no certificate read ({err or 'empty output'})",
                          risk=RISK_READ)
    match = _ENDATE.search(text)
    if not match:
        return ToolResult(ok=False, output=text[:200], risk=RISK_READ)
    end = datetime.strptime(match.group(1).strip(), "%Y%m%d%H%M%SZ").replace(tzinfo=timezone.utc)
    days = (end - datetime.now(timezone.utc)).days
    verdict = "EXPIRED" if days < 0 else "expiring soon" if days < 21 else "ok"
    return ToolResult(ok=True, output=f"{host or 'local'}:{path} expires in {days} days ({verdict})",
                      data={"days_left": days, "verdict": verdict}, risk=RISK_READ)


def _local(argv: list[str]) -> tuple[str, str]:
    """Direct exec, no shell: an example must not teach shell=True."""
    import shutil
    import subprocess

    if not shutil.which(argv[0]):
        return "", "openssl not found on PATH"
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    return proc.stdout, proc.stderr


TOOLS = [
    ToolSpec(
        name="cert.days_left",
        description="Days until a TLS certificate expires, locally or on an SSH host.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "default": "/etc/ssl/certs/server.crt"},
            "host": {"type": "string", "default": ""}}, "required": []},
        source="plugin",
        risk=RISK_READ,
        handler=cert_days_left,
        examples=["cert.days_left(host='app-01', path='/etc/letsencrypt/live/site/fullchain.pem')"],
    ),
]
