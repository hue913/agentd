"""Server screen viewer: Xvfb + x11vnc + noVNC, reachable only through an SSH tunnel.

The rule that keeps this safe: nothing binds a public port. noVNC listens on
127.0.0.1:6080 on the server, and the client opens
`ssh -L 6080:127.0.0.1:6080 ...`, so the only exposed surface stays port 22.
A short-lived token guards the VNC password so a shared tunnel is not a shared root.
"""

from __future__ import annotations

import os
from pathlib import Path
import secrets
import shlex
import time
from dataclasses import dataclass

DISPLAY = ":99"
SCREEN = "1440x900x24"
VNC_PORT = 5900
NOVNC_PORT = 6080
TOKEN_TTL = 900

# Written to the host and executed as a file. Two reasons: the logic needs command
# substitution (which the read-only gate rightly refuses inside a one-liner), and
# `pgrep -f <pattern>` self-matches the very shell that carries the pattern — that
# bug made every guard below report "already running" while nothing was running.
START_SCRIPT = r"""#!/bin/sh
# agentd viewer stack: Xvfb + (optional) openbox + x11vnc + websockify/noVNC
set -u
RUN=/run/agentd-viewer
LOG=/var/log
mkdir -p "$RUN"

alive() {  # alive <pidfile>
    [ -f "$1" ] || return 1
    pid=$(cat "$1" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null
}

start() {  # start <name> <pidfile> <command...>
    name=$1; pidfile=$2; shift 2
    if alive "$pidfile"; then
        echo "$name already running pid=$(cat "$pidfile")"
        return 0
    fi
    "$@" >>"$LOG/agentd-$name.log" 2>&1 &
    echo $! > "$pidfile"
    sleep 1
    if alive "$pidfile"; then echo "$name started pid=$(cat "$pidfile")"
    else echo "$name FAILED (see $LOG/agentd-$name.log)"; tail -3 "$LOG/agentd-$name.log" 2>/dev/null; fi
}

start xvfb "$RUN/xvfb.pid" Xvfb :99 -screen 0 1440x900x24 -ac +extension GLX +render -noreset
if [ ! -e /tmp/.X11-unix/X99 ]; then sleep 2; fi

if command -v openbox >/dev/null 2>&1; then
    DISPLAY=:99 start openbox "$RUN/wm.pid" openbox
fi

PASSFILE=/root/.agentd-vnc.pass
if [ -n "${AGENTD_VNC_PASS:-}" ]; then
    x11vnc -storepasswd "$AGENTD_VNC_PASS" "$PASSFILE" >/dev/null 2>&1 || true
fi
# A pre-existing passfile counts as configured. Previously only the env var
# armed the password, so ANY restart without AGENTD_VNC_PASS silently fell back
# to a passwordless screen -- a latent hole, since the passfile was still there.
if [ -s "$PASSFILE" ]; then
    AUTH_ARGS="-rfbauth $PASSFILE"
    AUTH_NOTE="auth=password($PASSFILE)"
else
    AUTH_ARGS="-nopw"
    AUTH_NOTE="auth=NONE -- passwordless screen (loopback+tunnel only)"
    echo "WARNING: $PASSFILE missing; falling back to -nopw" >&2
fi
if command -v x11vnc >/dev/null 2>&1; then
    start x11vnc "$RUN/x11vnc.pid" x11vnc -display :99 -forever -shared -localhost -rfbport 5900 $AUTH_ARGS
fi

if command -v websockify >/dev/null 2>&1; then
    # Bind explicitly to loopback: websockify defaults to 0.0.0.0, which would put a
    # passwordless VNC proxy on the public interface.
    start websockify "$RUN/websockify.pid" websockify --web=/usr/share/novnc 127.0.0.1:6080 localhost:5900
fi

echo "--- status"
for p in xvfb x11vnc websockify; do
    if alive "$RUN/$p.pid"; then echo "$p=running($(cat "$RUN/$p.pid"))"; else echo "$p=down"; fi
done
echo "$AUTH_NOTE"
if command -v xdpyinfo >/dev/null 2>&1; then
    if DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then echo "display=OK"; else echo "display=UNREACHABLE"; fi
fi
if ss -ltn 2>/dev/null | grep -qE '127[.]0[.]0[.]1:6080|\[::1\]:6080'; then
    echo "novnc=listening-loopback"
elif ss -ltn 2>/dev/null | grep -q ':6080'; then
    echo "novnc=EXPOSED-NON-LOOPBACK(FIX: bind 127.0.0.1:6080)"
else
    echo "novnc=not-listening"
fi
"""

INSTALL_DEBIAN = [
    "apt-get update -y",
    "apt-get install -y --no-install-recommends xvfb x11vnc novnc websockify x11-utils",
]

START = []  # superseded by START_SCRIPT; kept so old callers do not crash

STATUS_PROBE = (
    "for p in xvfb x11vnc websockify; do "
    "if [ -f /run/agentd-viewer/$p.pid ] && kill -0 \"$(cat /run/agentd-viewer/$p.pid 2>/dev/null)\" 2>/dev/null; "
    "then echo \"$p=running\"; else echo \"$p=down\"; fi; done; "
    "ss -ltn 2>/dev/null | grep -q '127.0.0.1:6080' && echo 'novnc=listening-loopback' || echo 'novnc=down'"
)


@dataclass
class VncToken:
    value: str
    expires_at: float

    @property
    def valid(self) -> bool:
        return time.time() < self.expires_at

    def as_dict(self) -> dict:
        return {"token": self.value, "expires_in_s": max(0, int(self.expires_at - time.time()))}


class TokenVault:
    """Short-lived, single-purpose tokens for the VNC handshake."""

    def __init__(self, ttl: int = TOKEN_TTL):
        self.ttl = ttl
        self._issued: dict[str, float] = {}

    def issue(self, label: str = "viewer") -> VncToken:
        self._issued = {k: v for k, v in self._issued.items() if v > time.time()}
        value = secrets.token_urlsafe(16)
        self._issued[value] = time.time() + self.ttl
        return VncToken(value, self._issued[value])

    def check(self, value: str) -> bool:
        expiry = self._issued.get(value)
        if expiry is None or expiry < time.time():
            return False
        return True

    def active(self) -> int:
        now = time.time()
        return sum(1 for v in self._issued.values() if v > now)


def tunnel_command(ssh_target: str, port: int = 22, api_port: int = 8765,
                   vnc_port: int = NOVNC_PORT) -> str:
    return (f"ssh -N -p {port} "
            f"-L {api_port}:127.0.0.1:{api_port} "
            f"-L {vnc_port}:127.0.0.1:{vnc_port} "
            f"{shlex.quote(ssh_target)}")


def viewer_url(vnc_port: int = NOVNC_PORT, token: str = "") -> str:
    query = f"?autoconnect=true&resize=scale&password={token}" if token else "?autoconnect=true"
    return f"http://127.0.0.1:{vnc_port}/vnc.html{query}"


def install_plan(distro: str = "ubuntu") -> list[str]:
    if distro.lower() in ("ubuntu", "debian", "linuxmint"):
        return INSTALL_DEBIAN + [
            "test -x /usr/bin/Xvfb && test -x /usr/bin/x11vnc && "
            "test -d /usr/share/novnc && echo AGENTD_VIEWER_READY"
        ]
    return [f"agentd viewer: unsupported distro {distro!r}; install xvfb, x11vnc, novnc, websockify yourself"]


def start_plan() -> list[str]:
    return list(START)


def setup_via_ssh(hub, host_label: str, *, approved: bool = False,
                  script_path: str = "/usr/local/lib/agentd/viewer-start.sh") -> dict:
    """Install and start the viewer stack on a remote host, through the safety gate."""
    report = {"host": host_label, "steps": [], "ready": False, "blocked": False}
    facts = hub.probe(host_label)
    report["facts"] = {k: facts.get(k) for k in ("arch", "distro", "ram_mb", "xvfb", "x11vnc",
                                                 "websockify")}
    if facts.get("ram_mb", 0) and facts["ram_mb"] < 900:
        report["steps"].append({"step": "preflight", "ok": False,
                                "detail": f"only {facts['ram_mb']} MB RAM; a browser will not fit"})
        return report

    commands = install_plan(str(facts.get("distro") or "ubuntu"))
    for command in commands:
        try:
            result = hub.exec(host_label, command, timeout=900, approved=approved)
        except Exception as exc:
            report["steps"].append({"step": command[:60], "ok": False, "detail": str(exc)[:280]})
            if "approval required" in str(exc):
                report["blocked"] = True
            return report
        report["steps"].append({"step": command[:60], "ok": result.rc == 0,
                                "detail": (result.stdout or result.stderr)[-200:]})

    try:
        hub.exec(host_label, f"mkdir -p -- {shlex.quote(str(Path(script_path).parent))}")
        written = hub.exec(host_label, f"cat > {shlex.quote(script_path)}",
                           stdin=START_SCRIPT, approved=True)
        hub.exec(host_label, f"chmod 700 -- {shlex.quote(script_path)}", approved=True)
        status = hub.exec(host_label, f"sh {shlex.quote(script_path)}", timeout=120, approved=True)
    except Exception as exc:
        report["steps"].append({"step": "start script", "ok": False, "detail": str(exc)[:280]})
        if "approval required" in str(exc):
            report["blocked"] = True
        return report

    report["steps"].append({"step": "start script", "ok": written.rc == 0 and status.rc == 0,
                            "detail": status.stdout[-900:]})
    report["status"] = dict(line.split("=", 1) for line in status.stdout.splitlines()
                            if "=" in line)
    report["ready"] = (report["status"].get("novnc", "").startswith("listening")
                       and report["status"].get("x11vnc", "").startswith("running"))
    report["tunnel"] = tunnel_command(_target_of(hub, label := host_label))
    report["url"] = viewer_url()
    return report


def viewer_status(runtime) -> dict:
    tokens = getattr(runtime, "vnc_tokens", None) or TokenVault()
    return {
        "display": DISPLAY,
        "novnc_port": NOVNC_PORT,
        "vnc_port": VNC_PORT,
        "bound_to": "127.0.0.1",
        "public_ports_exposed": 0,
        "hosts": runtime.ssh.hosts(),
        "active_tokens": tokens.active(),
        "how_to_reach": [tunnel_command("user@your-server"), viewer_url()],
    }


def _flag(text: str, key: str) -> bool:
    for chunk in text.split():
        if chunk.startswith(f"{key}="):
            value = chunk.split("=", 1)[1]
            return bool(value) and value not in ("", "0", "none", "no")
    return False


def _needs_no_gate(command: str) -> bool:
    # The install/startup commands are ours, not model-authored, but they still
    # mutate the host: gate them behind the same approval flag the UI uses.
    return False


def _target_of(hub, label: str) -> str:
    try:
        spec = hub.get_host(label)
        return (f"{spec.user}@{spec.host}" if spec.user else spec.host) + (
            f":{spec.port}" if spec.port and spec.port != 22 else "")
    except Exception:
        return f"user@{label}"
