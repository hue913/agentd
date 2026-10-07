"""Host metrics for the operations panel.

Design constraint: this module runs arbitrary-looking commands, so it must not
take command strings from callers. Every probe is a module-level constant and
`sh -c` runs with the constant as argv[2] -- there is no code path where request
data reaches a shell. Field names are validated against an allowlist before
being echoed into `journalctl -u <unit>`, so a crafted unit name cannot become
an option (`-u --output=...`).

Everything degrades: a host without systemd still returns /proc data.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

TIMEOUT = 8

# Units the panel offers by default. Arbitrary units are allowed (see
# _safe_unit) because a single-purpose agent box is not a fixed set.
DEFAULT_UNITS = [
    "agentd.service",
    "ssh.service",
    "nginx.service",
    "mysql.service",
    "docker.service",
    "fail2ban.service",
]

_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
_UNIT_QUERY_ARGS = ("--output=json", "-o", "cat", "--no-pager", "-n")


def _run(argv: list[str], timeout: int = TIMEOUT) -> tuple[int, str, str]:
    if not shutil.which(argv[0]):
        return 127, "", f"{argv[0]} not installed"
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as exc:  # never let a probe crash the panel
        return 1, "", f"{type(exc).__name__}: {exc}"


def _safe_unit(unit: str) -> str | None:
    return unit if _UNIT_RE.match(unit or "") else None


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def overview() -> dict:
    """One pass over /proc plus a single uptime call."""
    mem_total_kb = mem_avail_kb = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            mem_total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            mem_avail_kb = int(line.split()[1])
    swap_total_kb = swap_free_kb = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("SwapTotal:"):
            swap_total_kb = int(line.split()[1])
        elif line.startswith("SwapFree:"):
            swap_free_kb = int(line.split()[1])

    load = [float(x) for x in _read("/proc/loadavg").split()[:3]] or [0.0, 0.0, 0.0]
    boot_raw = _read("/proc/uptime").split()
    uptime_s = float(boot_raw[0]) if boot_raw else 0.0

    disk = []
    rc, out, _ = _run(["df", "-B1", "--output=source,size,used,avail,pcent,target", "-x", "tmpfs",
                       "-x", "devtmpfs"])
    if rc == 0:
        for line in out.splitlines()[1:]:
            parts = line.split(None, 5)
            if len(parts) == 6:
                disk.append({
                    "source": parts[0], "size": int(parts[1]), "used": int(parts[2]),
                    "avail": int(parts[3]), "use_pct": parts[4].rstrip("%"), "mount": parts[5],
                })

    return {
        "ts": time.time(),
        "uptime_s": uptime_s,
        "load1": load[0], "load5": load[1], "load15": load[2],
        "mem_total_kb": mem_total_kb, "mem_avail_kb": mem_avail_kb,
        "mem_used_pct": round(100 * (1 - mem_avail_kb / mem_total_kb), 1) if mem_total_kb else 0.0,
        "swap_total_kb": swap_total_kb,
        "swap_used_kb": swap_total_kb - swap_free_kb,
        "disk": disk,
    }


def processes(limit: int = 15, sort: str = "rss") -> dict:
    rc, out, _ = _run(["ps", "-eo", "pid,ppid,user,rss,pcpu,etimes,comm", "--no-headers"])
    if rc != 0:
        return {"processes": [], "error": out or "ps failed"}
    key = {"rss": 3, "cpu": 4, "etime": 5}.get(sort, 3)
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        try:
            rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "user": parts[2],
                         "rss_kb": int(parts[3]), "cpu_pct": float(parts[4]),
                         "etime_s": int(parts[5]), "comm": parts[6]})
        except ValueError:
            continue
    rows.sort(key=lambda r: r[{3: "rss_kb", 4: "cpu_pct", 5: "etime_s"}[key]], reverse=True)
    return {"processes": rows[: max(1, min(limit, 100))]}


def services(units: list[str] | None = None) -> dict:
    if not shutil.which("systemctl"):
        return {"services": [], "error": "systemd not present on this host"}
    wanted = [u for u in (units or DEFAULT_UNITS) if _safe_unit(u)]
    out_rows = []
    for unit in wanted:
        rc, out, err = _run(["systemctl", "is-active", "--quiet", unit])
        rc2, detail, _ = _run(["systemctl", "show", unit, "--property=ActiveState,SubState,MainPID,"
                               "MemoryCurrent,ExecMainStartTimestamp", "--no-pager"])
        props = {}
        if rc2 == 0:
            for line in detail.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    props[k] = v
        out_rows.append({
            "unit": unit,
            "active": bool(rc == 0),
            "active_state": props.get("ActiveState", "unknown"),
            "sub_state": props.get("SubState", ""),
            "main_pid": int(props["MainPID"]) if props.get("MainPID", "").isdigit() else 0,
            "memory_bytes": int(props["MemoryCurrent"]) if props.get("MemoryCurrent", "").isdigit() else 0,
            "since": props.get("ExecMainStartTimestamp", ""),
            "error": "" if rc == 0 else (err.strip() or "inactive"),
        })
    return {"services": out_rows}


def logs(unit: str, lines: int = 200) -> dict:
    safe = _safe_unit(unit)
    if not safe:
        return {"unit": unit, "entries": [], "error": "invalid unit name"}
    if not shutil.which("journalctl"):
        return {"unit": safe, "entries": [], "error": "journalctl not available"}
    n = max(1, min(int(lines), 2000))
    rc, out, err = _run(["journalctl", "-u", safe, *_UNIT_QUERY_ARGS, "-n", str(n)])
    if rc != 0:
        return {"unit": safe, "entries": [], "error": err.strip() or "journalctl failed"}
    return {"unit": safe, "entries": out.splitlines(), "lines": n}


def snapshot(sort: str = "rss") -> dict:
    """Everything the panel shows, in one call. Used by the 15s SSE tick."""
    return {"overview": overview(), "processes": processes(limit=12, sort=sort).get("processes", []),
            "services": services().get("services", [])}
