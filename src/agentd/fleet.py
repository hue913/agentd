"""Fleet: multi-host metrics, health polling and batch execution.

Everything here rides on the existing SSHHub transport and safety gate. Two
command sources exist and they are treated very differently:

* the built-in metrics probe — a module-level constant no caller can influence,
  so it runs with `approved=True` (same precedent as `ssh_env.PROBE_SCRIPT`);
* `HostSpec.metrics_cmd` — written by the operator into agentd.json and executed
  on the remote host with that host's credentials. It is NEVER trusted: it goes
  through `classify()` like any model-authored command, and anything not
  provably read-only needs a one-shot human approval before it runs.

Failures are honest: an unreachable host reports `online: false` with a short
reason, never a fabricated online state or leaked stack details.
"""

from __future__ import annotations

import os
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .envs.ssh_env import SSHError, SSHHub
from .log import get_logger
from .safety.audit import AuditLog, AuditRecord
from .safety.gate import LEVEL_ALLOW, classify, initial_readonly

log = get_logger("agentd.fleet")

# Read-only fleet metrics probe. POSIX sh; each section fails independently
# (`2>/dev/null` + empty default) so a host without /proc/loadavg or a busybox
# ps still reports whatever it can. `df -P` is the POSIX form (no GNU-only
# --output), so it works on alpine/busybox remotes too.
METRICS_PROBE_SCRIPT = (
    "echo arch=$(uname -m 2>/dev/null); "
    "echo mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null); "
    "echo mem_avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null); "
    "echo disk_pct=$(df -P / 2>/dev/null | awk 'NR==2{gsub(/%/,\"\",$5);print $5}'); "
    "echo load1=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null); "
    "echo procs=$(ps 2>/dev/null | wc -l)"
)

PROBE_TIMEOUT_S = 8
STDOUT_TAIL_BYTES = 2048
CONNECT_TIMEOUT_S = 3.0
MAX_REASON = 200

_KV_RE = re.compile(r"([a-z][a-z0-9_]*)=(.*)")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _short(text: object) -> str:
    """First line, bounded — the operator needs the outcome, not a traceback."""
    lines = str(text or "").strip().splitlines()
    return (lines[0] if lines else "")[:MAX_REASON]


def _int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_metrics(text: str) -> dict | None:
    """Parse key=value probe output into fleet metrics.

    Returns None when the host produced nothing usable (empty output, a login
    banner, a shell that does not understand the probe) — callers must treat
    that as "unknown", never as zeros.
    """
    kv: dict[str, str] = {}
    for line in (text or "").splitlines():
        match = _KV_RE.fullmatch(line.strip())
        if match:
            kv[match.group(1)] = match.group(2).strip()
    if not kv.get("arch"):
        return None
    mem_total = _int(kv.get("mem_total_kb"))
    mem_avail = _int(kv.get("mem_avail_kb"))
    return {
        "arch": kv["arch"],
        "mem_total": mem_total,                              # KB, per /proc/meminfo
        "mem_used": max(0, mem_total - mem_avail) if mem_total else 0,   # KB
        "disk_pct": _int(kv.get("disk_pct"), -1),            # -1 = unknown
        "load1": _float(kv.get("load1")),
        "procs": _int(kv.get("procs")),
    }


class Fleet:
    """Per-host metrics cache, background health polling and batch execution."""

    def __init__(self, hub: SSHHub, audit: AuditLog | None = None, approver=None,
                 metrics_ttl: float | None = None, health_interval: float | None = None):
        self.hub = hub
        self.audit = audit
        self.approver = approver
        # TTLs come from the environment with sane defaults; explicit arguments
        # win so tests can shrink them without touching the process env.
        self.metrics_ttl = metrics_ttl if metrics_ttl is not None \
            else _env_float("AGENTD_FLEET_METRICS_TTL", 60.0)
        self.health_interval = health_interval if health_interval is not None \
            else _env_float("AGENTD_FLEET_HEALTH_INTERVAL", 60.0)
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._poller: threading.Thread | None = None
        self._poller_stop = threading.Event()
        # In-flight batch dedup: identical concurrent exec_many requests share
        # one approval and one execution instead of stacking a second approval
        # card (and a second execution) on the operator.
        self._batch_lock = threading.Lock()
        self._inflight_batches: dict[tuple, dict] = {}

    # -- cache -------------------------------------------------------------
    def _peek(self, label: str) -> dict:
        with self._lock:
            entry = self._cache.get(label)
        return dict(entry) if entry else {}

    def _merge(self, label: str, fields: dict) -> dict:
        with self._lock:
            entry = dict(self._cache.get(label) or _empty_entry())
            entry.update(fields)
            self._cache[label] = entry
        return dict(entry)

    def online(self, label: str) -> bool:
        """Cached liveness only — never probes, safe to call per request."""
        return bool(self._peek(label).get("online", False))

    # -- metrics -----------------------------------------------------------
    def metrics(self, label: str, force: bool = False) -> dict:
        """Metrics for one host, served from the TTL cache when fresh."""
        entry = self._peek(label)
        if not force and entry and time.time() - entry.get("metrics_ts", 0.0) < self.metrics_ttl:
            return entry
        return self.collect(label)

    def collect(self, label: str) -> dict:
        """Probe one host now (bypassing the TTL) and refresh its cache entry."""
        try:
            spec = self.hub.get_host(label)
        except SSHError:
            return self._merge(label, {"metrics": None, "metrics_ts": time.time(),
                                       "online": False, "unreachable_reason": "unknown host"})
        custom = (getattr(spec, "metrics_cmd", "") or "").strip()
        if custom:
            result = self._run_custom(label, custom)
        else:
            result = self._run_probe(label, METRICS_PROBE_SCRIPT)
        now = time.time()
        if result is None:      # denied by the gate / approval flow
            return dict(self._peek(label))
        rc, stdout, stderr = result
        if rc != 0:
            reason = "timed out" if rc == 124 else (_short(stderr) or f"probe failed (rc={rc})")
            return self._merge(label, {"metrics": None, "metrics_ts": now, "online": False,
                                       "unreachable_reason": reason})
        parsed = parse_metrics(stdout)
        if parsed is None:
            return self._merge(label, {"metrics": None, "metrics_ts": now, "online": False,
                                       "unreachable_reason": "probe returned no usable data"})
        return self._merge(label, {"metrics": parsed, "metrics_ts": now, "online": True,
                                   "unreachable_reason": None})

    def _run_probe(self, label: str, script: str,
                   action: str = "readonly-probe") -> tuple[int, str, str] | None:
        """Execute a probe through the hub; audit it under the given marker."""
        try:
            result = self.hub.exec(label, script, timeout=PROBE_TIMEOUT_S, approved=True,
                                   action=action)
        except SSHError:
            self._merge(label, {"metrics": None, "metrics_ts": time.time(),
                                "online": False, "unreachable_reason": "unknown host"})
            return None
        return result.rc, result.stdout, result.stderr

    def _run_custom(self, label: str, command: str) -> tuple[int, str, str] | None:
        """Run an operator-authored metrics_cmd through the gate, untrusted.

        Only commands the gate classifies as provably read-only run without a
        human; everything else needs a one-shot approval. Denials are audited
        and reported as an offline host with a short reason.
        """
        verdict = classify(command)
        if verdict.hard_blocked:
            self._audit_probe_denial(label, command, verdict.level, "blocked by gate")
            self._merge(label, {"metrics": None, "metrics_ts": time.time(), "online": False,
                                "unreachable_reason": "custom metrics command blocked by policy"})
            return None
        if verdict.level != LEVEL_ALLOW or not initial_readonly(command):
            spec = self.hub.get_host(label)
            payload = {"tool": "fleet.metrics", "host": label, "command": command,
                       "policy": getattr(spec, "policy", "standard") or "standard",
                       **verdict.as_dict()}
            if not self._approved(payload):
                reason = "custom metrics command requires approval"
                self._audit_probe_denial(label, command, verdict.level, reason)
                self._merge(label, {"metrics": None, "metrics_ts": time.time(),
                                    "online": False,
                                    "unreachable_reason": reason})
                return None
        return self._run_probe(label, command, action="metrics-cmd")

    def _audit_probe_denial(self, label: str, command: str, level: str, error: str) -> None:
        if self.audit is None:
            return
        try:
            self.audit.write(AuditRecord(host=label, command=command, level=level,
                                         action="readonly-probe", error=error))
        except Exception as exc:    # an audit hiccup must not take down the API
            log.warning("fleet audit write failed: %s", exc)

    def metrics_report(self, label: str, force: bool = False) -> dict:
        """The /api/hosts/{label}/metrics shape."""
        entry = self.metrics(label, force=force)
        return {"metrics": entry.get("metrics"),
                "ts": entry.get("metrics_ts") or 0.0,
                "online": bool(entry.get("online")),
                "unreachable_reason": entry.get("unreachable_reason")}

    # -- health polling (F9) -------------------------------------------------
    def health_check(self, label: str) -> dict:
        """Light liveness: one TCP connect. No shell, no metrics, no gate."""
        try:
            spec = self.hub.get_host(label)
        except SSHError:
            return self._merge(label, {"online": False, "unreachable_reason": "unknown host",
                                       "health_ts": time.time()})
        try:
            with socket.create_connection((spec.host, int(spec.port or 22)),
                                          timeout=CONNECT_TIMEOUT_S):
                pass
        except OSError as exc:
            return self._merge(label, {"online": False,
                                       "unreachable_reason": _short(exc) or "unreachable",
                                       "health_ts": time.time()})
        return self._merge(label, {"online": True, "unreachable_reason": None,
                                   "health_ts": time.time()})

    def poll_once(self) -> None:
        for label in self.hub.hosts():
            try:
                self.health_check(label)
            except Exception as exc:    # one bad host must not stop the pass
                log.warning("fleet health check failed for %s: %s", label, exc)

    def start_polling(self, interval: float | None = None) -> None:
        """Poll host liveness every `interval` seconds from a daemon thread.

        Same shape as the PTY reaper: the panel/API should not have to be open
        for liveness to be tracked, and process exit never waits on the thread.
        """
        if self._poller and self._poller.is_alive():
            return
        self._poller_stop.clear()
        iv = max(0.05, float(interval if interval is not None else self.health_interval))

        def _loop() -> None:
            # Event.wait doubles as an interruptible sleep, so stop_polling()
            # returns promptly instead of after a full interval.
            while not self._poller_stop.wait(iv):
                try:
                    self.poll_once()
                except Exception as exc:
                    log.warning("fleet health pass failed: %s", exc)

        self._poller = threading.Thread(target=_loop, name="agentd-fleet-health", daemon=True)
        self._poller.start()

    def stop_polling(self) -> None:
        self._poller_stop.set()
        if self._poller is not None:
            self._poller.join(timeout=5)
            self._poller = None

    # -- listing -------------------------------------------------------------
    def host_summaries(self, tag: str | None = None) -> list[dict]:
        """The GET /api/hosts payload: config + cached liveness + last action."""
        out: list[dict] = []
        for label in self.hub.hosts():
            spec = self.hub.get_host(label)
            tags = list(getattr(spec, "tags", []) or [])
            if tag and tag not in tags:
                continue
            entry = self._peek(label)
            out.append({
                "label": label,
                "host": spec.host,
                "port": spec.port,
                "user": spec.user,
                "jump": spec.jump,
                "policy": (getattr(spec, "policy", "") or "standard"),
                "tags": tags,
                "online": bool(entry.get("online", False)),
                "unreachable_reason": entry.get("unreachable_reason"),
                "metrics": entry.get("metrics"),
                "metrics_ts": entry.get("metrics_ts") or None,
                "last_action": self.last_action(label),
            })
        return out

    def last_action(self, label: str) -> dict | None:
        """Most recent audited command for this host, if the audit log is on."""
        if self.audit is None:
            return None
        try:
            rows = self.audit.tail(limit=1, host=label)
        except Exception:
            return None
        if not rows:
            return None
        row = rows[-1]
        return {"ts": row.get("ts"), "command": row.get("command", ""),
                "level": row.get("level"), "rc": row.get("rc")}

    # -- batch execution (F6+F7) ----------------------------------------------
    def expand_labels(self, labels: list) -> list[str]:
        """Resolve the request's label list; `tag:xxx` expands to all tagged hosts."""
        if not isinstance(labels, list) or not labels:
            raise ValueError("'labels' must be a non-empty list")
        known = set(self.hub.hosts())
        resolved: list[str] = []
        for item in labels:
            name = str(item).strip()
            if not name:
                continue
            if name.startswith("tag:"):
                tag = name[4:].strip()
                if not tag:
                    raise ValueError(f"empty tag selector '{name}'")
                matches = [label for label in self.hub.hosts()
                           if tag in (getattr(self.hub.get_host(label), "tags", []) or [])]
                if not matches:
                    raise ValueError(f"no host carries tag '{tag}'")
                for match in matches:
                    if match not in resolved:
                        resolved.append(match)
            elif name in known:
                if name not in resolved:
                    resolved.append(name)
            else:
                raise ValueError(f"unknown host '{name}'. known: {', '.join(sorted(known)) or 'none'}")
        if not resolved:
            raise ValueError("no hosts selected")
        return resolved

    def exec_many(self, labels: list, command: str, mode: str = "parallel",
                  order: list | None = None) -> dict:
        """Run one command on many hosts, always behind a single human approval.

        Approval is unconditional — batch actions multiply blast radius, so
        even a provably read-only command waits for the operator. Per-host
        audits land through the normal SSHHub.exec path.

        Idempotent against retries: while an identical batch (same hosts,
        command and mode) is still parked in the approval queue, a duplicate
        request joins it and receives the same outcome instead of raising a
        second approval card. This makes the frontend's "re-POST while
        waiting for the human" pattern safe — approving one card can never
        execute the batch twice.
        """
        command = str(command or "").strip()
        if not command:
            raise ValueError("'command' is required")
        if mode not in ("parallel", "rolling"):
            raise ValueError("mode must be 'parallel' or 'rolling'")
        resolved = self.expand_labels(labels)
        sequence = resolved
        if mode == "rolling" and order:
            wanted = [str(x) for x in order]
            missing = [x for x in wanted if x not in resolved]
            if missing:
                raise ValueError(f"order references hosts outside labels: {', '.join(missing)}")
            # Ordered hosts run first, in the requested sequence; anything the
            # caller left out of `order` trails in listed order.
            sequence = wanted + [l for l in resolved if l not in wanted]

        key = (mode, command, tuple(sequence))
        # The owner parks inside the approver for up to AGENTD_APPROVAL_TIMEOUT;
        # joiners wait for that plus a margin for the execution itself.
        join_timeout = _env_float("AGENTD_APPROVAL_TIMEOUT", 300.0) + 60.0
        with self._batch_lock:
            record = self._inflight_batches.get(key)
            if record is None:
                record = {"event": threading.Event(), "result": None}
                self._inflight_batches[key] = record
                owner = True
            else:
                owner = False
        if not owner:
            if not record["event"].wait(join_timeout) or record["result"] is None:
                return {"blocked": True,
                        "reason": "an identical batch is already in flight"}
            return record["result"]

        try:
            return self._exec_many_owned(resolved, sequence, command, mode, record)
        finally:
            record["event"].set()
            with self._batch_lock:
                if self._inflight_batches.get(key) is record:
                    self._inflight_batches.pop(key, None)

    def _exec_many_owned(self, resolved: list[str], sequence: list[str], command: str,
                         mode: str, record: dict) -> dict:
        """The single owner's path: raise the approval, run, publish the result."""
        payload = {"tool": "fleet.exec_many", "host": ", ".join(resolved),
                   "labels": list(resolved), "command": command, "mode": mode,
                   "risk": "batch command on multiple hosts", "policy": "standard"}
        if not self._approved(payload):
            reason = ("batch execution requires approval" if self.approver
                      else "no approver configured")
            record["result"] = {"blocked": True, "reason": reason}
            if self.audit is not None:
                self.audit.write(AuditRecord(host=", ".join(resolved), command=command,
                                             action="fleet.exec-many", level="confirm",
                                             reasons=["batch execution always requires approval"],
                                             error=reason))
            return record["result"]

        def run_one(label: str) -> dict:
            try:
                result = self.hub.exec(label, command, approved=True)
                return {"label": label, "rc": result.rc,
                        "stdout_tail": result.stdout[-STDOUT_TAIL_BYTES:],
                        "stderr_tail": result.stderr[-STDOUT_TAIL_BYTES:],
                        "ok": result.ok}
            except SSHError as exc:
                # Unreachable/unknown hosts fail as their own row; they must
                # never take the other hosts' results down with them.
                return {"label": label, "rc": -1, "stdout_tail": "", "stderr_tail": "",
                        "ok": False, "error": _short(exc) or "unreachable"}
            except Exception:
                log.warning("exec-many row failed for %s", label, exc_info=True)
                return {"label": label, "rc": -1, "stdout_tail": "", "stderr_tail": "",
                        "ok": False, "error": "execution failed"}

        if mode == "parallel":
            # Thread pool mirrors asyncio.to_thread semantics: subprocess work
            # off the loop, capped so a 50-host fleet cannot spawn 50 ssh's.
            workers = min(8, max(1, len(sequence)))
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="agentd-fleet") as pool:
                results = list(pool.map(run_one, sequence))
            report = {"results": results, "stopped_at": None}
        else:
            results = []
            for label in sequence:
                row = run_one(label)
                results.append(row)
                if not row["ok"]:
                    report = {"results": results, "stopped_at": label}
                    break
            else:
                report = {"results": results, "stopped_at": None}
        # Publish for joiners before returning: they receive the exact same
        # outcome, never a second execution.
        record["result"] = report
        return report

    def _approved(self, payload: dict) -> bool:
        if self.approver is None:
            return False
        return bool(self.approver(payload))


def _empty_entry() -> dict:
    return {"metrics": None, "metrics_ts": 0.0, "online": False, "unreachable_reason": None}
