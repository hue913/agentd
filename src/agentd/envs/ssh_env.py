"""SSH transport built on the system `ssh`/`scp` binaries instead of a Python client.

Three reasons, each learned on this machine:
* system ssh inherits ~/.ssh/config, ssh-agent, certificates and ProxyJump rules,
  so hosts the user already trusts keep working without re-declaring them;
* paramiko pulls in cryptography, whose Rust wheel cannot build here (Intel mac,
  no OpenSSL) — dropping it removes the single most common install failure;
* `expect`-driven ssh hides output behind block buffering and its wrapper process
  self-matches pkill patterns. One argv plus deterministic stdout is safer.

Every command passes through the safety gate first. The one exception is
`PROBE_SCRIPT` below, a module constant that cannot be influenced by a model.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

from ..safety.audit import AuditLog, AuditRecord
from ..safety.gate import LEVEL_ALLOW, LEVEL_CONFIRM, Verdict, classify, initial_readonly

# Per-host enforcement tiers (fleet F4). The gate verdict itself is global and
# untouched; the policy only decides how that verdict is enforced per host.
POLICIES = ("standard", "strict", "permissive")

DEFAULT_TIMEOUT = 60
MAX_CAPTURE = 200_000
CHUNK = 32 * 1024 * 1024

# Trusted, static remote inventory. `approved=True` is only legitimate because
# this string is a constant in this file — never pass model-authored text here.
PROBE_SCRIPT = (
    "echo arch=$(uname -m); echo cores=$(nproc); "
    "echo mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo); "
    "echo mem_avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo); "
    "echo disk_avail_bytes=$(df -B1 --output=avail / | tail -1); "
    "echo distro=$(grep -m1 PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"'); "
    "echo docker=$(docker --version 2>/dev/null | cut -d, -f1); "
    "echo python=$(python3 -V 2>&1); "
    "echo xvfb=$(command -v Xvfb); echo x11vnc=$(command -v x11vnc); "
    "echo websockify=$(command -v websockify); echo novnc=$(ls /usr/share/novnc 2>/dev/null | head -1); "
    "echo sudo=$(command -v sudo); echo user=$(id -un)"
)


class SSHError(RuntimeError):
    pass


def cleanup_askpass(path: str | None) -> None:
    """Delete a per-invocation askpass script created by SSHHub._make_askpass."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass  # already gone, or never ours — either way nothing to clean


class ApprovalRequired(SSHError):
    def __init__(self, host: str, command: str, verdict: Verdict):
        self.host, self.command, self.verdict = host, command, verdict
        super().__init__(
            f"approval required on '{host}' "
            f"({', '.join(verdict.reasons) or 'not provably read-only'}):\n  {command}\n"
            "re-run with approved=True only after a human confirms"
        )


@dataclass
class HostSpec:
    label: str
    host: str
    port: int = 22
    user: str = ""                  # empty = whatever ~/.ssh/config says
    key_path: str = ""
    password_env: str = ""          # NAME of an env var holding the secret, never the secret
    jump: str = ""                  # label of a bastion to ProxyJump through
    use_login_shell: bool = False   # bash -lc imports profile PATH noise; off unless asked
    # Fleet fields. "standard" keeps the global gate semantics; "strict" forces
    # the two-phase approval flow for anything not provably read-only;
    # "permissive" only enforces block-level rules (audited, never silent).
    policy: str = "standard"
    tags: list[str] = field(default_factory=list)   # fleet grouping / tag:xxx selectors
    # Optional custom metrics collector. DANGER: it runs on the remote host with
    # that host's credentials, so it is never treated as trusted — it passes the
    # gate like any model-authored command, and anything not provably read-only
    # needs a one-shot human approval before it runs. Keep it to read-only
    # collectors (df, free, ...) and never paste secrets into it.
    metrics_cmd: str = ""

    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


@dataclass
class ExecResult:
    host: str
    command: str
    rc: int
    stdout: str
    stderr: str
    level: str
    reasons: list[str] = field(default_factory=list)
    duration_ms: int = 0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def as_dict(self) -> dict:
        return {
            "host": self.host, "command": self.command, "rc": self.rc, "ok": self.ok,
            "stdout": self.stdout, "stderr": self.stderr, "level": self.level,
            "reasons": self.reasons, "duration_ms": self.duration_ms, "truncated": self.truncated,
        }


class SSHHub:
    def __init__(self, audit: AuditLog | None = None, approver=None,
                 allow_confirm: bool = False, connect_timeout: int = 12):
        self.audit = audit or AuditLog(None)
        self.approver = approver
        self.allow_confirm = allow_confirm
        self.connect_timeout = connect_timeout
        self._hosts: dict[str, HostSpec] = {}

    # -- configuration ----------------------------------------------------
    def add_host(self, spec: HostSpec) -> None:
        self._hosts[spec.label] = spec

    def hosts(self) -> list[str]:
        return sorted(self._hosts)

    def get_host(self, label: str) -> HostSpec:
        spec = self._hosts.get(label)
        if spec is None:
            raise SSHError(
                f"unknown host '{label}'. known: {', '.join(self.hosts()) or 'none'} "
                "(declare it in agentd.json under ssh_hosts)"
            )
        return spec

    def remove_host(self, label: str) -> None:
        """Drop a host from the live registry (config stays the source of truth).

        Removal is hot but not violent: sessions that already reference the
        label simply fail with a clean `unknown host` on their next exec —
        nothing is force-killed and no in-flight command is interrupted.
        """
        self._hosts.pop(label, None)

    # -- argv ------------------------------------------------------------
    def _ssh_base(self, label: str, for_copy: bool = False) -> list[str]:
        spec = self.get_host(label)
        binary = "scp" if for_copy else "ssh"
        argv = [binary, "-o", f"ConnectTimeout={self.connect_timeout}"]
        if not (os.environ.get(spec.password_env) if spec.password_env else None):
            argv += ["-o", "BatchMode=yes"]
        if spec.key_path:
            argv += ["-i", os.path.expanduser(spec.key_path), "-o", "IdentitiesOnly=yes"]
        if spec.port:
            argv += ["-P", str(spec.port)] if for_copy else ["-p", str(spec.port)]
        if spec.jump and not for_copy:
            jump = self.get_host(spec.jump)
            argv += ["-J", jump.target() + (f":{jump.port}" if jump.port else "")]
        if for_copy:
            argv += ["-r"]
        return argv

    def _env(self, spec: HostSpec) -> dict:
        env = dict(os.environ)
        password = os.environ.get(spec.password_env) if spec.password_env else None
        if password:
            env["AGENTD_SSH_PASSWORD"] = password
            env["SSH_ASKPASS"] = self._make_askpass()
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env.pop("DISPLAY", None)
        return env

    @staticmethod
    def _make_askpass() -> str:
        """Create a private per-invocation askpass script and return its path.

        The old fixed path /tmp/agentd-askpass.sh was a race between concurrent
        sessions and a TOCTOU window: another local user could observe or
        replace the predictable file between write and exec. mkstemp gives an
        atomically-created 0600 file (chmod'ed to 0700 before use); the caller
        removes it via cleanup_askpass() once its ssh child is done.
        """
        body = '#!/bin/sh\nprintf "%s\\n" "$AGENTD_SSH_PASSWORD"\n'
        fd, path = tempfile.mkstemp(prefix="agentd-askpass-", suffix=".sh")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.chmod(path, 0o700)
        except BaseException:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return path

    # -- execution --------------------------------------------------------
    def exec(self, label: str, command: str, timeout: int = DEFAULT_TIMEOUT, approved: bool = False,
             stdin: str | None = None, session: str | None = None,
             episode_id: int | None = None, action: str | None = None) -> ExecResult:
        spec = self.get_host(label)
        verdict = classify(command)
        # Per-host enforcement tier (fleet F4). `policy` decides how the global
        # gate verdict is enforced for this host; the verdict itself is shared.
        policy = (getattr(spec, "policy", "") or "standard").strip().lower()
        if policy not in POLICIES:
            policy = "standard"     # unknown value: fall back to the safe default
        permissive_auto = False

        def _deny(error: str) -> ApprovalRequired:
            """Audit the refusal, then raise — a denial must leave a record."""
            self.audit.write(AuditRecord(host=label, command=command, level=verdict.level,
                                         reasons=verdict.reasons, session=session,
                                         episode_id=episode_id, error=error,
                                         action=action or ""))
            return ApprovalRequired(label, command, verdict)

        if verdict.hard_blocked and not approved:
            raise _deny("blocked by gate")

        if policy == "strict":
            # Strict hosts force the explicit two-phase approval flow for every
            # command that is not provably read-only, whatever the gate decided:
            # the inline approver shortcut is bypassed, the request is audited,
            # and only a human re-run with approved=True executes it.
            if not approved and not initial_readonly(command):
                raise _deny("host policy 'strict' requires explicit approval")
        elif policy == "permissive":
            # Permissive hosts only enforce block-level rules; confirm-level and
            # unprovable commands run without a human, but the audit record gets
            # a marker so the relaxation is never silent.
            if not approved and (verdict.level == LEVEL_CONFIRM
                                 or not initial_readonly(command)):
                permissive_auto = True
        else:  # standard: the historical gate behaviour, byte for byte
            if verdict.level == LEVEL_CONFIRM and not approved:
                if not self._granted(label, command, verdict, policy):
                    raise _deny("awaiting approval")
            if not approved and verdict.level == LEVEL_ALLOW and not initial_readonly(command):
                if not self._granted(label, command, verdict, policy):
                    raise _deny("awaiting approval")

        shell = "bash -lc" if spec.use_login_shell else "sh -c"
        argv = self._ssh_base(label) + [spec.target(), f"{shell} {shlex.quote(command)}"]
        t0 = time.time()
        env = self._env(spec)
        try:
            try:
                proc = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                                      timeout=timeout, env=env)
                rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
            except subprocess.TimeoutExpired:
                rc, out, err = 124, "", f"timed out after {timeout}s"
            except FileNotFoundError:
                raise SSHError("no 'ssh' binary on PATH; install OpenSSH client") from None
        finally:
            # The askpass script is per-invocation; remove it whether the
            # command succeeded, timed out or raised.
            cleanup_askpass(env.get("SSH_ASKPASS"))
        duration = int((time.time() - t0) * 1000)

        truncated = len(out) > MAX_CAPTURE
        if truncated:
            out = out[:MAX_CAPTURE] + f"\n... [{len(out) - MAX_CAPTURE} bytes elided] ..."
        result = ExecResult(host=label, command=command, rc=rc, stdout=out, stderr=err.strip(),
                            level=verdict.level, reasons=verdict.reasons,
                            duration_ms=duration, truncated=truncated)
        self.audit.write(AuditRecord(host=label, command=command, level=verdict.level,
                                     reasons=verdict.reasons,
                                     approved_by="human" if approved else None, rc=rc,
                                     stdout_len=len(out), error=err.strip() or None,
                                     session=session, episode_id=episode_id,
                                     action=action or ("policy=permissive auto-approved"
                                                       if permissive_auto else None)))
        return result

    def _granted(self, label: str, command: str, verdict: Verdict,
                 policy: str = "standard") -> bool:
        if self.approver is None:
            return False
        return bool(self.approver({"host": label, "command": command, "policy": policy,
                                   **verdict.as_dict()}))

    def probe(self, label: str, timeout: int = 25) -> dict:
        """Static inventory used to decide what this host can host (P0-1 gate)."""
        result = self.exec(label, PROBE_SCRIPT, timeout=timeout, approved=True)
        facts = _parse_kv(result.stdout)
        facts.update({
            "host": label, "rc": result.rc, "stderr": result.stderr,
            "ram_mb": round(int(facts.get("mem_total_kb", 0) or 0) / 1024),
            "ram_avail_mb": round(int(facts.get("mem_avail_kb", 0) or 0) / 1024),
            "disk_avail_gb": round(int(facts.get("disk_avail_bytes", 0) or 0) / 1024**3, 1),
        })
        facts["webarena_viable"] = (
            facts.get("arch") == "x86_64"
            and facts["ram_mb"] >= 16_000 and facts["disk_avail_gb"] >= 80
            and facts.get("docker") not in (None, "", "none")
        )
        facts["viewer_viable"] = all(facts.get(k) not in (None, "", "none")
                                     for k in ("xvfb", "x11vnc"))
        return facts

    def probe_many(self, labels: list[str], timeout: int = 25) -> list[dict]:
        return [self.probe(label, timeout) for label in labels]

    # -- files ------------------------------------------------------------
    def read_file(self, label: str, path: str, max_bytes: int = 200_000) -> str:
        return self.exec(label, f"head -c {int(max_bytes)} -- {shlex.quote(path)}").stdout

    def ls(self, label: str, path: str = ".") -> list[str]:
        out = self.exec(label, f"ls -la -- {shlex.quote(path)}").stdout
        return [ln for ln in out.splitlines() if ln.strip()]

    def tail_log(self, label: str, path: str, lines: int = 200, pattern: str = "") -> str:
        cmd = f"tail -n {int(lines)} -- {shlex.quote(path)}"
        if pattern:
            cmd += f" | grep -i -E {shlex.quote(pattern)} | tail -n {int(lines)}"
        return self.exec(label, cmd).stdout

    def port_check(self, label: str, port: int) -> bool:
        cmd = ("ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null") + f" | grep -c ':{int(port)} '"
        out = self.exec(label, cmd).stdout.strip()
        return out.isdigit() and int(out) > 0

    def transfer_resumable(self, label: str, local: str, remote: str, chunk: int = CHUNK) -> dict:
        """Chunked upload, per-part size check, then md5 reconciliation of the result.

        A truncated transfer that looks finished is worse than one that fails
        loudly, so nothing is renamed until the assembled md5 matches.
        """
        src = os.path.expanduser(local)
        if not os.path.exists(src):
            return {"ok": False, "error": f"no such local file: {src}"}
        spec = self.get_host(label)
        size = os.path.getsize(src)
        digest = hashlib.md5()
        with open(src, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        local_md5 = digest.hexdigest()

        stage = f"{remote}.parts"
        assembled = f"{remote}.assembled"
        report = {"ok": False, "bytes": size, "md5": local_md5, "resent": [], "skipped": []}
        self.exec(label, f"mkdir -p -- {shlex.quote(stage)} {shlex.quote(os.path.dirname(remote) or '.')}")

        plan = []
        offset, part_no = 0, 0
        while offset < size:
            length = min(chunk, size - offset)
            plan.append((offset, length, f"part_{part_no:04d}"))
            offset, part_no = offset + length, part_no + 1

        for offset, length, name in plan:
            remote_part = f"{stage}/{name}"
            if self._remote_size(label, remote_part) == length:
                report["skipped"].append(name)
                continue
            tmp = os.path.join(tempfile.gettempdir(), f"agentd-{os.getpid()}-{name}")
            with open(src, "rb") as fh, open(tmp, "wb") as out:
                fh.seek(offset)
                remaining = length
                while remaining > 0:
                    block = fh.read(min(1 << 20, remaining))
                    if not block:
                        break
                    out.write(block)
                    remaining -= len(block)
            if os.path.getsize(tmp) != length:
                os.remove(tmp)
                report["error"] = f"local split short: {name}"
                return report
            argv = self._ssh_base(label, for_copy=True) + [tmp, f"{spec.target()}:{remote_part}"]
            part_env = self._env(spec)
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800,
                                      env=part_env)
            finally:
                cleanup_askpass(part_env.get("SSH_ASKPASS"))
                os.remove(tmp)
            if proc.returncode != 0 or self._remote_size(label, remote_part) != length:
                report["error"] = f"upload failed for {name}: {proc.stderr.strip()[:200]}"
                return report
            report["resent"].append(name)

        listing = self.exec(label, f"ls -1 {shlex.quote(stage)}")
        names = sorted(n for n in listing.stdout.split() if n.startswith("part_"))
        if len(names) != len(plan):
            report["error"] = f"expected {len(plan)} parts, found {len(names)}"
            return report

        cat = " ".join(shlex.quote(f"{stage}/{n}") for n in names)
        assemble = self.exec(label, f"cat {cat} > {shlex.quote(assembled)}")
        if assemble.rc != 0:
            report["error"] = f"assemble failed: {assemble.stderr[:200]}"
            return report
        check = self.exec(label, f"md5sum -- {shlex.quote(assembled)}")
        remote_md5 = check.stdout.split()[0] if check.stdout.split() else None
        if remote_md5 != local_md5:
            report["error"] = f"md5 mismatch: local={local_md5} remote={remote_md5}"
            return report
        move = self.exec(label, f"mv -f -- {shlex.quote(assembled)} {shlex.quote(remote)}", approved=True)
        if move.rc != 0:
            report["error"] = f"finalize failed: {move.stderr[:200]}"
            return report
        self.exec(label, f"rm -rf -- {shlex.quote(stage)}", approved=True)
        report.update({"ok": True, "remote": remote})
        return report

    def _remote_size(self, label: str, path: str) -> int | None:
        out = self.exec(label, f"stat -c %s -- {shlex.quote(path)} 2>/dev/null || echo -").stdout.strip()
        return int(out) if out.isdigit() else None

    def close(self) -> None:
        return None


def _parse_kv(text: str) -> dict:
    facts: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([a-z_]+)=(.*)", line.strip())
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        facts[key] = "" if value in ("none", "") else value
    return facts
