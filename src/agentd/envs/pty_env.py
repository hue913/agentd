"""Interactive PTY sessions over the existing SSH hub.

Why this exists
---------------
`ssh_env.exec` is deliberately batch-only: it runs one command, captures stdout,
returns. That is the right shape for an LLM tool, and the wrong shape for a
human. This module is the human half -- a real login shell with a real TTY,
served to xterm.js over a WebSocket so the desktop client and the web console
can share one implementation.

The safety story, stated plainly
--------------------------------
A PTY is a byte stream. By the time "rm -rf /" has been typed, the gate has
nothing left to pattern-match: the characters arrive one at a time and the
final command is assembled by the remote shell. Pretending to gate keystrokes
would be theatre. So this module does the two things that are actually real:

1. Session-level admission. The caller states a purpose; the gate classifies
   the *intent* before a shell exists. A blocked purpose never gets a PTY.
2. Full session audit. Open, resize, close, and a rolling digest of what was
   typed -- not the raw bytes, which would put secrets in an append-only log.

Anything inside the session is the operator's own shell and their own risk,
which is exactly how a terminal has always worked. This does not weaken
`ssh_env`; an agent calling `ssh.exec` still goes through the per-command gate.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import pty
import shlex
import signal
import struct
import subprocess
import termios
import threading
import time
from dataclasses import dataclass, field

from ..log import get_logger
from ..safety.audit import AuditRecord
from ..safety.gate import LEVEL_BLOCK
from .ssh_env import SSHHub, cleanup_askpass, classify

# Cap on retained unread output. A forgotten tab running `tail -f` would
# otherwise grow without bound; the oldest bytes are dropped instead of
# blocking the reader, which is what a real terminal does on a slow consumer.
DEFAULT_BUFFER_BYTES = 256 * 1024
# A session nobody is attached to gets reaped. Without this, a client that dies
# mid-session leaves a live root shell on the box.
DEFAULT_IDLE_TTL_S = int(os.environ.get("AGENTD_PTY_IDLE_TTL", "1800") or "1800")
READ_CHUNK = 65536
# Background reaper cadence. GET /api/pty also reaps opportunistically, but
# nobody opening the panel must not mean dead shells live forever.
REAP_INTERVAL_S = int(os.environ.get("AGENTD_PTY_REAP_INTERVAL", "60") or "60")

log = get_logger("agentd.pty")


class PTYError(RuntimeError):
    pass


class PTYRefused(PTYError):
    """The stated purpose was blocked before any shell was created."""


@dataclass
class _Ring:
    """Byte buffer that drops the oldest data instead of growing."""

    cap: int = DEFAULT_BUFFER_BYTES
    _buf: bytearray = field(default_factory=bytearray)
    dropped: int = 0

    def put(self, data: bytes) -> None:
        self._buf.extend(data)
        if len(self._buf) > self.cap:
            excess = len(self._buf) - self.cap
            del self._buf[:excess]
            self.dropped += excess

    def drain(self) -> bytes:
        out = bytes(self._buf)
        self._buf.clear()
        return out

    def __len__(self) -> int:
        return len(self._buf)


class PTYSession:
    """One interactive shell on one host. Not thread-safe beyond its own lock."""

    _counter = 0

    def __init__(self, hub: SSHHub, label: str, rows: int = 24, cols: int = 80,
                 purpose: str = "", idle_ttl: int = DEFAULT_IDLE_TTL_S):
        PTYSession._counter += 1
        self.id = f"pty-{os.getpid()}-{PTYSession._counter}"
        self.hub = hub
        self.label = label
        self.purpose = purpose
        self.rows = max(4, min(rows, 500))
        self.cols = max(20, min(cols, 1000))
        self.idle_ttl = idle_ttl
        self.created_at = time.time()
        self.last_activity = self.created_at
        self.closed = False
        self.exit_reason = ""
        self._ring = _Ring()
        self._lock = threading.Lock()
        self._hash = hashlib.sha256()
        self._input_bytes = 0
        self._output_bytes = 0

        spec = self.hub.get_host(label)
        # -tt forces a TTY even though stdin is not our terminal. Without it ssh
        # silently produces a piped session and every curses program misbehaves.
        argv = self.hub._ssh_base(label) + ["-tt", spec.target()]
        self.master_fd, slave_fd = pty.openpty()
        self._set_winsize(slave_fd, self.rows, self.cols)
        env = self.hub._env(spec)
        # If ssh needs a password, the askpass script was created for this
        # session; it must die with the session, not linger in /tmp.
        self._askpass_path = env.get("SSH_ASKPASS")
        # A predictable prompt and TERM matter: without TERM the remote side
        # falls back to "dumb" and xterm.js renders nothing useful.
        env.setdefault("TERM", "xterm-256color")
        try:
            self.proc = subprocess.Popen(
                argv, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                env=env, close_fds=True, start_new_session=True,
            )
        except FileNotFoundError as exc:
            os.close(self.master_fd)
            os.close(slave_fd)
            raise PTYError("no 'ssh' binary on PATH; install the OpenSSH client") from exc
        finally:
            os.close(slave_fd)

        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    # -- io ---------------------------------------------------------------
    def _pump(self) -> None:
        while not self.closed:
            try:
                chunk = os.read(self.master_fd, READ_CHUNK)
            except OSError as exc:
                if exc.errno not in (errno.EIO, errno.EBADF):
                    # Honesty for the operator: EIO/EBADF is the ordinary
                    # "slave closed, shell exited" path, but any other OSError
                    # also ends the stream (there is no way to keep pumping
                    # from a dead fd) and must not pass as a normal exit.
                    log.warning("pty %s: reader stopped on unexpected OSError: %s", self.id, exc)
                break  # slave closed or fd unusable: the shell is gone either way
            if not chunk:
                break
            with self._lock:
                self._ring.put(chunk)
                self._output_bytes += len(chunk)
        self.closed = True
        # Only fill in a reason if nobody else already stated one. The closer
        # sets exit_reason BEFORE signalling the child, so an unconditional
        # write here would overwrite "idle 3600s" with "remote exited" and the
        # audit log would misreport why a session was reaped.
        if not self.exit_reason:
            self.exit_reason = f"remote exited rc={self.proc.poll()}"

    def read(self) -> bytes:
        with self._lock:
            self.last_activity = time.time()
            return self._ring.drain()

    def write(self, data: bytes) -> int:
        if self.closed:
            raise PTYError("session already closed")
        if not isinstance(data, (bytes, bytearray)):
            raise PTYError("write() takes bytes; decode on the client side")
        with self._lock:
            self._hash.update(data)
            self._input_bytes += len(data)
            self.last_activity = time.time()
        return os.write(self.master_fd, bytes(data))

    def resize(self, rows: int, cols: int) -> None:
        self.rows = max(4, min(int(rows), 500))
        self.cols = max(20, min(int(cols), 1000))
        self.last_activity = time.time()
        self._set_winsize(self.master_fd, self.rows, self.cols)
        # The ioctl alone updates the kernel's idea of the size, but ssh only
        # re-reads it -- and forwards it to the far end -- when it is signalled.
        # Without this the remote `stty size` keeps reporting the old geometry
        # even though the pty is the right size locally.
        self._notify_child()

    def _notify_child(self) -> None:
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGWINCH)
        except (ProcessLookupError, PermissionError, OSError):
            pass  # the child may already be gone; the resize is best-effort

    @staticmethod
    def _set_winsize(fd: int, rows: int, cols: int) -> None:
        try:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except OSError:
            pass  # a closed or non-tty fd is not fatal; size simply will not change

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if self.closed and self.proc.poll() is not None:
            return
        self.closed = True
        try:
            self.proc.send_signal(signal.SIGHUP)
        except (ProcessLookupError, OSError):
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                self.proc.kill()
        try:
            os.close(self.master_fd)
        except OSError:
            pass
        # The askpass script was created for this session's ssh; nothing else
        # uses it once the child is gone.
        cleanup_askpass(getattr(self, "_askpass_path", None))

    @property
    def alive(self) -> bool:
        return not self.closed and self.proc.poll() is None

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    def digest(self) -> dict:
        """Audit view of the session. Never contains raw typed bytes."""
        return {
            "session": self.id,
            "host": self.label,
            "purpose": self.purpose,
            "opened_at": self.created_at,
            "idle_s": round(self.idle_seconds, 1),
            "input_bytes": self._input_bytes,
            "output_bytes": self._output_bytes,
            "dropped_bytes": self._ring.dropped,
            "input_sha256": self._hash.hexdigest()[:16],
            "exit_reason": self.exit_reason,
        }


class PTYHub:
    """Owns live sessions and reaps the abandoned ones."""

    def __init__(self, hub: SSHHub, audit=None, idle_ttl: int = DEFAULT_IDLE_TTL_S):
        self.hub = hub
        self.audit = audit
        self.idle_ttl = idle_ttl
        self.sessions: dict[str, PTYSession] = {}
        self._reaper: threading.Thread | None = None
        self._reaper_stop = threading.Event()

    def open(self, label: str, purpose: str = "", rows: int = 24, cols: int = 80,
             approved: bool = False) -> PTYSession:
        # The gate sees the stated intent, not keystrokes. This is the only
        # place a PTY can be meaningfully policed, so it is checked up front.
        verdict = classify(purpose) if purpose else None
        if verdict is not None and verdict.level == LEVEL_BLOCK and not approved:
            self._audit(AuditRecord(host=label, command=f"[pty] {purpose}", level="block",
                                    reasons=verdict.reasons, error="blocked at session admission"))
            raise PTYRefused("; ".join(verdict.reasons) or "purpose blocked by safety gate")
        session = PTYSession(self.hub, label, rows=rows, cols=cols, purpose=purpose,
                             idle_ttl=self.idle_ttl)
        self.sessions[session.id] = session
        self._audit(AuditRecord(host=label, command=f"[pty-open] {purpose}", level="allow",
                                reasons=verdict.reasons if verdict else [],
                                session=session.id))
        return session

    def get(self, session_id: str) -> PTYSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise PTYError("no such pty session")
        return session

    def close(self, session_id: str, reason: str = "client closed") -> dict:
        session = self.get(session_id)
        session.exit_reason = reason
        session.close()
        self.sessions.pop(session_id, None)
        self._audit(AuditRecord(host=session.label, command="[pty-close]", level="allow",
                                session=session_id, error=reason))
        return session.digest()

    def reap(self, force: bool = False) -> list[dict]:
        """Close sessions that are dead, or that nobody has touched in TTL."""
        reaped = []
        for sid, session in list(self.sessions.items()):
            stale = not session.alive or (not force and session.idle_seconds > self.idle_ttl)
            if not stale:
                continue
            reason = "remote exited" if not session.alive else f"idle {int(session.idle_seconds)}s"
            session.exit_reason = reason
            session.close()
            self.sessions.pop(sid, None)
            reaped.append(session.digest())
            self._audit(AuditRecord(host=session.label, command="[pty-reap]", level="allow",
                                    session=sid, error=reason))
        return reaped

    # -- background reaper --------------------------------------------------
    def start_reaper(self, interval: int = REAP_INTERVAL_S) -> None:
        """Reap every `interval` seconds from a daemon thread.

        GET /api/pty reaps opportunistically, but only when somebody opens
        the panel — an abandoned shell with the panel closed never got a
        second look. The thread is a daemon, so process exit never blocks on
        it; stop_reaper() shuts it down cleanly at Runtime.close().
        """
        if self._reaper and self._reaper.is_alive():
            return
        self._reaper_stop.clear()
        interval = max(1, int(interval))

        def _loop() -> None:
            # Event.wait doubles as an interruptible sleep: stop_reaper()
            # returns promptly instead of after a full interval.
            while not self._reaper_stop.wait(interval):
                try:
                    self.reap()
                except Exception as exc:
                    # One failed pass (a session dying mid-close, an audit
                    # hiccup) must not kill the reaper thread.
                    log.warning("pty reaper pass failed: %s", exc)

        self._reaper = threading.Thread(target=_loop, name="agentd-pty-reaper", daemon=True)
        self._reaper.start()

    def stop_reaper(self) -> None:
        self._reaper_stop.set()
        if self._reaper is not None:
            self._reaper.join(timeout=5)
            self._reaper = None

    def listing(self) -> list[dict]:
        return [s.digest() for s in self.sessions.values()]

    def _audit(self, record: AuditRecord) -> None:
        if self.audit is not None:
            try:
                self.audit.write(record)
            except Exception as exc:
                # An audit write must never take down an interactive shell —
                # but it must also not vanish: log it for the operator.
                log.warning("pty audit write failed: %s", exc)
