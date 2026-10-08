"""Append-only audit trail. Every remote command leaves a record, blocked or not."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..log import get_logger

# One archive generation (audit.jsonl -> audit.jsonl.1) keeps the tail query
# bounded without growing a second append-only surface to secure.
ROTATE_BYTES = 5 * 1024 * 1024
# Chunk size for the backwards scan in tail(); small enough to keep memory flat.
_TAIL_CHUNK = 8192

log = get_logger("agentd.audit")


@dataclass
class AuditRecord:
    ts: float = field(default_factory=time.time)
    host: str = ""
    action: str = ""
    command: str = ""
    level: str = "allow"
    reasons: list[str] = field(default_factory=list)
    approved_by: str | None = None
    rc: int | None = None
    stdout_sha: str | None = None
    stdout_len: int = 0
    error: str | None = None
    session: str | None = None
    episode_id: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class AuditLog:
    def __init__(self, path: str | Path | None = None, max_bytes: int = ROTATE_BYTES):
        self.path = Path(path).expanduser() if path else None
        self.max_bytes = max(1, int(max_bytes))
        # Appends come from HTTP worker threads, the scheduler and the PTY
        # reaper; a raw open("a") from two threads can interleave inside one
        # line and corrupt the JSONL. The lock makes one write = one record.
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    def write(self, record: AuditRecord) -> None:
        line = json.dumps(record.as_dict(), ensure_ascii=False)
        if self.path:
            with self._lock:
                self._rotate_if_over_limit()
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        else:
            # No file configured (tests, CLI): the record still has to land
            # somewhere an operator can see it — the unified stderr channel.
            log.info("%s", line)

    def _rotate_if_over_limit(self) -> None:
        """Rename to <path>.1 (one generation), keeping the live file empty.

        Runs under the write lock, so no append can straddle the rename.
        A failed rotation must not block auditing: the log keeps growing
        rather than dropping records.
        """
        try:
            if self.path.stat().st_size < self.max_bytes:
                return
            backup = self.path.with_suffix(self.path.suffix + ".1")
            os.replace(self.path, backup)
            self.path.touch(exist_ok=True)
            try:
                os.chmod(backup, 0o600)
            except OSError:
                pass
        except OSError as exc:
            log.warning("audit rotation failed for %s: %s", self.path, exc)

    def tail(self, limit: int = 50, host: str | None = None) -> list[dict]:
        if not self.path or not self.path.exists():
            return []
        with self._lock:
            return self._tail_locked(limit, host)

    def _tail_locked(self, limit: int, host: str | None) -> list[dict]:
        """Last `limit` records in chronological order, read from the end.

        The old implementation read the whole file; on a rotated-bounded log
        that is up to 5MB of parsing per /api/audit request. Scanning backwards
        stops as soon as `limit` matching rows are found.
        """
        matched: list[dict] = []
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        with self.path.open("rb") as fh:
            buffer = b""
            pos = size
            while pos > 0 and len(matched) < limit:
                step = min(_TAIL_CHUNK, pos)
                pos -= step
                fh.seek(pos)
                buffer = fh.read(step) + buffer
                lines = buffer.split(b"\n")
                if pos > 0:
                    # The first element is a possibly-partial line that the
                    # next (leftward) chunk completes; keep it in the buffer.
                    buffer = lines.pop(0)
                else:
                    # The whole file has been read: every element is complete.
                    buffer = b""
                for raw in reversed(lines):
                    row = self._parse(raw)
                    if row is None:
                        continue
                    if host and row.get("host") != host:
                        continue
                    matched.append(row)
                    if len(matched) >= limit:
                        break
        matched.reverse()  # newest-first scan -> chronological output, as before
        return matched

    @staticmethod
    def _parse(raw: bytes) -> dict | None:
        raw = raw.strip()
        if not raw:
            return None
        try:
            row = json.loads(raw)
        except ValueError:
            return None  # a torn line (crash mid-write) is skipped, not fatal
        return row if isinstance(row, dict) else None

    def blocked_count(self) -> int:
        return sum(1 for row in self.tail(limit=10_000) if row.get("level") in ("block", "confirm"))
