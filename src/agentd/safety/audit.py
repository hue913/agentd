"""Append-only audit trail. Every remote command leaves a record, blocked or not."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path else None
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
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        else:
            print(line, flush=True)

    def tail(self, limit: int = 50, host: str | None = None) -> list[dict]:
        if not self.path or not self.path.exists():
            return []
        rows: list[dict] = []
        with self.path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if host and row.get("host") != host:
                    continue
                rows.append(row)
        return rows[-limit:]

    def blocked_count(self) -> int:
        return sum(1 for row in self.tail(limit=10_000) if row.get("level") in ("block", "confirm"))
