"""Recurring jobs: a dependency-free cron evaluator plus a scheduler loop.

No croniter, no system crontab: the jobs live in agentd's own config so they are
portable to Windows, and every run is recorded next to the memory it produces.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
    "@minutely": "* * * * *",
}

FIELDS = (("minute", 0, 59), ("hour", 0, 23), ("day", 1, 31), ("month", 1, 12), ("weekday", 0, 6))


class CronError(ValueError):
    pass


def _int(text: str, spec: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise CronError(f"not a number: {text!r} in {spec!r}") from None


def _parse_field(spec: str, low: int, high: int) -> set[int]:
    spec = spec.strip()
    if spec == "*":
        return set(range(low, high + 1))
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            part, _, step_text = part.partition("/")
            if not re.fullmatch(r"\d+", step_text):
                raise CronError(f"bad step {step_text!r} in {spec!r}")
            step = int(step_text)
            if step < 1:
                raise CronError(f"step must be >= 1 in {spec!r}")
        if part in ("", "*"):
            start, end = low, high
        elif "-" in part:
            left, _, right = part.partition("-")
            start, end = _int(left, spec), _int(right, spec)
        else:
            start = end = _int(part, spec)
        if start < low or end > high or start > end:
            raise CronError(f"value out of range {low}-{high} in {spec!r}")
        values.update(range(start, end + 1, step))
    return values


@dataclass
class Cron:
    expression: str

    def __post_init__(self):
        expr = ALIASES.get(self.expression.strip(), self.expression.strip())
        parts = expr.split()
        if len(parts) != 5:
            raise CronError(f"cron needs 5 fields, got {self.expression!r}")
        self.minutes, self.hours, self.days, self.months, self.weekdays = (
            _parse_field(part, low, high) for part, (_, low, high) in zip(parts, FIELDS)
        )
        self.normalized = expr

    def matches(self, moment: datetime) -> bool:
        dom_ok = moment.day in self.days
        dow_ok = moment.isoweekday() % 7 in self.weekdays
        # cron semantics: when both day-of-month and day-of-week are restricted,
        # either one satisfying is enough
        if self.days != set(range(1, 32)) and self.weekdays != set(range(0, 7)):
            day_ok = dom_ok or dow_ok
        else:
            day_ok = dom_ok and dow_ok
        return (moment.minute in self.minutes and moment.hour in self.hours
                and moment.month in self.months and day_ok)

    def next_after(self, moment: datetime, horizon_minutes: int = 60 * 24 * 366) -> datetime | None:
        cursor = (moment + timedelta(minutes=1)).replace(second=0, microsecond=0)
        for _ in range(horizon_minutes):
            if self.matches(cursor):
                return cursor
            cursor += timedelta(minutes=1)
        return None


@dataclass
class Job:
    name: str
    cron: str
    task: str
    provider: str = ""
    learning: bool = True
    max_steps: int = 12
    council: bool = False
    enabled: bool = True
    last_run: float | None = None
    last_status: str = ""
    created_at: float = field(default_factory=time.time)

    def schedule(self) -> Cron:
        return Cron(self.cron)

    def as_dict(self) -> dict:
        return asdict(self)


class ScheduleStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "schedule.json"
        self.history = self.dir / "schedule_history.jsonl"

    def load(self) -> list[Job]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [Job(**row) for row in raw.get("jobs", [])]

    def save(self, jobs: list[Job]) -> None:
        payload = {"jobs": [job.as_dict() for job in jobs], "updated": time.time()}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.path.chmod(0o600)

    def upsert(self, job: Job) -> list[Job]:
        jobs = {j.name: j for j in self.load()}
        jobs[job.name] = job
        out = sorted(jobs.values(), key=lambda j: j.name)
        self.save(out)
        return out

    def remove(self, name: str) -> bool:
        jobs = self.load()
        kept = [j for j in jobs if j.name != name]
        if len(kept) == len(jobs):
            return False
        self.save(kept)
        return True

    def record(self, result: dict) -> None:
        with self.history.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


class Scheduler:
    """Runs due jobs on a fixed tick, never two of the same job at once."""

    def __init__(self, runtime, store: ScheduleStore, tick_seconds: int = 30):
        self.runtime = runtime
        self.store = store
        self.tick = tick_seconds
        self._running: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def last_scheduled_minute(self, cron: Cron, now: datetime,
                             lookback_minutes: int = 60) -> datetime | None:
        """Most recent scheduled minute at or before `now`, within the lookback.

        The lookback is what makes a stopped scheduler catch up exactly once
        instead of firing every missed minute the moment it comes back.
        """
        cursor = now.replace(second=0, microsecond=0)
        for _ in range(max(1, lookback_minutes)):
            if cron.matches(cursor):
                return cursor
            cursor -= timedelta(minutes=1)
        return None

    def due(self, jobs: list[Job], now: datetime, lookback_minutes: int = 60) -> list[Job]:
        fired: list[Job] = []
        for job in jobs:
            if not job.enabled:
                continue
            try:
                cron = job.schedule()
            except CronError:
                continue
            target = self.last_scheduled_minute(cron, now, lookback_minutes)
            if target is None:
                continue
            last = datetime.fromtimestamp(job.last_run).replace(second=0, microsecond=0) if job.last_run else None
            if last is None or last < target:
                fired.append(job)
        return fired

    def run_job(self, job: Job) -> dict:
        with self._lock:
            if job.name in self._running:
                return {"job": job.name, "skipped": "already running", "ts": time.time()}
            self._running.add(job.name)
        started = time.time()
        result = {"job": job.name, "task": job.task, "started_at": started, "ts": started}
        try:
            session = self.runtime.create_session(job.task, provider=job.provider,
                                                  learning=job.learning, max_steps=job.max_steps,
                                                  council=job.council)
            report = self.runtime.run_session(session)
            result.update({"status": session.status, "success": report.get("success"),
                           "steps": len(report.get("steps", [])), "session": session.id})
        except Exception as exc:
            result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            with self._lock:
                self._running.discard(job.name)
        result["duration_s"] = round(time.time() - started, 2)
        self.store.record(result)
        self.runtime.publish({"type": "schedule_run", **result})
        return result

    def tick_once(self, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now()
        jobs = self.store.load()
        pending = self.due(jobs, now)
        if not pending:
            return []
        names = {job.name for job in pending}
        outputs = []
        for job in pending:
            output = self.run_job(job)
            job.last_run = output["started_at"]
            job.last_status = output.get("status", "?")
            outputs.append(output)
        # persist the stamps for every job, keeping untouched ones as they were
        self.store.save(jobs)
        return outputs

    def daemon(self, stop_event: threading.Event | None = None) -> None:
        event = stop_event or self._stop
        while not event.is_set():
            try:
                self.tick_once()
            except Exception as exc:
                self.runtime.publish({"type": "schedule_error", "error": str(exc)})
            event.wait(self.tick)

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.daemon, daemon=True, name="agentd-scheduler")
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
