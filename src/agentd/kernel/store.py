"""SQLite-backed episodic memory: (state, action, discounted_return) triplets."""

from __future__ import annotations

from .credit import credit_factor, migrate

import functools
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT NOT NULL,
    goal        TEXT,
    started_at  REAL,
    finished_at REAL,
    success     INTEGER,
    score       REAL,
    analysis    TEXT,
    meta        TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER REFERENCES episodes(id) ON DELETE CASCADE,
    t          INTEGER NOT NULL,
    state      TEXT NOT NULL,
    state_fp   TEXT NOT NULL,
    action     TEXT NOT NULL,
    action_fp  TEXT NOT NULL,
    scope      TEXT,
    z          REAL,
    adv        REAL,
    z_prime    REAL,
    chosen     INTEGER,
    reward     REAL,
    ret        REAL
);
CREATE INDEX IF NOT EXISTS steps_action_fp_idx ON steps(action_fp);
CREATE INDEX IF NOT EXISTS steps_scope_idx     ON steps(scope);
CREATE INDEX IF NOT EXISTS steps_episode_idx   ON steps(episode_id);
CREATE INDEX IF NOT EXISTS steps_ret_idx       ON steps(ret);
CREATE TABLE IF NOT EXISTS model_stats (
    member   TEXT NOT NULL,
    scope    TEXT NOT NULL,
    trials   INTEGER DEFAULT 0,
    wins     INTEGER DEFAULT 0,
    ret_sum  REAL DEFAULT 0.0,
    PRIMARY KEY (member, scope)
);
CREATE TABLE IF NOT EXISTS risks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id  INTEGER REFERENCES episodes(id) ON DELETE CASCADE,
    ts          REAL,
    member      TEXT,
    description TEXT,
    severity    TEXT,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_task  ON episodes(task);
CREATE INDEX IF NOT EXISTS idx_risks_episode  ON risks(episode_id);
PRAGMA journal_mode=WAL;
"""

# One public Store method = one transaction. The class holds a single SQLite
# connection created with check_same_thread=False because FastAPI thread-pool
# workers, asyncio.to_thread workers and the scheduler all touch it; SQLite
# connections are not thread-safe, and without serialisation two threads can
# interleave statements inside one transaction (sqlite3 only catches misuse
# when check_same_thread is left on). The store is small enough that an RLock
# costs nothing and is far easier to reason about than per-thread connections.
def _transaction(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


@dataclass
class Step:
    id: int = 0
    episode_id: int = 0
    t: int = 0
    state: str = ""
    state_fp: str = ""
    action: str = ""
    action_fp: str = ""
    scope: str = ""
    z: float | None = None
    adv: float | None = None
    z_prime: float | None = None
    chosen: int | None = None
    reward: float = 0.0
    ret: float = 0.0
    # Credit accounting. all_steps() does Step(**dict(row)), so these three must
    # exist on the dataclass or every read of the table raises TypeError.
    recalls: int = 0
    adopted: int = 0
    rejected: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Episode:
    id: int = 0
    task: str = ""
    goal: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    success: bool | None = None
    score: float | None = None
    analysis: str = ""
    meta: dict = field(default_factory=dict)


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self._lock = threading.RLock()
        path = str(path)
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        # Enforce the episode_id foreign keys. This is a connection-level
        # pragma and a no-op inside a transaction, so it runs before the
        # schema script. Existing rows written before enforcement are never
        # re-validated by SQLite, so opening an old database stays safe.
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(_SCHEMA)
        migrate(self.db)
        self.db.commit()
        # Step rows are mutable after insertion (credit counters, outcomes).
        # Consumers that snapshot steps — the retrieval index — need to know
        # which ids changed so they can refresh only those rows.
        self._mutated_steps: list[int] = []

    @_transaction
    def close(self) -> None:
        self.db.close()

    # -- episodes ---------------------------------------------------------
    @_transaction
    def start_episode(self, task: str, goal: str = "", meta: dict | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO episodes(task, goal, started_at, meta) VALUES(?,?,?,?)",
            (task, goal, time.time(), json.dumps(meta or {}, ensure_ascii=False)),
        )
        self.db.commit()
        return int(cur.lastrowid)

    @_transaction
    def finish_episode(
        self,
        episode_id: int,
        success: bool | None,
        score: float | None = None,
        analysis: str = "",
    ) -> None:
        self.db.execute(
            "UPDATE episodes SET finished_at=?, success=?, score=?, analysis=? WHERE id=?",
            (time.time(), None if success is None else int(success), score, analysis, episode_id),
        )
        self.db.commit()

    @_transaction
    def get_episode(self, episode_id: int) -> Episode | None:
        row = self.db.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if row is None:
            return None
        return Episode(
            id=row["id"], task=row["task"], goal=row["goal"] or "",
            started_at=row["started_at"] or 0.0, finished_at=row["finished_at"],
            success=None if row["success"] is None else bool(row["success"]),
            score=row["score"], analysis=row["analysis"] or "",
            meta=json.loads(row["meta"] or "{}"),
        )

    @_transaction
    def recent_analyses(self, task: str | None = None, limit: int = 3) -> list[str]:
        if task:
            rows = self.db.execute(
                "SELECT analysis FROM episodes WHERE task=? AND analysis!='' "
                "ORDER BY id DESC LIMIT ?",
                (task, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT analysis FROM episodes WHERE analysis!='' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["analysis"] for r in rows]

    # -- steps ------------------------------------------------------------
    @_transaction
    def add_step(self, step: Step) -> int:
        cur = self.db.execute(
            "INSERT INTO steps(episode_id, t, state, state_fp, action, action_fp, scope,"
            " z, adv, z_prime, chosen, reward, ret) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                step.episode_id, step.t, step.state, step.state_fp, step.action,
                step.action_fp, step.scope, step.z, step.adv, step.z_prime, step.chosen,
                step.reward, step.ret,
            ),
        )
        self.db.commit()
        return int(cur.lastrowid)

    @_transaction
    def set_step_outcome(self, step_id: int, reward: float, ret: float) -> None:
        self.db.execute("UPDATE steps SET reward=?, ret=? WHERE id=?", (reward, ret), step_id)
        self.db.commit()

    @_transaction
    def all_steps(self) -> list[Step]:
        return [Step(**dict(r)) for r in self.db.execute("SELECT * FROM steps ORDER BY id")]

    @_transaction
    def steps_after(self, step_id: int) -> list[Step]:
        """Steps with id > step_id, ascending — the delta feed the retrieval
        index consumes so a growing memory never triggers a full rebuild."""
        return [
            Step(**dict(r))
            for r in self.db.execute("SELECT * FROM steps WHERE id > ? ORDER BY id", (step_id,))
        ]

    @_transaction
    def steps_for_episode(self, episode_id: int) -> list[Step]:
        return [
            Step(**dict(r))
            for r in self.db.execute("SELECT * FROM steps WHERE episode_id=? ORDER BY t", (episode_id,))
        ]

    @_transaction
    def count_steps(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM steps").fetchone()[0])

    @_transaction
    def max_step_id(self) -> int:
        return int(self.db.execute("SELECT COALESCE(MAX(id), 0) FROM steps").fetchone()[0])

    @_transaction
    def count_episodes(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])

    @_transaction
    def delete_episode(self, episode_id: int) -> None:
        self.db.execute("DELETE FROM steps WHERE episode_id=?", (episode_id,))
        self.db.execute("DELETE FROM episodes WHERE id=?", (episode_id,))
        self.db.commit()

    @_transaction
    def wipe(self) -> None:
        """Drop all learned experience (used by the --memory off control arm)."""
        self.db.execute("DELETE FROM steps")
        self.db.execute("DELETE FROM episodes")
        self.db.execute("DELETE FROM model_stats")
        self.db.execute("DELETE FROM risks")
        self.db.commit()

    # -- council: per-member reliability ----------------------------------
    @_transaction
    def record_member_outcome(self, member: str, scope: str, won: bool, ret: float = 0.0) -> None:
        self.db.execute(
            "INSERT INTO model_stats(member, scope, trials, wins, ret_sum) VALUES(?,?,1,?,?) "
            "ON CONFLICT(member, scope) DO UPDATE SET trials=trials+1, "
            "wins=wins+excluded.wins, ret_sum=ret_sum+excluded.ret_sum",
            (member, scope, int(won), float(ret)),
        )
        self.db.commit()

    @_transaction
    def member_reliability(self, scope: str = "") -> dict[str, float]:
        """Add-one smoothed win rate per member, globally pooled with scope preference.

        A member with no history still gets weight: without smoothing the first
        loss would silence it forever and the council could never recover.
        """
        rows = self.db.execute(
            "SELECT member, SUM(trials) AS trials, SUM(wins) AS wins FROM model_stats "
            "WHERE (? = '' OR scope = ? OR scope = '') GROUP BY member",
            (scope, scope),
        ).fetchall()
        out: dict[str, float] = {}
        for row in rows:
            trials, wins = row["trials"] or 0, row["wins"] or 0
            out[row["member"]] = (wins + 1) / (trials + 2)
        return out

    @_transaction
    def add_risk(self, episode_id: int | None, member: str, description: str,
                 severity: str = "medium", detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO risks(episode_id, ts, member, description, severity, detail) VALUES(?,?,?,?,?,?)",
            (episode_id, time.time(), member, description, severity, detail),
        )
        self.db.commit()

    @_transaction
    def risks_for(self, episode_id: int | None = None, limit: int = 50) -> list[dict]:
        if episode_id is None:
            rows = self.db.execute("SELECT * FROM risks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM risks WHERE episode_id=? ORDER BY id DESC LIMIT ?", (episode_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- credit accounting ------------------------------------------------
    @_transaction
    def note_recall(self, step_ids: list[int]) -> None:
        """Record that these experiences were surfaced to the kernel."""
        ids = [int(i) for i in step_ids if i]
        if not ids:
            return
        self.db.executemany("UPDATE steps SET recalls = recalls + 1 WHERE id = ?",
                            [(i,) for i in ids])
        self.db.commit()
        self._mutated_steps.extend(ids)

    @_transaction
    def note_outcome(self, step_id: int, adopted: bool) -> None:
        """Record whether the decision that followed agreed with this step."""
        step_id = int(step_id)
        # Static SQL branches instead of an f-string column name: the column
        # is never data, and a stable statement is both safer and kinder to
        # the query planner's statement cache.
        if adopted:
            self.db.execute("UPDATE steps SET adopted = adopted + 1 WHERE id = ?", (step_id,))
        else:
            self.db.execute("UPDATE steps SET rejected = rejected + 1 WHERE id = ?", (step_id,))
        self.db.commit()
        self._mutated_steps.append(step_id)

    @_transaction
    def steps_by_ids(self, step_ids: list[int]) -> list[Step]:
        """Re-read specific steps — the delta refresh for index snapshots."""
        if not step_ids:
            return []
        marks = ",".join("?" * len(step_ids))
        return [Step(**dict(r)) for r in
                self.db.execute(f"SELECT * FROM steps WHERE id IN ({marks})", step_ids)]

    @_transaction
    def mutated_steps_since(self, seq: int) -> tuple[list[int], int]:
        """Ids of steps mutated in place since `seq`, plus the new sequence number.

        This is process-local bookkeeping: it detects mutations made through
        this Store instance, which is the instance every kernel in the process
        shares.
        """
        return self._mutated_steps[seq:], len(self._mutated_steps)

    @_transaction
    def credit_report(self, limit: int = 50) -> list[dict]:
        """Per-memory credit, most-decided first. Feeds /api/credit."""
        rows = self.db.execute(
            "SELECT id, episode_id, substr(state, 1, 120) AS state, action, ret, "
            "       recalls, adopted, rejected "
            "FROM steps WHERE (adopted + rejected) > 0 "
            "ORDER BY (adopted + rejected) DESC, id DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r["id"], "episode_id": r["episode_id"], "state": r["state"],
                "action": r["action"], "ret": r["ret"],
                "recalls": r["recalls"], "adopted": r["adopted"], "rejected": r["rejected"],
                "credit": round(credit_factor(r["adopted"], r["rejected"]), 4),
            })
        return out

    @_transaction
    def credit_totals(self) -> dict:
        row = self.db.execute(
            "SELECT COUNT(*) AS steps, "
            "       COALESCE(SUM(recalls), 0) AS recalls, "
            "       COALESCE(SUM(adopted), 0) AS adopted, "
            "       COALESCE(SUM(rejected), 0) AS rejected FROM steps"
        ).fetchone()
        decided = row["adopted"] + row["rejected"]
        return {
            "steps": row["steps"], "recalls": row["recalls"],
            "adopted": row["adopted"], "rejected": row["rejected"],
            "decisions": decided,
            "adoption_rate": round(row["adopted"] / decided, 4) if decided else None,
        }
