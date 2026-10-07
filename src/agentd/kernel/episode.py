"""Episode buffer: collects a trajectory and persists discounted returns."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .state import fingerprint, normalize_action
from .store import Step, Store


@dataclass
class StepRecord:
    t: int
    state: str
    action: str
    z: float | None = None
    adv: float | None = None
    z_prime: float | None = None
    chosen_index: int | None = None
    scope: str = ""
    reward: float = 0.0
    meta: dict = field(default_factory=dict)


class EpisodeRecorder:
    """G_t = r_t + gamma * G_{t+1}, computed when the episode ends."""

    def __init__(self, store: Store, task: str, goal: str = "", gamma: float = 0.5, meta: dict | None = None):
        self.store = store
        self.task = task
        self.goal = goal
        self.gamma = gamma
        self.episode_id = store.start_episode(task, goal, meta)
        self.steps: list[StepRecord] = []
        self._t = 0

    def record(self, state: str, action: str, z=None, adv=None, z_prime=None,
               chosen_index=None, reward: float = 0.0, scope: str = "",
               meta: dict | None = None) -> StepRecord:
        rec = StepRecord(
            t=self._t, state=state, action=action, z=z, adv=adv, z_prime=z_prime,
            chosen_index=chosen_index, reward=reward, scope=scope, meta=meta or {},
        )
        self.steps.append(rec)
        self._t += 1
        return rec

    def returns(self) -> list[float]:
        out: list[float] = []
        g = 0.0
        for rec in reversed(self.steps):
            g = rec.reward + self.gamma * g
            out.append(g)
        return list(reversed(out))

    def finish(self, success: bool | None, score: float | None = None, analysis: str = "") -> dict:
        returns = self.returns()
        rows_written = 0
        for rec, ret in zip(self.steps, returns):
            self.store.add_step(
                Step(
                    episode_id=self.episode_id, t=rec.t, state=rec.state,
                    state_fp=fingerprint(rec.state), action=rec.action,
                    action_fp=normalize_action(rec.action), scope=rec.scope, z=rec.z, adv=rec.adv,
                    z_prime=rec.z_prime, chosen=rec.chosen_index, reward=rec.reward, ret=ret,
                )
            )
            rows_written += 1
        self.store.finish_episode(self.episode_id, success, score, analysis)
        return {
            "episode_id": self.episode_id, "steps": rows_written,
            "success": success, "score": score, "finished_at": time.time(),
        }
