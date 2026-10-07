"""The kernel the whole product hangs on: retrieve, estimate advantage, nudge logits.

This is JitRL (arXiv:2601.18510) reduced to something you can run against *any*
model endpoint, with the two things the reference implementation left implicit
made configurable: the bias strength (beta) and where the decision token sits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .advantage import AdvantageResult, estimate
from .state import normalize_action
from .episode import EpisodeRecorder
from .rerank import RerankResult, rerank
from .retrieval import Neighbor, Retriever
from .store import Store


@dataclass
class Decision:
    rerank: RerankResult
    neighbors: list[Neighbor] = field(default_factory=list)
    advantage: AdvantageResult = field(default_factory=AdvantageResult)

    @property
    def chosen_action(self) -> str:
        return self.rerank.chosen.text

    def summary(self) -> dict:
        return {
            "mode": self.rerank.mode,
            "beta": self.rerank.beta,
            "chosen_index": self.rerank.chosen_index,
            "chosen_action": self.chosen_action,
            "retrieved": len(self.neighbors),
            "baseline": round(self.advantage.baseline, 6),
            "note": self.advantage.notes,
            "options": self.rerank.trace_rows(),
        }


class JitRLKernel:
    def __init__(
        self,
        store: Store | None = None,
        beta: float = 1.0,
        gamma: float = 0.95,
        top_k: int = 8,
        ngram: int = 2,
        min_sim: float = 0.02,
        exploration_prob: float = 0.05,
        alpha: float = 1.0,
        seed: int | None = None,
        enabled: bool = True,
        recall_analysis: int = 2,
        track_credit: bool = True,
    ):
        self.store = store or Store()
        self.retriever = Retriever(self.store, ngram=ngram, top_k=top_k, min_sim=min_sim)
        self.beta = beta
        self.gamma = gamma
        self.exploration_prob = exploration_prob
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.enabled = enabled
        self.recall_analysis = recall_analysis
        # Off gives the A/B arm: same retrieval, no credit feedback.
        self.track_credit = track_credit

    def decide(
        self,
        state_text: str,
        candidates: list[str],
        z: dict[str, float],
        mode: str = "token",
        reasoning: str | None = None,
        scope: str = "",
    ) -> Decision:
        if not self.enabled:
            # Control arm: pure model choice, memory never consulted.
            result = rerank(candidates, z, {}, beta=0.0, mode=mode, reasoning=reasoning)
            return Decision(result, [], AdvantageResult(scope=scope, notes="memory disabled"))

        neighbors = self.retriever.neighbors(state_text)
        # Count the recall before deciding, so a step is credited for being
        # surfaced even if this decision goes a different way.
        self.store.note_recall([n.step_id for n in neighbors])
        adv = estimate(candidates, neighbors, exploration_prob=self.exploration_prob,
                       alpha=self.alpha, rng=self.rng, scope=scope)
        result = rerank(candidates, z, adv.normalized, beta=self.beta, mode=mode, reasoning=reasoning)
        decision = Decision(result, neighbors, adv)
        if self.track_credit:
            # Agreement is judged on the normalised action, the same key the
            # advantage estimator groups by -- otherwise punctuation differences
            # would make every recall look rejected.
            chosen_fp = normalize_action(result.chosen.text)
            for n in neighbors:
                self.store.note_outcome(n.step_id, normalize_action(n.action) == chosen_fp)
        return decision

    def begin(self, task: str, goal: str = "", meta: dict | None = None) -> EpisodeRecorder:
        return EpisodeRecorder(self.store, task, goal, gamma=self.gamma, meta=meta)

    def analyses_for(self, task: str) -> list[str]:
        return self.store.recent_analyses(task, limit=self.recall_analysis)

    def stats(self) -> dict:
        return {
            "episodes": self.store.count_episodes(),
            "steps": self.store.count_steps(),
            "enabled": self.enabled,
            "beta": self.beta,
            "gamma": self.gamma,
            "track_credit": self.track_credit,
            "credit": self.store.credit_totals(),
        }
