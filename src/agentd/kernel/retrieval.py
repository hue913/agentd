"""Jaccard n-gram retrieval over stored experience.

JitRL deliberately uses lexical n-gram similarity instead of an embedding model,
which keeps the kernel dependency-free and cheap. We add an inverted index so a
growing memory does not turn every step into a full table scan.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import ngrams, normalize_action, normalize_state, tokenize
from .store import Store


@dataclass
class Neighbor:
    step_id: int
    episode_id: int
    state: str
    action: str
    action_fp: str
    scope: str
    ret: float
    sim: float


class Retriever:
    def __init__(self, store: Store, ngram: int = 2, top_k: int = 8, min_sim: float = 0.02):
        self.store = store
        self.ngram = ngram
        self.top_k = top_k
        self.min_sim = min_sim
        self._rows: list[tuple] = []
        self._postings: dict[str, list[int]] = {}
        self._built_at = -1

    def _ensure_index(self) -> None:
        watermark = self.store.max_step_id()
        if watermark == self._built_at:
            return
        self._rows = []
        self._postings = {}
        for i, step in enumerate(self.store.all_steps()):
            grams = ngrams(tokenize(step.state), self.ngram)
            self._rows.append((i, step, grams))
            for g in grams:
                self._postings.setdefault(g, []).append(i)
        self._built_at = watermark

    def neighbors(self, state_text: str) -> list[Neighbor]:
        """Most similar past states, ranked by Jaccard similarity of n-gram sets."""
        self._ensure_index()
        query = ngrams(tokenize(state_text), self.ngram)
        if not query or not self._rows:
            return []

        shared: dict[int, int] = {}
        for g in query:
            for row_i in self._postings.get(g, ()):
                shared[row_i] = shared.get(row_i, 0) + 1

        q_size = len(query)
        scored: list[Neighbor] = []
        for row_i, inter in shared.items():
            idx_i, step, grams = self._rows[row_i]
            union = q_size + len(grams) - inter
            sim = inter / union if union else 0.0
            if sim < self.min_sim:
                continue
            scored.append(
                Neighbor(
                    step_id=step.id, episode_id=step.episode_id, state=step.state,
                    action=step.action, action_fp=step.action_fp, scope=step.scope, ret=step.ret, sim=sim,
                )
            )
        scored.sort(key=lambda n: (-n.sim, -n.ret))
        return scored[: self.top_k]

    def action_returns(self, neighbors: list[Neighbor]) -> dict[str, list[float]]:
        """Group neighbour returns by normalised action, for advantage estimation."""
        grouped: dict[str, list[float]] = {}
        for n in neighbors:
            grouped.setdefault(normalize_action(n.action), []).append(n.ret)
        return grouped
