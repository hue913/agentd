"""Jaccard n-gram retrieval over stored experience.

JitRL deliberately uses lexical n-gram similarity instead of an embedding model,
which keeps the kernel dependency-free and cheap. We add an inverted index so a
growing memory does not turn every step into a full table scan.
"""

from __future__ import annotations

from dataclasses import dataclass

from .credit import credit_factor
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
    recalls: int = 0
    adopted: int = 0
    rejected: int = 0
    credit: float = 1.0


class Retriever:
    def __init__(self, store: Store, ngram: int = 2, top_k: int = 8, min_sim: float = 0.02):
        self.store = store
        self.ngram = ngram
        self.top_k = top_k
        self.min_sim = min_sim
        self._rows: list[tuple] = []
        self._postings: dict[str, list[int]] = {}
        self._row_by_step: dict[int, int] = {}
        self._built_at = -1
        self._mutation_seq = 0

    def _ensure_index(self) -> None:
        mutated, self._mutation_seq = self.store.mutated_steps_since(self._mutation_seq)
        watermark = self.store.max_step_id()
        if watermark == self._built_at and not mutated:
            return
        if watermark < self._built_at:
            # the store shrank underneath us (reset/swap): start over
            self._rows = []
            self._postings = {}
            self._row_by_step = {}
            self._built_at = -1
        # Append-only: postings reference _rows subscripts, and subscripts only
        # grow, so new steps extend the index without touching existing rows.
        for step in self.store.steps_after(self._built_at):
            i = len(self._rows)
            grams = ngrams(tokenize(step.state), self.ngram)
            self._rows.append((i, step, grams))
            self._row_by_step[step.id] = i
            for g in grams:
                self._postings.setdefault(g, []).append(i)
        self._built_at = watermark
        if mutated:
            # Credit counters (and outcomes) are written AFTER a step lands in
            # the index; re-read just those rows so ranking stays honest.
            known = [sid for sid in set(mutated) if sid in self._row_by_step]
            for step in self.store.steps_by_ids(known):
                i = self._row_by_step[step.id]
                _, _, grams = self._rows[i]
                self._rows[i] = (i, step, grams)

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
                    recalls=step.recalls, adopted=step.adopted, rejected=step.rejected,
                )
            )
        # Credit multiplies similarity; it never replaces it. A step with no
        # outcomes yet has factor exactly 1.0 and is untouched, so this can only
        # re-order experiences that were already similar enough to be retrieved --
        # it can never pull an unrelated step into the top-k.
        for n in scored:
            n.credit = credit_factor(n.adopted, n.rejected)
        scored.sort(key=lambda n: (-(n.sim * n.credit), -n.ret))
        return scored[: self.top_k]

    def action_returns(self, neighbors: list[Neighbor]) -> dict[str, list[float]]:
        """Group neighbour returns by normalised action, for advantage estimation."""
        grouped: dict[str, list[float]] = {}
        for n in neighbors:
            grouped.setdefault(normalize_action(n.action), []).append(n.ret)
        return grouped
