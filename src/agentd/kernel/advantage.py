"""Advantage estimation, faithful to JitRL's reference semantics (jitrl_agent.py:280-330).

Two details that are easy to get wrong and change behaviour:

1. Unseen actions do *not* always get an exploration bonus. With probability
   `exploration_prob` they get `baseline + alpha/count`; otherwise they are
   pinned to 0 — which drags the baseline down and *sharpens* the advantage of
   actions that history says are good.
2. After unseen actions are assigned values, the baseline is recomputed over all
   action values, and advantages are measured against that recomputed baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .retrieval import Neighbor
from .state import normalize_action


@dataclass
class AdvantageResult:
    raw: dict[str, float] = field(default_factory=dict)
    normalized: dict[str, float] = field(default_factory=dict)
    action_means: dict[str, float] = field(default_factory=dict)
    baseline_before: float = 0.0
    baseline: float = 0.0
    explored: list[str] = field(default_factory=list)
    n_neighbors: int = 0
    scope: str = ""
    notes: str = ""


def estimate(
    candidates: list[str],
    neighbors: list[Neighbor],
    *,
    exploration_prob: float = 0.05,
    alpha: float = 1.0,
    rng: random.Random | None = None,
    scope: str = "",
) -> AdvantageResult:
    rng = rng or random.Random()
    result = AdvantageResult(n_neighbors=len(neighbors), scope=scope)

    if scope:
        scoped = [n for n in neighbors if (n.scope or "") == scope]
        if scoped:
            neighbors = scoped
            result.notes = f"scoped to '{scope}'"

    if not neighbors or not candidates:
        result.notes = (result.notes + "; " if result.notes else "") + "no retrieved experience; advantage=0"
        result.normalized = {normalize_action(c): 0.0 for c in candidates}
        return result

    grouped: dict[str, list[float]] = {}
    for n in neighbors:
        grouped.setdefault(normalize_action(n.action), []).append(n.ret)

    all_returns = [r for rs in grouped.values() for r in rs]
    count = max(len(all_returns), 1)
    baseline_before = sum(all_returns) / len(all_returns)

    values: dict[str, float] = {a: sum(rs) / len(rs) for a, rs in grouped.items()}

    for candidate in candidates:
        key = normalize_action(candidate)
        if key in values:
            continue
        if rng.random() < exploration_prob:
            values[key] = baseline_before + alpha / count
            result.explored.append(key)
        else:
            values[key] = 0.0

    baseline = sum(values.values()) / len(values)
    result.baseline_before = baseline_before
    result.baseline = baseline
    result.action_means = values
    result.raw = {key: value - baseline for key, value in values.items()}
    result.normalized = _normalize(result.raw)
    if result.explored:
        result.notes = (result.notes + "; " if result.notes else "") + f"explored {len(result.explored)} unseen action(s)"
    return result


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    """Scale advantages into [-1, 1]: divide by the largest positive, else by |largest negative|."""
    if not raw:
        return {}
    positives = [v for v in raw.values() if v > 0]
    negatives = [v for v in raw.values() if v < 0]
    if positives:
        scale = max(positives)
    elif negatives:
        scale = abs(min(negatives))
    else:
        return {k: 0.0 for k in raw}
    if not scale:
        return {k: 0.0 for k in raw}
    return {k: v / scale for k, v in raw.items()}
