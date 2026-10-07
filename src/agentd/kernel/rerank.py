"""z'(s,a) = z(s,a) + beta * A(s,a) — the whole of JitRL's policy improvement."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class RankedOption:
    index: int
    text: str
    z: float
    advantage: float
    z_prime: float


@dataclass
class RerankResult:
    chosen_index: int
    options: list[RankedOption] = field(default_factory=list)
    mode: str = "token"
    beta: float = 1.0
    reasoning: str | None = None

    @property
    def chosen(self) -> RankedOption:
        return next(o for o in self.options if o.index == self.chosen_index)

    def trace_rows(self) -> list[dict]:
        return [
            {
                "index": o.index, "action": o.text, "z": round(o.z, 6),
                "advantage": round(o.advantage, 6), "z_prime": round(o.z_prime, 6),
                "chosen": o.index == self.chosen_index,
            }
            for o in self.options
        ]


def rerank(
    candidates: list[str],
    z: dict[str, float],
    advantages: dict[str, float],
    beta: float = 1.0,
    mode: str = "token",
    reasoning: str | None = None,
    tie_break_seed: int | None = None,
) -> RerankResult:
    """Score every candidate and pick the argmax of z + beta*A.

    beta=0 degenerates to the plain model choice, which is exactly the control
    arm used to prove the memory is doing something.
    """
    from .state import normalize_action

    options: list[RankedOption] = []
    for i, text in enumerate(candidates, start=1):
        key = normalize_action(text)
        z_i = float(z.get(key, z.get(text, float("-inf"))))
        adv_i = float(advantages.get(key, 0.0))
        z_prime = z_i + beta * adv_i
        options.append(RankedOption(index=i, text=text, z=z_i, advantage=adv_i, z_prime=z_prime))

    if not options:
        raise ValueError("no candidates to rerank")

    best = max(o.z_prime for o in options)
    winners = [o for o in options if o.z_prime == best]
    if len(winners) > 1:
        chosen = random.Random(tie_break_seed).choice(winners)
    else:
        chosen = winners[0]

    return RerankResult(
        chosen_index=chosen.index, options=options, mode=mode, beta=beta, reasoning=reasoning
    )
