"""Deterministic offline provider: makes the kernel testable and demos keyless."""

from __future__ import annotations

from .base import Choice, DecodeMode, Provider, ProviderSpec, ScoreSpace, Usage


class MockProvider(Provider):
    """`weights` maps a candidate's text (or substring of it) to a score z."""

    def __init__(self, spec: ProviderSpec | None = None, weights: dict[str, float] | None = None,
                 mode: DecodeMode = DecodeMode.TOKEN, calls: list | None = None,
                 critiques: list[str] | None = None):
        super().__init__(spec or ProviderSpec(label="mock", model="mock", kind="mock"))
        self.weights = weights or {}
        self.mode = mode
        self.calls = calls if calls is not None else []
        self.critiques = critiques or []

    def choose(self, system: str, user: str, candidates: list[str]) -> Choice:
        self.calls.append({"system": system, "user": user, "candidates": list(candidates)})
        z: dict[str, float] = {}
        for action in candidates:
            hit = next((w for key, w in self.weights.items() if key in action), None)
            z[action] = float(hit) if hit is not None else 0.01
        chosen = max(candidates, key=lambda a: z[a])
        return Choice(
            z=z, raw_text=chosen, mode=self.mode, score_space=ScoreSpace.PROB,
            usage=Usage(prompt_tokens=len(system + user) // 4, completion_tokens=1),
            notes="mock",
        )

    def text(self, prompt: str) -> tuple[str, Usage]:
        reply = self.critiques.pop(0) if self.critiques else '{"objections": []}'
        self.calls.append({"text_prompt": prompt, "reply": reply})
        return reply, Usage(prompt_tokens=len(prompt) // 4, completion_tokens=20)
