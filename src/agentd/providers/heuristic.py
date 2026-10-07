"""A non-LLM policy that actually reads the page.

Why this exists: the web suite should be measurable on a laptop with no API key, so
that "does memory help?" can be answered before anyone pays for tokens. It is a real
policy — lexical goal/label matching plus role priors and seeded noise — and it is
labelled as one. It is not a substitute for a language model on ambiguous pages, and
`agentd bench` prints which policy produced the numbers.
"""

from __future__ import annotations

import random
import re

from .base import Choice, DecodeMode, Provider, ProviderSpec, ScoreSpace, Usage

_GOAL_RE = re.compile(r"Task:\s*(.+?)(?:\n|$)", re.S)
_ELEMS = re.compile(r"^\[(\d+)\]\s+(\S+).*?:\s*(.*)$", re.M)
_CLICK = re.compile(r"^click\s+\[(\d+)\]", re.I)
_TYPE = re.compile(r"^type\s+\[(\d+)\]", re.I)
_SELECT = re.compile(r"^select\s+\[(\d+)\]", re.I)

_FILL_HINTS = ("fill", "enter", "type", "name", "email", "quantity", "password",
               "username", "describe", "set a")
_KIND_PRIOR = {"button": 0.30, "submit": 0.32, "link": 0.24, "a": 0.24, "checkbox": 0.26,
               "radio": 0.20, "text": 0.22, "email": 0.22, "password": 0.20, "search": 0.22,
               "textarea": 0.22, "number": 0.20, "select-one": 0.18}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z0-9']+", (text or "").lower()) if len(w) > 2}


class HeuristicModel(Provider):
    def __init__(self, seed: int = 7, noise: float = 0.18):
        super().__init__(ProviderSpec(label="heuristic", model="heuristic-lexical", kind="heuristic"))
        self.rng = random.Random(seed)
        self.noise = noise
        self.calls: list[dict] = []

    def _scores(self, user: str, candidates: list[str]) -> dict[str, float]:
        goal_match = _GOAL_RE.search(user or "")
        goal = goal_match.group(1) if goal_match else (user or "")
        goal_tokens = _tokens(goal)

        elements = {int(m.group(1)): (m.group(2).lower(), m.group(3).lower())
                    for m in _ELEMS.finditer(user or "")}

        scores: dict[str, float] = {}
        for candidate in candidates:
            lowered = candidate.lower().strip()
            if lowered.startswith(("scroll", "wait")):
                scores[candidate] = 0.05 + self.rng.gauss(0, self.noise / 3)
                continue

            match = _CLICK.match(candidate) or _TYPE.match(candidate) or _SELECT.match(candidate)
            if not match:
                scores[candidate] = 0.1
                continue
            index = int(match.group(1))
            kind, label = elements.get(index, ("", candidate.lower()))

            overlap = len(goal_tokens & _tokens(label)) / max(len(goal_tokens), 1)
            score = _KIND_PRIOR.get(kind, 0.18) + 1.7 * overlap

            if _TYPE.match(candidate):
                wants_input = any(hint in goal.lower() for hint in _FILL_HINTS)
                field_hint = any(hint in label for hint in ("name", "email", "qty", "quantity",
                                                            "password", "user", "describe", "value"))
                score = (0.34 if wants_input and field_hint else 0.10 if wants_input else 0.22)
                score += 1.2 * overlap
            if kind == "checkbox" or "checkbox" in lowered:
                number = re.search(r"(\d+)", label)
                target = re.search(r"(\d+)", " ".join(re.findall(r"totalling exactly \d+|total \d+|exactly \d+", goal.lower())))
                if number and target:
                    score = 0.85 if int(number.group(1)) == int(target.group(1)) else 0.45
                if "submit" in goal.lower():
                    score = max(score, 0.35)

            if any(word in goal.lower() for word in ("dismiss", "first", "banner", "accept")):
                if "accept" in label or "dismiss" in label:
                    score += 0.5
            if "cheapest" in goal.lower() and any(c.isdigit() for c in label):
                digits = int(re.search(r"\d+", label).group(0))
                score += 0.6 - min(digits, 200) / 400.0
            if any(word in goal.lower() for word in ("then", "after", "next", "wizard", "create")):
                if "next" in label or "continue" in label or "confirm" in label or "create" in label:
                    score += 0.4
            if "skip" in label:
                score -= 0.35

            scores[candidate] = max(score + self.rng.gauss(0, self.noise), 0.01)
        return scores

    def choose(self, system: str, user: str, candidates: list[str]) -> Choice:
        self.calls.append({"candidates": list(candidates)})
        z = self._scores(user, candidates)
        return Choice(z=z, raw_text=max(z, key=z.get) if z else "", mode=DecodeMode.TOKEN,
                      score_space=ScoreSpace.PROB,
                      usage=Usage(prompt_tokens=len(user) // 4, completion_tokens=2),
                      notes="heuristic lexical policy")

    def text(self, prompt: str) -> tuple[str, Usage]:
        return '{"objections": []}', Usage(prompt_tokens=len(prompt) // 4, completion_tokens=8)
