"""Token thrift: stable prefixes, compacted observations, a reconcilable ledger.

Every number in here is meant to be checked against the provider's own usage
report, because the single most common complaint about agent tooling is a cost
HUD that does not match the vendor bill.

Ordering rule (matches how server-side prompt caching actually works):
    tools -> fixed system instructions -> history -> live state -> memory recall
Blocks are ordered by volatility: least-volatile first. Recall is frozen once
per episode (loop.py snapshots it at episode start) and history only ever
appends within an episode, so together with the fixed system prefix they form
a byte-stable cached prefix; only the state block (and the candidate list the
loop appends after it) is re-tokenized each step.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_WS = re.compile(r"[ \t]+")
_DUP_BLANK = re.compile(r"\n{3,}")
_DATA_URL = re.compile(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+")
_LONG_TOKEN = re.compile(r"\b[0-9a-f]{24,}\b")


def estimate_tokens(text: str) -> int:
    """Cheap heuristic (4 chars/token). Real counts come back in `usage`."""
    return max(1, len(text) // 4)


@dataclass
class PromptPlan:
    system: str
    user: str
    prefix: str
    est_prompt_tokens: int
    prefix_hash: str
    savings: dict = field(default_factory=dict)

    def messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.user}]


class ContextBuilder:
    def __init__(self, history_budget_chars: int = 6_000, observation_budget_chars: int = 5_000,
                 compact: bool = True):
        # History is append-only within an episode: a sliding window rewrites the
        # head of the user block every step, which invalidates the cached prefix
        # the history block exists to protect. The char budget only kicks in for
        # pathological episodes (default 6000 chars ≈ far more than max_steps=12
        # ever produces).
        self.history_budget = history_budget_chars
        self.observation_budget = observation_budget_chars
        self.compact = compact

    def build(self, instructions: str, catalog: str, recall: list[str], history: list[str],
              state: str, scope: str = "") -> PromptPlan:
        """Assemble the prompt so volatility increases towards the tail."""
        fixed = []
        if catalog:
            fixed.append(f"# tools\n{catalog}")
        fixed.append(f"# mission\n{instructions}")
        if scope:
            fixed.append(f"# scope\n{scope}")
        prefix = "\n\n".join(fixed)

        # Within the user block the same volatility rule applies: append-only
        # history first, then the per-step state, then episode-frozen recall last.
        # Candidates (even more volatile) are appended by the loop after recall.
        tail_blocks = []
        history_block = self._history_block(history)
        if history_block:
            tail_blocks.append(history_block)

        compacted, stats = (compact_observation(state, self.observation_budget)
                            if self.compact else (state, {}))
        if state:
            tail_blocks.append(f"# current state\n{compacted}")
        if recall:
            tail_blocks.append("# what worked here before\n" + "\n".join(f"- {line}" for line in recall))

        system = prefix
        user = "\n\n".join(tail_blocks)
        full = system + user
        return PromptPlan(
            system=system, user=user, prefix=prefix,
            est_prompt_tokens=estimate_tokens(full),
            prefix_hash=hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:12],
            savings=stats,
        )

    def _history_block(self, history: list[str]) -> str:
        """Render the full step history, truncating the OLDEST entries on budget.

        Truncation is bounded: with the default 6000-char budget and max_steps=12
        it never fires, so within an episode the block only ever grows at the tail.
        """
        if not history:
            return ""
        entries = [f"{i + 1}. {h}" for i, h in enumerate(history)]
        body = "\n".join(entries)
        if len(body) > self.history_budget:
            dropped = 0
            while entries and len("\n".join(entries)) > self.history_budget:
                entries.pop(0)
                dropped += 1
            # Keep the original numbering so the model can still see how far
            # into the episode it is; the marker says why the count starts late.
            body = f"[{dropped} earlier steps omitted]\n" + "\n".join(entries)
        return "# recent steps\n" + body


def compact_observation(text: str, budget_chars: int = 5_000) -> tuple[str, dict]:
    """Shrink a raw observation without dropping the lines that carry decisions.

    Strategy, in order of payoff: drop base64 payloads, dedupe identical lines,
    collapse whitespace, strip 24+ hex blobs (ids nobody reasons about), then
    truncate in the middle so both the head (page/summary) and tail (last lines)
    survive.
    """
    original = text or ""
    if not original:
        return "", {"orig_chars": 0, "kept_chars": 0, "saved_pct": 0.0}

    work = _DATA_URL.sub("[image omitted]", original)
    work = _LONG_TOKEN.sub("[id]", work)
    work = _WS.sub(" ", work)
    work = _DUP_BLANK.sub("\n\n", work)

    seen: set[str] = set()
    kept: list[str] = []
    duplicates = 0
    for line in work.splitlines():
        key = line.strip()
        if not key:
            if kept and kept[-1] == "":
                continue
            kept.append("")
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(key)
    work = "\n".join(kept).strip()

    if len(work) > budget_chars:
        head = int(budget_chars * 0.6)
        tail = budget_chars - head
        omitted = len(work) - head - tail
        work = f"{work[:head]}\n... [{omitted} chars omitted] ...\n{work[-tail:]}"

    return work, {
        "orig_chars": len(original), "kept_chars": len(work),
        "saved_pct": round(100 * (1 - len(work) / max(len(original), 1)), 1),
        "dropped_duplicate_lines": duplicates,
    }


@dataclass
class PriceTable:
    """USD per million tokens. Empty by default: no invented prices."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None       # typically ~0.1x input when caching is on

    def usd(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        fresh = max(prompt_tokens - cached_tokens, 0)
        cached_rate = self.cache_read if self.cache_read is not None else self.input
        return (fresh * self.input + cached_tokens * cached_rate + completion_tokens * self.output) / 1e6


class Ledger:
    """Per-episode token accounting, with the server's own cache-hit report."""

    def __init__(self, price: PriceTable | None = None):
        self.price = price or PriceTable()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.calls = 0
        self.per_step: list[dict] = []

    def record(self, usage, note: dict | None = None) -> dict:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.cached_tokens += usage.cached_tokens
        self.calls += max(usage.calls, 1)
        row = {
            "prompt": usage.prompt_tokens, "completion": usage.completion_tokens,
            "cached": usage.cached_tokens,
            "cache_hit_pct": round(100 * usage.cache_hit_rate, 1),
            "usd": round(self.price.usd(usage.prompt_tokens, usage.completion_tokens,
                                        usage.cached_tokens), 6),
            **(note or {}),
        }
        self.per_step.append(row)
        return row

    def summary(self) -> dict:
        total = self.prompt_tokens + self.completion_tokens
        return {
            "calls": self.calls, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "cached_tokens": self.cached_tokens,
            "cache_hit_pct": round(100 * self.cached_tokens / max(self.prompt_tokens, 1), 1),
            "total_tokens": total,
            "est_usd": round(self.price.usd(self.prompt_tokens, self.completion_tokens,
                                            self.cached_tokens), 6),
        }


@dataclass
class TierRouter:
    """Ask the cheap model first; escalate only when it is unsure or the step is risky."""

    cheap: str = ""
    strong: str = ""
    escalate_margin: float = 0.12
    escalate_on_risk: tuple[str, ...] = ("dangerous",)
    escalations: int = 0
    total_steps: int = 0

    def pick(self, choice_z: dict[str, float], risk: str = "") -> str:
        self.total_steps += 1
        if not self.strong or not self.cheap:
            return self.strong or self.cheap
        top = sorted(choice_z.values(), reverse=True)
        thin = len(top) >= 2 and (top[0] - top[1]) < self.escalate_margin
        if thin or risk in self.escalate_on_risk:
            self.escalations += 1
            return self.strong
        return self.cheap

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.total_steps if self.total_steps else 0.0
