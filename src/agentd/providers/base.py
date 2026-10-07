"""Provider contract: the model layer only ever has to answer one question.

Given a state and a numbered list of candidate actions, return a score per
candidate (z) in whichever space the endpoint can supply, plus token usage.
Everything above this layer is decoding-strategy agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class DecodeMode(str, Enum):
    TOKEN = "token"          # endpoint exposes top_logprobs -> JitRL as published
    N_SAMPLE = "n_sample"    # endpoint gives no per-candidate mass -> sample k times
    VERBALIZED = "verbalized"  # endpoint can't help -> model grades options in words


class ScoreSpace(str, Enum):
    PROB = "prob"        # z = exp(logprob); matches the reference implementation
    LOGPROB = "logprob"  # z = logprob; closer to the paper's logit formulation


class ProviderError(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            calls=self.calls + other.calls,
        )


@dataclass
class Choice:
    """Model-side scores for one decision point."""

    z: dict[str, float] = field(default_factory=dict)
    raw_text: str = ""
    reasoning: str | None = None
    mode: DecodeMode = DecodeMode.TOKEN
    score_space: ScoreSpace = ScoreSpace.PROB
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    notes: str = ""


@dataclass
class ProviderSpec:
    label: str
    model: str
    base_url: str = ""
    api_key: str = ""
    kind: str = "openai_compat"
    tier: str = "strong"          # cheap | strong — drives difficulty-based routing
    temperature: float = 0.0
    max_tokens: int = 8
    decode_mode: DecodeMode | None = None   # force, skipping capability probe
    score_space: ScoreSpace = ScoreSpace.PROB
    extra_headers: dict = field(default_factory=dict)
    timeout_s: int = 120


class Provider(ABC):
    def __init__(self, spec: ProviderSpec):
        self.spec = spec

    @property
    def label(self) -> str:
        return self.spec.label

    @abstractmethod
    def choose(self, system: str, user: str, candidates: list[str]) -> Choice:
        """Score `candidates` for the state described in `user`."""

    def text(self, prompt: str) -> tuple[str, Usage]:
        """Free-text channel, used by the council's critique phase."""
        raise NotImplementedError(f"provider '{self.spec.label}' has no free-text channel")

    def health(self) -> dict:
        return {"label": self.label, "model": self.spec.model, "kind": self.spec.kind}
