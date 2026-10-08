"""Native protocol providers: Anthropic Messages and Gemini generateContent.

Why these exist
---------------
`openai_compat` can use the JitRL TOKEN tier because `/v1/chat/completions`
returns `logprobs`. Neither of these two protocols does:

* Anthropic Messages returns no per-token logprobs.
* Gemini `generateContent` returns no per-token logprobs.

So both are pinned to DecodeMode.VERBALIZED -- the same tier the JitRL authors
describe as "for models that do not expose log-probabilities, we prompt the LLM
to explicitly output a confidence score and transform it to logits".

A note on beta calibration
--------------------------
VERBALIZED scores land in a different numeric range than TOKEN scores, so the
two tiers must not be assumed to want the same beta. See notes_score_space() for
the measured ranges, and the beta sweep in `bench.py` for why this is a
calibration problem rather than a correctness one.
"""

from __future__ import annotations

import time

from .base import Choice, DecodeMode, Provider, ProviderError, ScoreSpace, Usage
from .http import ProviderHTTPStatus, post_json
from .openai_compat import _parse_index, _parse_scores

# VERBALIZED maps a self-reported 0-100 confidence onto this curve. It is the
# same transform openai_compat uses, deliberately: keeping one formula means a
# beta chosen for one provider's verbalized tier transfers to another, instead
# of every provider inventing its own scale.
def confidence_to_z(raw: float) -> float:
    p = max(raw, 1.0) / 100.0
    return _log(p) + p


def _log(x: float) -> float:
    import math

    return math.log(x)


_VERBALIZED_SUFFIX = (
    "\n\nAfter choosing, output a JSON object on the LAST line: "
    '{"choice": <number>, "confidence": <0-100>, "scores": {<option number>: <0-100>, ...}}'
)


class _NativeJSONProvider(Provider):
    """Shared plumbing: POST JSON, get one text blob back, score it verbally."""

    endpoint_path = ""
    default_base_url = ""
    max_output_tokens = 600

    def __init__(self, spec):
        super().__init__(spec)
        # ProviderSpec defaults kind to openai_compat; a native provider that
        # reports the wrong kind makes /api/state and health checks lie.
        self.spec.kind = self.kind

    def _headers(self) -> dict:
        raise NotImplementedError

    def _payload(self, system: str, user: str, max_tokens: int, temperature: float) -> dict:
        raise NotImplementedError

    def _text_of(self, data: dict) -> str:
        raise NotImplementedError

    def _usage_of(self, data: dict) -> Usage:
        raise NotImplementedError

    def _post(self, body: dict, timeout: int | None = None) -> dict:
        url = (self.spec.base_url or self.default_base_url).rstrip("/") + self.endpoint_path
        try:
            return post_json(url, headers=self._headers(), payload=body,
                             timeout=timeout or self.spec.timeout_s)
        except ProviderHTTPStatus as exc:
            # Upstream bodies can echo the prompt or the key; never relay them.
            raise ProviderError(f"{self.label}: HTTP {exc.code} from endpoint") from exc
        except ProviderError as exc:
            raise ProviderError(f"{self.label}: {type(exc).__name__}: {exc}") from exc

    # -- Provider contract -------------------------------------------------
    def choose(self, system: str, user: str, candidates: list[str]) -> Choice:
        t0 = time.time()
        data = self._post(
            self._payload(system, user + _VERBALIZED_SUFFIX, 160, self.spec.temperature)
        )
        text = self._text_of(data)
        parsed = _parse_scores(text, len(candidates)) or _parse_index_scores(text, len(candidates))
        if not parsed:
            # Never invent a ranking. Equal scores mean "the model gave us
            # nothing", and a flat field is the honest representation of that.
            parsed = {i: 50.0 for i in range(1, len(candidates) + 1)}
        z = {a: confidence_to_z(parsed.get(i, 1.0)) for i, a in enumerate(candidates, start=1)}
        return Choice(
            z=z,
            raw_text=text,
            mode=DecodeMode.VERBALIZED,
            score_space=ScoreSpace.LOGPROB,
            usage=self._usage_of(data),
            latency_ms=int((time.time() - t0) * 1000),
            notes="native protocol: no logprobs, scored via self-reported confidence",
        )

    def text(self, prompt: str) -> tuple[str, Usage]:
        data = self._post(self._payload("", prompt, self.max_output_tokens, self.spec.temperature))
        return self._text_of(data), self._usage_of(data)

    def health(self) -> dict:
        base = super().health()
        base["logprobs"] = False
        base["tier"] = DecodeMode.VERBALIZED.value
        return base


def _parse_index_scores(text: str, n: int) -> dict[int, float]:
    """Fallback when the model gave a bare choice but no per-option scores."""
    idx = _parse_index(text, n)
    if not idx:
        return {}
    out = {i: 100.0 for i in range(1, n + 1)}
    out[idx] = 100.0
    return out


class AnthropicProvider(_NativeJSONProvider):
    kind = "anthropic"
    default_base_url = "https://api.anthropic.com"
    endpoint_path = "/v1/messages"
    default_version = "2023-06-01"

    def _headers(self) -> dict:
        return {
            "x-api-key": self.spec.api_key,
            "anthropic-version": self.spec.extra_headers.get("anthropic_version", self.default_version),
        }

    def _payload(self, system: str, user: str, max_tokens: int, temperature: float) -> dict:
        body = {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            body["system"] = system
        if temperature:
            body["temperature"] = temperature
        return body

    def _text_of(self, data: dict) -> str:
        parts = [b.get("text", "") for b in (data.get("content") or []) if b.get("type") == "text"]
        return "\n".join(parts).strip()

    def _usage_of(self, data: dict) -> Usage:
        u = data.get("usage") or {}
        return Usage(
            prompt_tokens=int(u.get("input_tokens") or 0),
            completion_tokens=int(u.get("output_tokens") or 0),
            cached_tokens=int(u.get("cache_read_input_tokens") or 0),
        )


class GeminiProvider(_NativeJSONProvider):
    kind = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com"

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.spec.api_key}

    def endpoint_for(self, model: str) -> str:
        return f"/v1beta/models/{model}:generateContent"

    @property
    def endpoint_path(self) -> str:  # type: ignore[override]
        return self.endpoint_for(self.spec.model)

    def _payload(self, system: str, user: str, max_tokens: int, temperature: float) -> dict:
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                # Deterministic choice is the point; the verbalized tier relies on
                # the model committing to a number rather than sampling variety.
                "candidateCount": 1,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _text_of(self, data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()

    def _usage_of(self, data: dict) -> Usage:
        u = data.get("usageMetadata") or {}
        return Usage(
            prompt_tokens=int(u.get("promptTokenCount") or 0),
            completion_tokens=int(u.get("candidatesTokenCount") or 0),
            cached_tokens=int(u.get("cachedContentTokenCount") or 0),
        )


NATIVE_PROVIDERS = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}
