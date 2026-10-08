"""Zero-dependency OpenAI-compatible client (stdlib urllib only).

Works against llama.cpp's llama-server, vLLM, OpenRouter, DeepSeek, Moonshot,
SiliconFlow, Ollama, LM Studio — anything speaking /v1/chat/completions. This is
what "plug in any model" means here: base_url + api_key + model name.
"""

from __future__ import annotations

import math
import time

from .base import Choice, DecodeMode, Provider, ProviderError, ScoreSpace, Usage
from .http import post_json
from .jsonutil import balanced_objects as _balanced_objects  # noqa: F401  (forwarding alias)

_DIGITS = {str(i) for i in range(1, 100)}


def _digit_value(token: str) -> int | None:
    t = token.strip()
    if t in _DIGITS:
        try:
            return int(t)
        except ValueError:
            return None
    return None


def _post(url: str, payload: dict, headers: dict, timeout: int, retries: int = 2) -> dict:
    """Backwards-compatible shim: the shared post_json owns retry/backoff now."""
    return post_json(url, headers=headers, payload=payload, timeout=timeout, max_retries=retries)


def extract_candidate_scores(logprobs_content: list[dict], n_candidates: int) -> dict[int, float] | None:
    """Find the decision position and return {candidate_index: logprob}.

    The reference implementation hardcodes `content[-2]`, which silently breaks
    across chat templates (reasoning models append extra tokens, some emit a
    leading space, some omit the trailing stop token). We instead scan every
    returned position and take the first one whose top_logprobs resolve to at
    least two distinct in-range candidate numbers.
    """
    best: dict[int, float] | None = None
    for pos in logprobs_content:
        top = pos.get("top_logprobs") or []
        found: dict[int, float] = {}
        for entry in top:
            idx = _digit_value(entry.get("token", ""))
            if idx is not None and 1 <= idx <= n_candidates:
                lp = entry.get("logprob")
                if lp is not None:
                    found[idx] = max(found.get(idx, -math.inf), float(lp))
        if len(found) >= 2:
            best = found
            break
        if best is None and len(found) == 1:
            best = found
    return best


class OpenAICompatProvider(Provider):
    def __init__(self, spec):
        super().__init__(spec)
        if not spec.base_url:
            raise ProviderError(f"provider '{spec.label}' needs a base_url")
        self.base = spec.base_url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base = self.base + "/v1"

    # -- request building -------------------------------------------------
    def _headers(self) -> dict:
        h = {"Authorization": f"Bearer {self.spec.api_key}"} if self.spec.api_key else {}
        h.update(self.spec.extra_headers or {})
        return h

    def _chat(self, messages: list[dict], *, want_logprobs: bool, max_tokens: int | None = None,
              temperature: float | None = None) -> dict:
        payload = {
            "model": self.spec.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.spec.max_tokens,
            "temperature": self.spec.temperature if temperature is None else temperature,
            "stream": False,
        }
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 20
        return _post(f"{self.base}/chat/completions", payload, self._headers(), self.spec.timeout_s)

    @staticmethod
    def _usage(data: dict) -> Usage:
        u = data.get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        return Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            cached_tokens=int(details.get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0),
        )

    @staticmethod
    def _message(data: dict) -> tuple[str, str]:
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        return (msg.get("content") or "").strip(), reasoning

    # -- the three decode tiers -------------------------------------------
    def choose(self, system: str, user: str, candidates: list[str]) -> Choice:
        mode = self.spec.decode_mode or self._probe_mode(system, user, candidates)
        t0 = time.time()
        if mode == DecodeMode.TOKEN:
            return self._choose_token(system, user, candidates, t0)
        if mode == DecodeMode.N_SAMPLE:
            return self._choose_nsample(system, user, candidates, t0)
        return self._choose_verbalized(system, user, candidates, t0)

    def _probe_mode(self, system: str, user: str, candidates: list[str]) -> DecodeMode:
        from .capability import probe_capabilities

        caps = probe_capabilities(self.spec)
        return caps.mode

    def text(self, prompt: str) -> tuple[str, Usage]:
        data = self._chat([{"role": "user", "content": prompt}], want_logprobs=False, max_tokens=600)
        content, _ = self._message(data)
        return content, self._usage(data)

    def _choose_token(self, system: str, user: str, candidates: list[str], t0: float) -> Choice:
        data = self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            want_logprobs=True,
        )
        text, reasoning = self._message(data)
        content = ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
        scores = extract_candidate_scores(content, len(candidates)) if content else None
        if not scores:
            # Endpoint claimed logprobs but gave nothing usable at the decision point.
            return self._choose_nsample(system, user, candidates, t0)
        z: dict[str, float] = {}
        for idx, action in enumerate(candidates, start=1):
            lp = scores.get(idx)
            z[action] = (math.exp(lp) if self.spec.score_space == ScoreSpace.PROB else lp) if lp is not None else float("-inf")
        return Choice(
            z=z, raw_text=text, reasoning=reasoning or None, mode=DecodeMode.TOKEN,
            score_space=self.spec.score_space, usage=self._usage(data),
            latency_ms=int((time.time() - t0) * 1000),
        )

    def _choose_nsample(self, system: str, user: str, candidates: list[str], t0: float, k: int = 5) -> Choice:
        counts: dict[int, int] = {}
        texts: list[str] = []
        usage = Usage(calls=0)
        for i in range(k):
            data = self._chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                want_logprobs=False, max_tokens=3, temperature=0.8 if i else 0.0,
            )
            text, _ = self._message(data)
            texts.append(text)
            usage = usage + self._usage(data)
            idx = _parse_index(text, len(candidates))
            if idx:
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return self._choose_verbalized(system, user, candidates, t0)
        z = {}
        for idx, action in enumerate(candidates, start=1):
            z[action] = counts.get(idx, 0) / k
        return Choice(
            z=z, raw_text=" ".join(texts)[:200], mode=DecodeMode.N_SAMPLE,
            score_space=ScoreSpace.PROB, usage=usage, latency_ms=int((time.time() - t0) * 1000),
            notes=f"samples={k} votes={counts}",
        )

    def _choose_verbalized(self, system: str, user: str, candidates: list[str], t0: float) -> Choice:
        grading = (
            user
            + "\n\nAfter choosing, output a JSON object on the LAST line: "
            + '{"choice": <number>, "confidence": <0-100>, "scores": {<option number>: <0-100>, ...}}'
        )
        data = self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": grading}],
            want_logprobs=False, max_tokens=160,
        )
        text, reasoning = self._message(data)
        parsed = _parse_scores(text, len(candidates))
        if not parsed:
            parsed = {i: 50.0 for i in range(1, len(candidates) + 1)}
        z = {}
        for idx, action in enumerate(candidates, start=1):
            raw = parsed.get(idx, 1.0)
            z[action] = math.log(max(raw, 1.0) / 100.0) + max(raw, 1.0) / 100.0
        return Choice(
            z=z, raw_text=text, reasoning=reasoning or None, mode=DecodeMode.VERBALIZED,
            score_space=ScoreSpace.PROB, usage=self._usage(data),
            latency_ms=int((time.time() - t0) * 1000),
        )


def _parse_index(text: str, n: int) -> int | None:
    digits = "".join(ch for ch in text if ch.isdigit())
    for size in (2, 1):
        if len(digits) >= size:
            try:
                value = int(digits[:size])
            except ValueError:
                continue
            if 1 <= value <= n:
                return value
    return None


def _coerce_scores(source: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in source.items():
        try:
            idx, val = int(str(key).strip()), float(value)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= 100 and 0 <= val <= 100:
            out[idx] = val
    return out


def _parse_scores(text: str, n: int) -> dict[int, float]:
    import re

    for obj in _balanced_objects(text):
        scores = obj.get("scores")
        if isinstance(scores, dict):
            parsed = {k: v for k, v in _coerce_scores(scores).items() if k <= n}
            if parsed:
                return parsed
        parsed = {k: v for k, v in _coerce_scores(obj).items() if k <= n}
        if parsed:
            return parsed

    out: dict[int, float] = {}
    for match in re.finditer(r'(\d{1,2})\s*"?\s*[:=]\s*"?(\d{1,3})\b', text):
        idx, value = int(match.group(1)), float(match.group(2))
        if 1 <= idx <= n and 0 <= value <= 100:
            out[idx] = value
    return out
