"""Ask an endpoint what it can actually do, then pick the cheapest decode tier.

The probe is what makes "connect any model" honest instead of aspirational:
one endpoint gives top_logprobs (JitRL as published), another only the sampled
token's logprob (needs vote sampling), a third gives nothing (verbalised grading).

The result is cached with a TTL: /api/state probes every configured provider on
every poll, and one unreachable endpoint used to add its full timeout to each
poll. Within the TTL the cached answer is returned with `cached: true` and no
HTTP request is made.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from dataclasses import dataclass, field

from .base import DecodeMode, ProviderSpec, ScoreSpace
from .http import ProviderHTTPStatus, post_json

PROBE_SYSTEM = "You answer with exactly one digit and nothing else."
PROBE_USER = (
    "Pick the option that is a browsing action.\n"
    "1) click the product link\n"
    "2) stop the web server\n"
    "Answer with only 1 or 2."
)

# Override with AGENTD_STATE_PROBE_TTL (seconds); 0 disables caching.
PROBE_TTL_S = int(os.environ.get("AGENTD_STATE_PROBE_TTL", "60") or "60")

# (kind, base_url, model, label, api_key digest) -> (Capabilities, monotonic ts)
_CACHE: dict[tuple, tuple["Capabilities", float]] = {}


@dataclass
class Capabilities:
    model: str
    label: str
    reachable: bool = False
    supports_logprobs: bool = False
    supports_top_logprobs: bool = False
    mode: DecodeMode = DecodeMode.VERBALIZED
    notes: str = ""
    latency_ms: int = 0
    cached: bool = False
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label, "model": self.model, "reachable": self.reachable,
            "supports_logprobs": self.supports_logprobs,
            "supports_top_logprobs": self.supports_top_logprobs,
            "decode_mode": self.mode.value, "notes": self.notes,
            "latency_ms": self.latency_ms, "cached": self.cached,
        }


def _url(spec: ProviderSpec) -> str:
    base = spec.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return f"{base}/chat/completions"


def _headers(spec: ProviderSpec) -> dict:
    h = {"Content-Type": "application/json"}
    if spec.api_key:
        h["Authorization"] = f"Bearer {spec.api_key}"
    h.update(spec.extra_headers or {})
    return h


def _cache_key(spec: ProviderSpec) -> tuple:
    # The key must include the protocol AND the credentials: two specs can
    # share a model name while speaking different wire formats or holding
    # different keys, and a stale hit would report one endpoint's capabilities
    # for the other's. The key is hashed, never stored in clear.
    return (spec.kind, spec.base_url, spec.model, spec.label,
            hashlib.sha256((spec.api_key or "").encode()).hexdigest()[:16])


def _cached_copy(key: tuple) -> "Capabilities | None":
    entry = _CACHE.get(key)
    if entry is None:
        return None
    caps, probed_at = entry
    if PROBE_TTL_S <= 0 or (time.monotonic() - probed_at) >= PROBE_TTL_S:
        return None
    # A copy, not the shared instance: callers mutate capabilities (forced
    # decode modes) and a shared object would leak that between callers.
    return dataclasses.replace(caps, cached=True)


def probe_capabilities(spec: ProviderSpec, use_cache: bool = True, timeout: int = 30) -> Capabilities:
    key = _cache_key(spec)
    if use_cache:
        hit = _cached_copy(key)
        if hit is not None:
            return hit

    if spec.kind != "openai_compat":
        # The probe below is an OpenAI-shaped request (logprobs + top_logprobs
        # against /v1/chat/completions). Sending it to Anthropic Messages or
        # Gemini generateContent produced "unknown url type: '/v1/chat/completions'"
        # because native providers carry no base_url. Those protocols have no
        # logprobs to discover, so the tier is known a priori.
        caps = Capabilities(model=spec.model, label=spec.label)
        caps.decode_mode = DecodeMode.VERBALIZED
        caps.notes = f"kind={spec.kind}: native protocol exposes no logprobs; tier is fixed"
        _CACHE[key] = (caps, time.monotonic())
        return caps

    caps = Capabilities(model=spec.model, label=spec.label)
    payload = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": PROBE_SYSTEM},
            {"role": "user", "content": PROBE_USER},
        ],
        "max_tokens": 2,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    t0 = time.time()
    try:
        # max_retries=0: this probe runs on every state poll, so a slow
        # endpoint must not multiply its latency by a retry ladder.
        data = post_json(_url(spec), headers=_headers(spec), payload=payload,
                         timeout=timeout, max_retries=0)
    except ProviderHTTPStatus as exc:
        caps.notes = f"HTTP {exc.code}: {exc.detail[:200]}"
    except Exception as exc:
        caps.notes = f"{type(exc).__name__}: {exc}"
    else:
        caps.reachable = True
        caps.raw = {"usage": data.get("usage")}
        from .openai_compat import extract_candidate_scores

        content = ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
        caps.supports_logprobs = bool(content)
        caps.supports_top_logprobs = bool(
            content and any((c.get("top_logprobs") or []) for c in content)
        )
        scores = extract_candidate_scores(content, 2) or {}
        if caps.supports_top_logprobs and len(scores) >= 2:
            caps.mode = DecodeMode.TOKEN
            caps.notes = f"top_logprobs at decision position resolved {len(scores)} candidates"
        elif content:
            caps.mode = DecodeMode.N_SAMPLE
            caps.notes = "logprobs present but no candidate mass; using k-sample voting"
        else:
            caps.mode = DecodeMode.VERBALIZED
            caps.notes = "endpoint returned no logprobs; falling back to verbalised grading"
    caps.latency_ms = int((time.time() - t0) * 1000)
    if spec.decode_mode:
        caps.mode = spec.decode_mode
        caps.notes += f" | forced by config -> {caps.mode.value}"
    _CACHE[key] = (caps, time.monotonic())
    return caps


def clear_cache() -> None:
    _CACHE.clear()
