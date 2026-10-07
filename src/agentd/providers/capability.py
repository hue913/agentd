"""Ask an endpoint what it can actually do, then pick the cheapest decode tier.

The probe is what makes "connect any model" honest instead of aspirational:
one endpoint gives top_logprobs (JitRL as published), another only the sampled
token's logprob (needs vote sampling), a third gives nothing (verbalised grading).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .base import DecodeMode, ProviderSpec, ScoreSpace

PROBE_SYSTEM = "You answer with exactly one digit and nothing else."
PROBE_USER = (
    "Pick the option that is a browsing action.\n"
    "1) click the product link\n"
    "2) stop the web server\n"
    "Answer with only 1 or 2."
)

_CACHE: dict[tuple[str, str], "Capabilities"] = {}


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
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label, "model": self.model, "reachable": self.reachable,
            "supports_logprobs": self.supports_logprobs,
            "supports_top_logprobs": self.supports_top_logprobs,
            "decode_mode": self.mode.value, "notes": self.notes,
            "latency_ms": self.latency_ms,
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


def probe_capabilities(spec: ProviderSpec, use_cache: bool = True, timeout: int = 30) -> Capabilities:
    # The cache key must include the protocol. Two specs can share a model name
    # while speaking different wire formats, and a stale hit would report one
    # endpoint's capabilities for the other's.
    key = (spec.kind, spec.base_url, spec.model)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    if spec.kind != "openai_compat":
        # The probe below is an OpenAI-shaped request (logprobs + top_logprobs
        # against /v1/chat/completions). Sending it to Anthropic Messages or
        # Gemini generateContent produced "unknown url type: '/v1/chat/completions'"
        # because native providers carry no base_url. Those protocols have no
        # logprobs to discover, so the tier is known a priori.
        caps = Capabilities(model=spec.model, label=spec.label)
        caps.decode_mode = DecodeMode.VERBALIZED
        caps.notes = f"kind={spec.kind}: native protocol exposes no logprobs; tier is fixed"
        _CACHE[key] = caps
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
        req = urllib.request.Request(_url(spec), data=json.dumps(payload).encode(), headers=_headers(spec))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as exc:
        caps.notes = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:200]}"
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
    _CACHE[key] = caps
    return caps


def clear_cache() -> None:
    _CACHE.clear()
