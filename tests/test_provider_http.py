"""Provider layer against a fake OpenAI-compatible endpoint.

Guards the two claims that matter: (1) the decision token is located by scanning
the returned positions, not by the paper's hardcoded content[-2]; (2) an endpoint
that cannot supply logprobs still produces a decision instead of crashing.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentd.providers import DecodeMode, OpenAICompatProvider, ProviderSpec, clear_cache, probe_capabilities
from agentd.providers.openai_compat import extract_candidate_scores

STATE = {"behavior": "token", "votes": [], "requests": []}


def _logprobs_payload(digit_first: bool) -> dict:
    decision = {
        "token": "1",
        "logprob": math.log(0.2),
        "top_logprobs": [
            {"token": "1", "logprob": math.log(0.2)},
            {"token": "2", "logprob": math.log(0.75)},
        ],
    }
    filler = [
        {"token": "x", "logprob": -0.1, "top_logprobs": [{"token": "x", "logprob": -0.1}]},
        {"token": "y", "logprob": -0.2, "top_logprobs": [{"token": "y", "logprob": -0.2}]},
    ]
    content = [decision] + filler if digit_first else filler + [decision]
    return {
        "choices": [{"message": {"role": "assistant", "content": "1"}, "logprobs": {"content": content}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 32}},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        STATE["requests"].append(body)
        behavior = STATE["behavior"]

        if behavior == "token":
            payload = _logprobs_payload(digit_first=True)
        elif behavior == "token_at_minus2":
            payload = _logprobs_payload(digit_first=False)
        elif behavior == "logprobs_without_mass":
            payload = {
                "choices": [{
                    "message": {"role": "assistant", "content": "1"},
                    "logprobs": {"content": [{"token": "one", "logprob": -0.5, "top_logprobs": []}]},
                }],
                "usage": {"prompt_tokens": 40, "completion_tokens": 2},
            }
        elif behavior == "no_logprobs":
            answer = STATE["votes"][min(len(STATE["requests"]) - 1, len(STATE["votes"]) - 1)]
            if answer.startswith("{"):
                payload = {
                    "choices": [{"message": {"role": "assistant", "content": answer}}],
                    "usage": {"prompt_tokens": 400, "completion_tokens": 60},
                }
            else:
                payload = {
                    "choices": [{"message": {"role": "assistant", "content": answer}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1},
                }
        else:
            self.send_error(500, "unknown behavior")
            return

        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def server():
    STATE.update(behavior="token", votes=[], requests=[])
    clear_cache()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    clear_cache()


def spec(url, **kw) -> ProviderSpec:
    defaults = dict(label="t", model="m", base_url=url, api_key="k", max_tokens=4)
    defaults.update(kw)
    return ProviderSpec(**defaults)


CANDIDATES = ["click add to cart", "restart nginx"]


def test_token_mode_scores_candidates_and_reports_cache_hits(server):
    STATE["behavior"] = "token"
    provider = OpenAICompatProvider(spec(server, decode_mode=DecodeMode.TOKEN))
    choice = provider.choose("sys", "user state", CANDIDATES)
    assert choice.mode == DecodeMode.TOKEN
    assert choice.z[CANDIDATES[1]] == pytest.approx(0.75, abs=1e-6)
    assert choice.z[CANDIDATES[0]] == pytest.approx(0.2, abs=1e-6)
    assert choice.usage.prompt_tokens == 40 and choice.usage.cached_tokens == 32
    assert choice.usage.cache_hit_rate == pytest.approx(0.8)
    assert STATE["requests"][0]["top_logprobs"] == 20


def test_decision_token_is_found_even_when_not_at_minus2(server):
    for behavior, digit_first in (("token", True), ("token_at_minus2", False)):
        STATE["behavior"] = behavior
        provider = OpenAICompatProvider(spec(server, decode_mode=DecodeMode.TOKEN))
        choice = provider.choose("sys", "user", CANDIDATES)
        assert choice.z[CANDIDATES[1]] == pytest.approx(0.75, abs=1e-6), behavior


def test_extract_scores_prefers_position_with_two_candidates():
    content = [
        {"token": "The", "logprob": -0.1, "top_logprobs": [{"token": "The", "logprob": -0.1}]},
        {"token": "1", "logprob": -1.6, "top_logprobs": [
            {"token": "1", "logprob": -1.6}, {"token": "2", "logprob": -0.3}]},
        {"token": "\n", "logprob": -0.01, "top_logprobs": [{"token": "\n", "logprob": -0.01}]},
    ]
    assert extract_candidate_scores(content, 2) == {1: pytest.approx(-1.6), 2: pytest.approx(-0.3)}


def test_capability_probe_chooses_token_mode(server):
    STATE["behavior"] = "token"
    caps = probe_capabilities(spec(server))
    assert caps.reachable and caps.supports_top_logprobs
    assert caps.mode == DecodeMode.TOKEN


def test_capability_probe_degrades_to_n_sample(server):
    STATE["behavior"] = "logprobs_without_mass"
    caps = probe_capabilities(spec(server))
    assert caps.reachable and caps.supports_logprobs
    assert caps.mode == DecodeMode.N_SAMPLE


def test_capability_probe_degrades_to_verbalized(server):
    STATE["behavior"] = "no_logprobs"
    STATE["votes"] = ["1"]
    caps = probe_capabilities(spec(server))
    assert caps.reachable and not caps.supports_logprobs
    assert caps.mode == DecodeMode.VERBALIZED


def test_n_sample_falls_back_to_voting(server):
    STATE["behavior"] = "no_logprobs"
    STATE["votes"] = ["2", "2", "1", "2", "2"]
    provider = OpenAICompatProvider(spec(server, decode_mode=DecodeMode.N_SAMPLE))
    choice = provider.choose("sys", "user", CANDIDATES)
    assert choice.mode == DecodeMode.N_SAMPLE
    assert choice.z[CANDIDATES[1]] == pytest.approx(0.8)
    assert choice.z[CANDIDATES[0]] == pytest.approx(0.2)
    assert choice.usage.calls == 5


def test_verbalized_mode_parses_confidences(server):
    STATE["behavior"] = "no_logprobs"
    STATE["votes"] = ['I pick option 2. {"choice": 2, "confidence": 90, "scores": {"1": 10, "2": 90}}']
    provider = OpenAICompatProvider(spec(server, decode_mode=DecodeMode.VERBALIZED))
    choice = provider.choose("sys", "user", CANDIDATES)
    assert choice.mode == DecodeMode.VERBALIZED
    assert choice.z[CANDIDATES[1]] > choice.z[CANDIDATES[0]]


def test_unreachable_endpoint_raises_provider_error():
    from agentd.providers import ProviderError

    provider = OpenAICompatProvider(spec("http://127.0.0.1:1/v1", timeout_s=2))
    with pytest.raises(ProviderError):
        provider.choose("sys", "user", CANDIDATES)
