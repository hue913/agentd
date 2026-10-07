"""Native-protocol providers (Anthropic Messages, Gemini generateContent).

The claims under test, in order of importance:

1. Neither protocol is sent `logprobs`; both are pinned to VERBALIZED. If either
   ever claims a TOKEN tier it would be silently computing JitRL advantages over
   fabricated scores.
2. The wire shape is what the real APIs expect (path, auth header, body keys).
3. Self-reported confidence maps onto a bounded score, and an unparseable reply
   produces a FLAT field rather than an invented ranking.
4. An upstream error body is never relayed to the caller -- upstream bodies can
   echo the prompt or the key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentd.providers import DecodeMode, ProviderSpec
from agentd.providers.base import ProviderError
from agentd.providers.native import (
    AnthropicProvider,
    GeminiProvider,
    NATIVE_PROVIDERS,
    confidence_to_z,
)

STATE = {"reply": "", "requests": [], "headers": [], "path": "", "status": 200}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        STATE["requests"].append(json.loads(self.rfile.read(length) or b"{}"))
        STATE["headers"].append(dict(self.headers))
        STATE["path"] = self.path
        if STATE["status"] != 200:
            body = json.dumps({"error": {"message": "invalid key sk-SECRET"}}).encode()
            self.send_response(STATE["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/v1/messages"):
            payload = {
                "content": [{"type": "text", "text": STATE["reply"]}],
                "usage": {"input_tokens": 120, "output_tokens": 18,
                          "cache_read_input_tokens": 64},
            }
        else:
            payload = {
                "candidates": [{"content": {"parts": [{"text": STATE["reply"]}]}}],
                "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 12},
            }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def endpoint():
    STATE.update(reply="", requests=[], headers=[], path="", status=200)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def make(kind: str, base: str):
    spec = ProviderSpec(label=kind, model="m1", base_url=base, api_key="test-key")
    return NATIVE_PROVIDERS[kind](spec)


CANDS = ["read the log", "restart nginx", "delete /var/log"]


# --- tier pinning -----------------------------------------------------------

def test_both_native_kinds_are_registered():
    assert set(NATIVE_PROVIDERS) == {"anthropic", "gemini"}


@pytest.mark.parametrize("kind", ["anthropic", "gemini"])
def test_choose_never_claims_the_token_tier(endpoint, kind):
    STATE["reply"] = '{"choice": 2, "confidence": 80, "scores": {"1": 10, "2": 90, "3": 5}}'
    choice = make(kind, endpoint).choose("sys", "state", CANDS)
    assert choice.mode is DecodeMode.VERBALIZED
    assert choice.notes and "no logprobs" in choice.notes


@pytest.mark.parametrize("kind", ["anthropic", "gemini"])
def test_no_logprobs_field_is_ever_sent(endpoint, kind):
    STATE["reply"] = '{"choice": 1, "confidence": 70, "scores": {"1": 70, "2": 20, "3": 10}}'
    make(kind, endpoint).choose("sys", "state", CANDS)
    blob = json.dumps(STATE["requests"][0])
    assert "logprobs" not in blob
    assert "top_logprobs" not in blob


# --- wire shape -------------------------------------------------------------

def test_anthropic_request_shape(endpoint):
    STATE["reply"] = '{"choice": 1, "confidence": 60, "scores": {"1": 60, "2": 30, "3": 10}}'
    make("anthropic", endpoint).choose("you are a sysadmin", "state text", CANDS)
    assert STATE["path"] == "/v1/messages"
    body = STATE["requests"][0]
    assert body["model"] == "m1"
    assert body["system"] == "you are a sysadmin"          # system is a top-level field
    assert body["messages"] == [{"role": "user", "content": body["messages"][0]["content"]}]
    assert "system" not in body["messages"][0]
    headers = {k.lower(): v for k, v in STATE["headers"][0].items()}
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"]


def test_gemini_request_shape(endpoint):
    STATE["reply"] = '{"choice": 1, "confidence": 60, "scores": {"1": 60, "2": 30, "3": 10}}'
    make("gemini", endpoint).choose("you are a sysadmin", "state text", CANDS)
    assert STATE["path"] == "/v1beta/models/m1:generateContent"
    body = STATE["requests"][0]
    assert body["contents"][0]["parts"][0]["text"].startswith("state text")
    assert body["systemInstruction"]["parts"][0]["text"] == "you are a sysadmin"
    assert body["generationConfig"]["candidateCount"] == 1
    headers = {k.lower(): v for k, v in STATE["headers"][0].items()}
    assert headers["x-goog-api-key"] == "test-key"


@pytest.mark.parametrize("kind,prompt,completions", [("anthropic", 120, 18), ("gemini", 90, 12)])
def test_usage_is_read_from_each_native_shape(endpoint, kind, prompt, completions):
    STATE["reply"] = '{"choice": 1, "confidence": 60, "scores": {"1": 60, "2": 30, "3": 10}}'
    choice = make(kind, endpoint).choose("s", "u", CANDS)
    assert choice.usage.prompt_tokens == prompt
    assert choice.usage.completion_tokens == completions


# --- scoring ----------------------------------------------------------------

def test_confidence_maps_onto_a_bounded_logprob_like_score():
    assert confidence_to_z(100) == pytest.approx(1.0)
    assert confidence_to_z(50) == pytest.approx(-0.193, abs=0.01)
    assert confidence_to_z(1) < -4.0
    # monotone: more confidence never means a lower score
    values = [confidence_to_z(v) for v in (1, 25, 50, 75, 100)]
    assert values == sorted(values)


def test_scores_are_ordered_by_reported_confidence(endpoint):
    STATE["reply"] = '{"choice": 3, "confidence": 95, "scores": {"1": 5, "2": 10, "3": 95}}'
    choice = make("anthropic", endpoint).choose("s", "u", CANDS)
    assert max(choice.z, key=choice.z.get) == CANDS[2]


@pytest.mark.parametrize("kind", ["anthropic", "gemini"])
def test_unparseable_reply_yields_a_flat_field_not_an_invented_ranking(endpoint, kind):
    STATE["reply"] = "I think option two is probably the safest, but honestly it depends."
    choice = make(kind, endpoint).choose("s", "u", CANDS)
    assert len(set(round(v, 9) for v in choice.z.values())) == 1, \
        "an unreadable reply must not silently produce a preference"


@pytest.mark.parametrize("kind", ["anthropic", "gemini"])
def test_upstream_error_body_is_not_relayed(endpoint, kind):
    STATE["status"] = 401
    with pytest.raises(ProviderError) as exc:
        make(kind, endpoint).choose("s", "u", CANDS)
    assert "sk-SECRET" not in str(exc.value), "upstream bodies can echo the key"
    assert "401" in str(exc.value)


@pytest.mark.parametrize("kind", ["anthropic", "gemini"])
def test_text_channel_works_for_the_council_critique_phase(endpoint, kind):
    STATE["reply"] = "The objection is that step 2 needs a rollback plan."
    text, usage = make(kind, endpoint).text("critique this plan")
    assert "rollback" in text
    assert usage.total_tokens > 0
