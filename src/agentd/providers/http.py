"""Shared JSON-over-HTTP POST plumbing for the provider layer (stdlib only).

Three providers used to hand-roll the same urllib dance with slightly
different retry, backoff and error semantics. One function now owns it, so a
bug fixed here (e.g. the missing jitter that made every retry thunder at
exactly the same moment) is fixed everywhere.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request

from .base import ProviderError

# 1.5s起步的指数退避, plus jitter. 4xx-class errors that a retry cannot fix
# (bad key, bad URL, malformed request) raise immediately instead of burning
# the retry budget against a deterministic failure.
BACKOFF_BASE_S = 1.5
BACKOFF_CAP_S = 8.0
BACKOFF_JITTER_S = 0.5
NO_RETRY_STATUS = (400, 401, 403, 404)


class ProviderHTTPStatus(ProviderError):
    """HTTP failure carrying the status code as data.

    Callers that must sanitize the message (upstream bodies can echo the
    prompt or the API key) catch this subclass instead of parsing the text.
    """

    def __init__(self, code: int, url: str, detail: str):
        self.code = code
        self.url = url
        self.detail = detail
        super().__init__(f"HTTP {code} from {url}: {detail}")


def post_json(url: str, *, headers: dict, payload: dict, timeout: float,
              max_retries: int = 2) -> dict:
    """POST a JSON object and decode the JSON response.

    Retries transient failures (timeouts, DNS, connection resets, 5xx) with
    exponential backoff starting at BACKOFF_BASE_S plus jitter. Non-JSON
    bodies on a 200 raise ProviderError: every caller assumes dict, and a
    silent string would only move the crash somewhere harder to diagnose.
    """
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url, data=body, headers={**headers, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "ignore")
            return json.loads(raw)
        except ValueError as exc:
            raise ProviderError(f"non-JSON response from {url}") from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            last = ProviderHTTPStatus(exc.code, url, detail)
            if exc.code in NO_RETRY_STATUS:
                raise last from exc
        except Exception as exc:  # timeouts, DNS, connection resets
            last = exc
        if attempt < max_retries:
            backoff = min(BACKOFF_BASE_S * (2 ** attempt), BACKOFF_CAP_S)
            time.sleep(backoff + random.uniform(0.0, BACKOFF_JITTER_S))
    raise ProviderError(f"request failed after {max_retries + 1} attempts: {last}")
