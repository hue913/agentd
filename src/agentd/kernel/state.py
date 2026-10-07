"""State/action normalisation and similarity features shared by the kernel."""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
# Element ids like [1234] / ref=5 change every run and must not drive retrieval.
_VOLATILE_ID_RE = re.compile(r"[\[\(]?\b(?:ref|id|node|target)\s*=?\s*\d+[\]\)]?", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")


def normalize_state(text: str) -> str:
    return _WS_RE.sub(" ", _VOLATILE_ID_RE.sub(" ", _URL_RE.sub(" url", text or ""))).strip().lower()


def normalize_action(text: str) -> str:
    """Collapse whitespace but keep the action verb and its argument shape."""
    return _WS_RE.sub(" ", (text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_state(text))


def ngrams(tokens: list[str], n: int = 2) -> set[str]:
    if not tokens:
        return set()
    grams: set[str] = set()
    for size in range(1, max(1, n) + 1):
        if size == 1:
            grams.update(tokens)
        else:
            grams.update(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))
    return grams


def fingerprint(text: str) -> str:
    return hashlib.sha1(normalize_state(text).encode("utf-8", "ignore")).hexdigest()[:16]
