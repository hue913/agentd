"""Parsing helpers shared across the provider layer.

`balanced_objects` used to live as a private helper in openai_compat while
loop.py and council.py reached into it with private imports. It is a general
"models rarely emit clean JSON" tool, so it now lives in its own module with a
public name.
"""

from __future__ import annotations

import json


def balanced_objects(text: str) -> list[dict]:
    """Return every top-level JSON object found in `text`.

    Models rarely emit clean JSON: they wrap objects in prose, emit several,
    or fence them in code blocks. A brace-depth scan recovers every
    well-formed top-level object without pretending the text was valid JSON.
    """
    out: list[dict] = []
    depth, start = 0, -1
    for i, ch in enumerate(text or ""):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict):
                        out.append(obj)
                    start = -1
    return out
