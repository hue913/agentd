"""Tool specification + argument validation shared by all four tool sources."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

RISK_READ = "read"
RISK_WRITE = "write"
RISK_DANGEROUS = "dangerous"

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    data: dict = field(default_factory=dict)
    error: str | None = None
    risk: str = RISK_READ
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "output": self.output, "data": self.data, "error": self.error,
            "risk": self.risk, "duration_ms": self.duration_ms,
        }


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    source: str = "builtin"          # builtin | plugin | skill | mcp
    risk: str = RISK_READ
    handler: Callable[..., ToolResult] | None = None
    hidden: bool = False             # excluded from the prompt catalog until expanded
    examples: list[str] = field(default_factory=list)

    @property
    def namespace(self) -> str:
        return self.name.split(".", 1)[0]

    def schema_tokens(self) -> int:
        """Rough token cost of putting this tool in the prompt (4 chars/token)."""
        return len(self.name + self.description + json.dumps(self.parameters, ensure_ascii=False)) // 4

    def signature(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


def validate_args(spec: ToolSpec, args: dict) -> tuple[dict, list[str]]:
    """Fill defaults, then report problems instead of throwing at the model.

    A validation error is returned as text the model can read and fix; raising
    would just abort the episode and teach the kernel nothing.
    """
    problems: list[str] = []
    schema = spec.parameters or {}
    props: dict = schema.get("properties") or {}
    required: list = schema.get("required") or []
    clean = dict(args or {})

    for key in required:
        if key not in clean or clean[key] in (None, ""):
            problems.append(f"missing required argument '{key}'")

    for key, value in list(clean.items()):
        if key not in props:
            problems.append(f"unknown argument '{key}' (allowed: {', '.join(props) or 'none'})")
            continue
        expected = props[key].get("type")
        checker = _TYPE_CHECKS.get(expected if isinstance(expected, str) else "")
        if checker and not checker(value):
            problems.append(f"argument '{key}' must be {expected}, got {type(value).__name__}")

    for key, prop in props.items():
        if key not in clean and "default" in prop:
            clean[key] = prop["default"]

    return clean, problems
