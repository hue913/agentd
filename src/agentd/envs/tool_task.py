"""Tool task: a real goal, executed for real, through the tool bus.

The scripted tasks in `ops_tasks` exist so the kernel can be *measured* on a
closed action set. This task is the other end of the spectrum: an open goal
whose actions are actual tool calls, executed by the bus against this host or
over SSH -- so the same loop, kernel, council and safety gate that were
benchmarked on toy chains now drive a real machine.

Action grammar (the model proposes these as candidates):

    ssh.exec {"host": "self", "command": "uptime -p"}    explicit JSON args
    ssh.exec uptime -p                                   shorthand for *.exec
    memory.stats                                         zero-argument tools
    finish {"answer": "..."}                             end the episode

Parsing lives in one function (`parse_action`), and a malformed option becomes
an observation the model can correct rather than a crashed episode -- the whole
point is that bad steps are visible and recoverable.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

MAX_OBS_CHARS = 2500
MAX_ANSWER_CHARS = 2000

_TOOL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_.]*)\s*(.*)$", re.DOTALL)


class ActionFormatError(ValueError):
    """The proposed action cannot be read as a tool call."""


def parse_action(text: str) -> tuple[str, str, dict]:
    """Return `("finish", "", {"answer": ...})` or `("call", tool_name, args)`."""
    raw = (text or "").strip()
    if not raw:
        raise ActionFormatError("the action is empty")
    match = _TOOL_RE.match(raw)
    if not match:
        raise ActionFormatError(
            f"cannot read an action from {raw[:80]!r}; expected `tool.name {{json}}` or `finish {{...}}`"
        )
    name, rest = match.group(1), match.group(2).strip()

    if name == "finish":
        return "finish", "", {"answer": _finish_payload(rest)}

    if not rest:
        return "call", name, {}

    if rest.startswith("{"):
        try:
            args = json.loads(rest)
        except ValueError as exc:
            raise ActionFormatError(f"arguments are not valid JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise ActionFormatError("arguments must be a JSON object")
        return "call", name, args

    if name.endswith(".exec"):
        # `ssh.exec df -h` is what a model writes first; accept it instead of
        # bouncing the step back just to add a wrapper.
        return "call", name, {"command": rest}

    raise ActionFormatError(
        f"'{name}' needs JSON arguments, e.g. {name} {{\"arg\": \"value\"}}"
    )


def _finish_payload(rest: str) -> str:
    if not rest:
        return "done"
    if rest.startswith("{"):
        try:
            payload = json.loads(rest)
        except ValueError:
            return rest.strip()[:MAX_ANSWER_CHARS]
        if isinstance(payload, dict):
            return str(payload.get("answer", "")).strip()[:MAX_ANSWER_CHARS] or "done"
        return str(payload)[:MAX_ANSWER_CHARS]
    return rest.strip().strip('"')[:MAX_ANSWER_CHARS]


@dataclass
class ToolTask:
    """An open goal driven by real tool calls. One instance per episode."""

    goal_text: str
    hosts: list[str] = field(default_factory=list)
    default_host: str = ""
    scope: str = "ops"
    name: str = ""
    _state: str = ""
    _steps: int = 0
    _finished: bool = False

    def __post_init__(self) -> None:
        self.hosts = list(self.hosts)
        if not self.default_host and self.hosts:
            self.default_host = self.hosts[0]
        if not self.name:
            slug = re.sub(r"[^a-z0-9]+", "-", self.goal_text.lower()).strip("-")[:40]
            self.name = f"ops-{slug or 'goal'}"

    # -- Task protocol ----------------------------------------------------
    def reset(self) -> str:
        self._steps = 0
        self._finished = False
        self._state = self._initial_observation()
        return self._state

    def observe(self) -> str:
        return self._state or self.reset()

    def goal(self) -> str:
        return self.goal_text

    def candidates(self, state: str) -> list[str]:
        # Empty by contract: an open goal has no enumerable action space, so
        # AgentLoop.enumerate_candidates asks the model to propose options.
        return []

    def apply(self, action: str, ctx=None) -> tuple[str, bool, float]:
        self._steps += 1
        try:
            kind, name, args = parse_action(action)
        except ActionFormatError as exc:
            self._state = f"[{self._steps}] unusable action. {exc}"
            return self._state, False, 0.0

        if kind == "finish":
            self._finished = True
            answer = str(args.get("answer", "")).strip()
            self._state = (f"[{self._steps}] finish after {self._steps - 1} tool call(s).\n"
                           f"Answer: {answer}")
            return self._state, True, 1.0

        bus = (getattr(ctx, "extra", {}) or {}).get("bus") if ctx is not None else None
        if bus is None:
            self._state = "[tool] no tool bus attached to this session; cannot execute"
            return self._state, False, 0.0

        # A host the caller did not pin falls back to the task's default, so
        # `ssh.exec df -h` works without repeating the label on every step.
        if name.startswith("ssh.") and self.default_host and "host" not in args:
            args = {**args, "host": self.default_host}

        t0 = time.time()
        result = bus.call(name, args, ctx)
        duration = int((time.time() - t0) * 1000)
        head = f"[{self._steps}] {name} -> " + ("ok" if result.ok else "FAILED")
        body = (result.output or result.error or "").strip()
        if len(body) > MAX_OBS_CHARS:
            body = body[:MAX_OBS_CHARS // 2] + "\n... [clipped] ...\n" + body[-MAX_OBS_CHARS // 2:]
        self._state = f"{head} ({duration}ms)\n{body}" if body else f"{head} ({duration}ms)"
        return self._state, False, 0.0

    # -- prompt material --------------------------------------------------
    def _initial_observation(self) -> str:
        hosts = ", ".join(self.hosts) if self.hosts else "none configured"
        default = self.default_host or "n/a"
        return (
            f"Goal: {self.goal_text}\n"
            f"SSH hosts: {hosts} (default: {default}). This machine itself is reachable "
            f"via local.exec.\n"
            "Work in single tool calls; verify effects instead of assuming them. "
            "Call `finish` with an answer when the goal is met."
        )

    @property
    def propose_hint(self) -> str:
        default = self.default_host or "none"
        return (
            "Each candidate must be exactly one action in one of these forms:\n"
            '- `tool.name {"arg": "value"}` -- JSON arguments, e.g. '
            '`ssh.exec {"host": "self", "command": "df -h"}`\n'
            "- `ssh.exec <command>` or `local.exec <command>` -- bare-command shorthand\n"
            '- `finish {"answer": "..."}` -- end the episode with what you found\n'
            f"Default SSH host: {default}. Use only tool names from the catalog; "
            "never invent host labels."
        )
