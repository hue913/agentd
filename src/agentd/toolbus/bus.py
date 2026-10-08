"""One registry, four sources: builtin tools, plugins, skill packs, MCP servers.

The bus also owns the *token* question: tool schemas are the biggest fixed cost
in an agent prompt, so the catalog is rendered against a budget and the rest is
progressively disclosed on demand.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .spec import RISK_DANGEROUS, ToolResult, ToolSpec, validate_args


def _accepts_ctx(handler) -> bool:
    """Some tools need the call context, most do not — don't force the signature
    on third-party plugins."""
    try:
        params = inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "ctx" in params


# Minimal structural contracts for the collaborators that ride on CallContext.
# They replace bare `object` annotations: tools get honest types without the
# bus importing their concrete modules (SSHHub, AuditLog, JitRLKernel), which
# would drag heavy imports — and a circular one via runtime — into the bus.
@runtime_checkable
class Approver(Protocol):
    """A human-in-the-loop check; True means the request was explicitly allowed."""

    def __call__(self, request: dict) -> bool: ...


@runtime_checkable
class AuditSink(Protocol):
    def write(self, record: object) -> None: ...


@runtime_checkable
class SSHHubLike(Protocol):
    """The slice of SSHHub tool handlers actually depend on."""

    def exec(self, label: str, command: str, **kwargs) -> object: ...

    def hosts(self) -> list[str]: ...


@runtime_checkable
class KernelLike(Protocol):
    """The store-reading slice of the kernel handlers may consult."""

    def stats(self) -> dict: ...


@dataclass
class CallContext:
    session: str = "default"
    episode_id: int | None = None
    approved: bool = False
    approver: Approver | None = None
    ssh: SSHHubLike | None = None
    audit: AuditSink | None = None
    kernel: KernelLike | None = None
    extra: dict = field(default_factory=dict)


class ToolBus:
    def __init__(self, require_approval_for: tuple[str, ...] = (RISK_DANGEROUS,)):
        self._tools: dict[str, ToolSpec] = {}
        self._disabled: set[str] = set()
        self.require_approval_for = tuple(require_approval_for)

    # -- registration -----------------------------------------------------
    def register(self, spec: ToolSpec, *, replace: bool = False) -> None:
        if not spec.name or "." not in spec.name:
            raise ValueError(f"tool name must be namespaced (got {spec.name!r})")
        if spec.handler is None:
            raise ValueError(f"tool {spec.name} has no handler")
        if spec.name in self._tools and not replace:
            raise ValueError(f"tool {spec.name} already registered (pass replace=True)")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def disable(self, name: str) -> None:
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        self._disabled.discard(name)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self, include_hidden: bool = False) -> list[str]:
        return sorted(
            n for n, s in self._tools.items()
            if n not in self._disabled and (include_hidden or not s.hidden)
        )

    def by_source(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, spec in self._tools.items():
            out.setdefault(spec.source, []).append(name)
        return {k: sorted(v) for k, v in sorted(out.items())}

    # -- invocation -------------------------------------------------------
    def call(self, name: str, args: dict, ctx: CallContext | None = None) -> ToolResult:
        ctx = ctx or CallContext()
        spec = self._tools.get(name)
        if spec is None:
            available = ", ".join(self.names()) or "none"
            return ToolResult(ok=False, error=f"unknown tool '{name}'. available: {available}")
        if name in self._disabled:
            return ToolResult(ok=False, error=f"tool '{name}' is disabled in configuration")

        clean, problems = validate_args(spec, args)
        if problems:
            return ToolResult(ok=False, error="argument error: " + "; ".join(problems), risk=spec.risk)

        if spec.risk in self.require_approval_for and not ctx.approved:
            granted = bool(ctx.approver and ctx.approver({"tool": name, "args": clean, "risk": spec.risk}))
            if not granted:
                return ToolResult(
                    ok=False, risk=spec.risk,
                    error=(f"tool '{name}' is {spec.risk} and needs human approval before running; "
                           f"re-request it with approval=true once the user confirms"),
                )

        t0 = time.time()
        kwargs = dict(clean)
        if _accepts_ctx(spec.handler):
            kwargs["ctx"] = ctx
        try:
            result = spec.handler(**kwargs)
        except TypeError as exc:
            return ToolResult(ok=False, risk=spec.risk, error=f"bad call to {name}: {exc}")
        except Exception as exc:
            return ToolResult(ok=False, risk=spec.risk,
                              error=f"{type(exc).__name__} in {name}: {exc}",
                              duration_ms=int((time.time() - t0) * 1000))
        if not isinstance(result, ToolResult):
            result = ToolResult(ok=True, output=str(result))
        result.risk = result.risk or spec.risk
        result.duration_ms = result.duration_ms or int((time.time() - t0) * 1000)
        return result

    # -- prompt rendering (the token thrift switch) -----------------------
    def catalog(self, token_budget: int = 1200, style: str = "json") -> str:
        """Render the tool list within a hard token budget.

        Order of sacrifice: drop examples, then descriptions, then schemas, and
        finally fall back to bare names plus an `expand` hint. Never silently
        exceed the budget — that is how agent bills balloon.
        """
        visible = [self._tools[n] for n in self.names()]
        if not visible:
            return ""

        if style == "json":
            for level in ("full", "brief", "names"):
                text = _render_json(visible, level, self)
                if len(text) // 4 <= token_budget or level == "names":
                    return text

        lines = []
        for spec in visible:
            lines.append(f"{spec.name}: {spec.description[:120]}")
        text = "\n".join(lines)
        if len(text) // 4 > token_budget:
            text = "\n".join(spec.name for spec in visible)
        return text

    def expand(self, name: str) -> dict:
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"unknown tool '{name}'"}
        return {
            "name": spec.name, "description": spec.description, "parameters": spec.parameters,
            "risk": spec.risk, "source": spec.source, "examples": spec.examples,
        }

    def signature_tokens(self) -> int:
        return sum(spec.schema_tokens() for spec in self._tools.values() if not spec.hidden)


def _render_json(visible: list[ToolSpec], level: str, bus: "ToolBus") -> str:
    import json

    out = []
    for spec in visible:
        if level == "names":
            out.append({"name": spec.name})
        elif level == "brief":
            out.append({"name": spec.name, "description": spec.description[:60],
                        "args": list((spec.parameters.get("properties") or {}).keys())})
        else:
            out.append(spec.signature())
    tail = "" if level == "full" else "\n(use agentd.tools.expand <name> before calling for the full schema)"
    return json.dumps(out, ensure_ascii=False, indent=None if level != "full" else 2) + tail
