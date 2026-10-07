"""Runtime: the long-lived object the HTTP API and the MCP server both hang on."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .context import ContextBuilder, Ledger
from .council import Council, TriggerPolicy
from .envs.ops_tasks import BUILTIN_TASKS
from .envs.ssh_env import HostSpec, SSHHub
from .kernel import JitRLKernel, Store
from .loop import AgentLoop, LoopConfig
from .providers import (
    NATIVE_PROVIDERS,
    DecodeMode,
    OpenAICompatProvider,
    ProviderSpec,
    probe_capabilities,
)
from .safety.audit import AuditLog
from .toolbus import ToolBus, load_plugin_dir, load_skill_dir, register_builtins, register_mcp_server
from .toolbus.mcp_client import MCPServerStdio


@dataclass
class Session:
    id: str
    task: str
    loop: AgentLoop
    status: str = "queued"
    report: dict | None = None
    events: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    pending_approvals: dict = field(default_factory=dict)


class Runtime:
    def __init__(self, config: dict | None = None, db_path: str | None = None):
        self.config = config or load_config()
        self.db_path = os.path.expanduser(db_path or self.config.get("db", "~/.config/agentd/memory.db"))
        self.store = Store(self.db_path)
        self.audit = AuditLog(Path(self.db_path).parent / "audit.jsonl")
        self.bus = ToolBus()
        register_builtins(self.bus)
        self.sessions: dict[str, Session] = {}
        self.subscribers: list[asyncio.Queue] = []
        self._mcp_servers: dict[str, MCPServerStdio] = {}
        from .viewer import TokenVault

        self.vnc_tokens = TokenVault()
        self.ssh = SSHHub(audit=self.audit, approver=self._ask_human)
        from .envs.pty_env import PTYHub

        # Interactive shells are a separate capability from batch exec, with
        # their own admission check and their own reaper.
        self.pty = PTYHub(self.ssh, audit=self.audit)
        # One ScreenCapture for the process: change detection is stateful, and a
        # per-request capture would never see its own previous frame.
        from .screen import Perceiver, ScreenActions, ScreenCapture

        self.screen = (ScreenCapture(), Perceiver(actions=None))
        self.screen_actions = ScreenActions(audit=self.audit)
        self.screen[1].actions = self.screen_actions
        for host in self.config.get("ssh_hosts", []):
            self.ssh.add_host(HostSpec(**host))
        self.providers = self._build_providers()
        self.kernel = self._build_kernel(enabled=True)
        self.extensions = self._load_extensions()

    # -- wiring -----------------------------------------------------------
    def _build_providers(self) -> dict[str, object]:
        out = {}
        for name, spec in (self.config.get("providers") or {}).items():
            merged = {"label": name, **spec}
            merged["api_key"] = os.environ.get(merged.pop("api_key_env", ""), "") \
                if "api_key_env" in spec else spec.get("api_key", "")
            if "decode_mode" in merged and isinstance(merged["decode_mode"], str):
                merged["decode_mode"] = DecodeMode(merged["decode_mode"])
            kind = merged.get("kind") or "openai_compat"
            if kind in NATIVE_PROVIDERS:
                # Native protocols carry no logprobs; the class pins its own tier.
                merged["decode_mode"] = DecodeMode.VERBALIZED
                out[name] = NATIVE_PROVIDERS[kind](ProviderSpec(**merged))
            elif kind != "openai_compat":
                raise ValueError(
                    f"provider '{name}': unknown kind '{kind}'. "
                    f"known: openai_compat, {', '.join(sorted(NATIVE_PROVIDERS))}"
                )
            else:
                out[name] = OpenAICompatProvider(ProviderSpec(**merged))
        return out

    def default_provider(self, name: str = ""):
        if name == "demo":
            from .bench import SyntheticModel, SYNTHETIC_WEIGHTS

            return SyntheticModel(SYNTHETIC_WEIGHTS, noise=0.22, seed=7)
        if not self.providers:
            raise RuntimeError(
                "no model configured. Add one under \"providers\" in agentd.json, or set "
                "AGENTD_BASE_URL / AGENTD_MODEL / AGENTD_API_KEY. Use provider=\"demo\" for the "
                "built-in synthetic model (no key needed, cannot do real work)."
            )
        if name:
            if name not in self.providers:
                raise KeyError(f"unknown provider '{name}'. configured: {', '.join(self.providers)}")
            return self.providers[name]
        return self.providers[self.config.get("default_provider") or next(iter(self.providers))]

    def _build_kernel(self, enabled: bool) -> JitRLKernel:
        cfg = self.config.get("kernel", {})
        return JitRLKernel(store=self.store, beta=float(cfg.get("beta", 1.0)),
                           gamma=float(cfg.get("gamma", 0.95)), top_k=int(cfg.get("top_k", 8)),
                           ngram=int(cfg.get("ngram", 2)), min_sim=float(cfg.get("min_sim", 0.02)),
                           exploration_prob=float(cfg.get("epsilon", 0.05)),
                           seed=cfg.get("seed"), enabled=enabled)

    def _load_extensions(self) -> dict:
        report = {"plugins": {}, "skills": {}, "mcp": {}}
        plugins_dir = self.config.get("plugins_dir")
        if plugins_dir:
            report["plugins"] = load_plugin_dir(self.bus, plugins_dir)
        skills_dir = self.config.get("skills_dir")
        if skills_dir:
            report["skills"] = load_skill_dir(self.bus, skills_dir)
        for name, spec in (self.config.get("mcp_servers") or {}).items():
            server = MCPServerStdio(name, spec["command"], spec.get("args", []), spec.get("env", {}))
            result = register_mcp_server(self.bus, server)
            if not result["error"]:
                self._mcp_servers[name] = server
            report["mcp"][name] = result
        return report

    # -- approvals --------------------------------------------------------
    def _ask_human(self, payload: dict) -> bool:
        """Blocking approval request surfaced to connected clients."""
        if os.environ.get("AGENTD_AUTO_APPROVE") == "1":
            self.publish({"type": "approval", "auto": True, **payload})
            return True
        token = uuid.uuid4().hex[:12]
        waiter: asyncio.Queue = asyncio.Queue()
        record = {"token": token, "payload": payload, "waiter": waiter, "decided": None}
        session = self._current_session()
        if session:
            session.pending_approvals[token] = record
        self.publish({"type": "approval_required", "token": token, **payload})
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        return _wait_blocking(loop, waiter, timeout=int(os.environ.get("AGENTD_APPROVAL_TIMEOUT", "300")))

    def resolve_approval(self, token: str, approved: bool) -> bool:
        for session in self.sessions.values():
            record = session.pending_approvals.get(token)
            if record:
                record["waiter"].put_nowait(approved)
                record["decided"] = approved
                self.publish({"type": "approval_resolved", "token": token, "approved": approved})
                return True
        return False

    _current = None

    def _current_session(self):
        return self._current

    # -- sessions ---------------------------------------------------------
    def create_session(self, task_name: str, *, provider: str = "", learning: bool = True,
                       max_steps: int = 12, council: bool = False) -> Session:
        factory = BUILTIN_TASKS.get(task_name)
        if factory is None:
            raise KeyError(f"unknown task '{task_name}'. available: {', '.join(sorted(BUILTIN_TASKS))}")
        model = self.default_provider(provider)
        kernel = self._build_kernel(learning)
        members = [model] + ([p for p in self.providers.values() if p is not model][:2]
                             if council and len(self.providers) > 1 else [])
        loop = AgentLoop(
            kernel, model, self.bus, context=ContextBuilder(), ledger=Ledger(),
            council=Council(kernel, members, trigger=TriggerPolicy()) if council and len(members) > 1 else None,
            config=LoopConfig(max_steps=max_steps, deliberate=bool(council and len(members) > 1)),
            approver=self._ask_human,
        )
        session = Session(id=uuid.uuid4().hex[:10], task=task_name, loop=loop)
        self.sessions[session.id] = session
        return session

    def run_session(self, session: Session) -> dict:
        import threading

        self._current = session
        session.status = "running"
        self.publish({"type": "session_started", "session": session.id, "task": session.task})
        try:
            report = session.loop.run(BUILTIN_TASKS[session.task]())
            session.report = report.as_dict()
            session.status = "success" if report.success else "failed"
        except Exception as exc:
            session.status = "error"
            session.report = {"error": f"{type(exc).__name__}: {exc}"}
            self.publish({"type": "session_error", "session": session.id, "error": str(exc)})
        self._current = None
        self.publish({"type": "session_finished", "session": session.id, "status": session.status,
                      "report": session.report})
        return session.report or {}

    def publish(self, event: dict) -> None:
        event["ts"] = time.time()
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    # -- snapshot for the UI ---------------------------------------------
    def state(self) -> dict:
        caps = {}
        for name, provider in self.providers.items():
            try:
                caps[name] = probe_capabilities(provider.spec, timeout=8).as_dict()
            except Exception as exc:
                caps[name] = {"label": name, "reachable": False, "notes": str(exc)[:200]}
        return {
            "providers": {name: {"model": p.spec.model, "base_url": p.spec.base_url, "tier": p.spec.tier}
                          for name, p in self.providers.items()},
            "capabilities": caps,
            "kernel": self.kernel.stats(),
            "tools": {"count": len(self.bus.names()), "by_source": self.bus.by_source()},
            "hosts": self.ssh.hosts(),
            "tasks": sorted(BUILTIN_TASKS),
            "extensions": self.extensions,
            "sessions": [{"id": s.id, "task": s.task, "status": s.status} for s in self.sessions.values()],
            "db": self.db_path,
        }

    def close(self) -> None:
        for server in self._mcp_servers.values():
            server.stop()
        self.ssh.close()


def _wait_blocking(loop, waiter: asyncio.Queue, timeout: int) -> bool:
    """Block the worker thread until a client resolves the approval (or it times out)."""
    async def _wait():
        try:
            return await asyncio.wait_for(waiter.get(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return False

    future = asyncio.run_coroutine_threadsafe(_wait(), loop)
    try:
        return bool(future.result(timeout=timeout + 5))
    except Exception:
        return False


def config_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("AGENTD_CONFIG", "~/.config/agentd/agentd.json")).expanduser()


def load_config(explicit: str | None = None) -> dict:
    path = config_path(explicit)
    cfg: dict = {}
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))

    base_url = os.environ.get("AGENTD_BASE_URL")
    if base_url:
        cfg.setdefault("providers", {})
        name = os.environ.get("AGENTD_MODEL", "default")
        cfg["providers"].setdefault(name, {
            "base_url": base_url, "model": name, "api_key_env": "AGENTD_API_KEY",
            "tier": os.environ.get("AGENTD_TIER", "strong"),
        })
        cfg.setdefault("default_provider", name)
    cfg.setdefault("kernel", {})
    if os.environ.get("AGENTD_BETA"):
        cfg["kernel"]["beta"] = float(os.environ["AGENTD_BETA"])
    if os.environ.get("AGENTD_GAMMA"):
        cfg["kernel"]["gamma"] = float(os.environ["AGENTD_GAMMA"])
    if os.environ.get("AGENTD_EPSILON"):
        cfg["kernel"]["epsilon"] = float(os.environ["AGENTD_EPSILON"])
    return cfg
