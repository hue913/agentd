"""Runtime: the long-lived object the HTTP API and the MCP server both hang on."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .context import ContextBuilder, Ledger
from .council import Council, TriggerPolicy
from .envs.ops_tasks import BUILTIN_TASKS
from .envs.ssh_env import POLICIES, HostSpec, SSHError, SSHHub
from .fleet import Fleet
from .kernel import JitRLKernel, Store
from .log import get_logger
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
from .templates import templates_from_config

log = get_logger("agentd.runtime")


@dataclass
class Session:
    id: str
    task: str
    loop: AgentLoop
    status: str = "queued"
    report: dict | None = None
    created_at: float = field(default_factory=time.time)
    pending_approvals: dict = field(default_factory=dict)
    task_obj: object | None = None      # the environment this session runs


class Runtime:
    def __init__(self, config: dict | None = None, db_path: str | None = None):
        self.config = config or load_config()
        self.db_path = os.path.expanduser(db_path or self.config.get("db", "~/.config/agentd/memory.db"))
        self.store = Store(self.db_path)
        self.audit = AuditLog(Path(self.db_path).parent / "audit.jsonl")
        self.bus = ToolBus()
        register_builtins(self.bus)
        # OrderedDict so eviction is oldest-first; sessions only ever grow in a
        # long-lived process otherwise.
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.max_sessions = int(self.config.get("max_sessions", 200))
        self.session_ttl_s = int(self.config.get("session_ttl_s", 6 * 3600))
        # Every approval request registers here, keyed by token, so it is
        # resolvable no matter which thread asked. Session copies are only the
        # per-session UI view; without this table an approval raised outside a
        # session worker (e.g. the HTTP ssh.exec route) would hang unresolvable.
        self.pending_approvals: dict[str, dict] = {}
        self.subscribers: list[asyncio.Queue] = []
        self._subscribers_lock = threading.Lock()
        # Captured on the first subscribe (which happens on the server's event
        # loop). Worker threads publish through it; see publish().
        self.main_loop: asyncio.AbstractEventLoop | None = None
        self.dropped_events = 0
        # The session whose agent loop is running *on this thread* — approvals
        # and other context lookups must follow the calling thread, not a
        # process-wide "current" pointer that concurrent sessions overwrite.
        self._local = threading.local()
        self._mcp_servers: dict[str, MCPServerStdio] = {}
        from .viewer import TokenVault

        self.vnc_tokens = TokenVault()
        self.ssh = SSHHub(audit=self.audit, approver=self._ask_human)
        from .envs.pty_env import PTYHub

        # Interactive shells are a separate capability from batch exec, with
        # their own admission check and their own reaper. The reaper runs in
        # the background so dead shells are collected even if nobody opens
        # the PTY panel; close() stops it.
        self.pty = PTYHub(self.ssh, audit=self.audit)
        self.pty.start_reaper()
        # One ScreenCapture for the process: change detection is stateful, and a
        # per-request capture would never see its own previous frame.
        from .screen import Perceiver, ScreenActions, ScreenCapture

        self.screen = (ScreenCapture(), Perceiver(actions=None))
        self.screen_actions = ScreenActions(audit=self.audit)
        self.screen[1].actions = self.screen_actions
        for host in self.config.get("ssh_hosts", []):
            self.ssh.add_host(HostSpec(**host))
        # Fleet management: metrics cache + background health polling + batch
        # execution, all riding on the same hub/approver as the SSH tools.
        # Polling starts with the runtime (like the PTY reaper) so the fleet
        # page shows liveness even before anyone opens it.
        self.fleet = Fleet(self.ssh, audit=self.audit, approver=self._ask_human)
        self.fleet.start_polling()
        self.providers = self._build_providers()
        self.kernel = self._build_kernel(enabled=True)
        self.extensions = self._load_extensions()
        self.browser_runs: OrderedDict[str, dict] = OrderedDict()

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

    # -- provider management (config-backed, hot-reloaded) ----------------
    _PROVIDER_KINDS = ("openai_compat", "anthropic", "gemini")

    def provider_details(self) -> list[dict]:
        raw = self.config.get("providers") or {}
        default_name = self.config.get("default_provider") or next(iter(self.providers), "")
        out = []
        for name, provider in self.providers.items():
            spec = provider.spec
            entry = raw.get(name) or {}
            out.append({
                "name": name,
                "kind": spec.kind or "openai_compat",
                "model": spec.model,
                "base_url": spec.base_url,
                "tier": spec.tier,
                "key_env": entry.get("api_key_env", ""),
                "key_required": bool(entry.get("api_key_env") or entry.get("api_key")),
                "key_present": bool(spec.api_key),
                "default": name == default_name,
            })
        return out

    def upsert_provider(self, name: str, body: dict) -> dict:
        """Add or replace one seat, persist it, and rebuild the provider map."""
        import re as _re

        if not _re.match(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$", name or ""):
            raise ValueError("provider name must start with a letter and use letters, digits, "
                             "'-' or '_' (max 32 chars)")
        kind = str(body.get("kind") or "openai_compat").strip()
        if kind not in self._PROVIDER_KINDS:
            raise ValueError(f"unknown kind '{kind}'. known: {', '.join(self._PROVIDER_KINDS)}")
        model = str(body.get("model") or "").strip()
        if not model:
            raise ValueError("'model' is required (the model id the endpoint expects)")
        entry: dict = {"kind": kind, "model": model}
        base_url = str(body.get("base_url") or "").strip()
        if kind == "openai_compat" and not base_url:
            raise ValueError("an openai_compat provider needs a base_url (e.g. https://host/v1)")
        if base_url:
            entry["base_url"] = base_url
        if body.get("api_key_env"):
            entry["api_key_env"] = str(body["api_key_env"]).strip()
        elif body.get("api_key"):
            entry["api_key"] = str(body["api_key"]).strip()
        if body.get("tier") in ("cheap", "strong"):
            entry["tier"] = body["tier"]
        if body.get("temperature") is not None:
            try:
                entry["temperature"] = float(body["temperature"])
            except (TypeError, ValueError):
                raise ValueError("temperature must be a number")

        providers = self.config.setdefault("providers", {})
        providers[name] = entry
        if len(providers) == 1 or not self.config.get("default_provider"):
            self.config["default_provider"] = name
        self.save_config()
        self.reload_providers()
        out = {"name": name, **{k: v for k, v in entry.items() if k != "api_key"}}
        if "api_key" in entry:
            out["note"] = "key stored in plain text inside agentd.json (0600); " \
                          "prefer api_key_env pointing at a systemd EnvironmentFile"
        return out

    def remove_provider(self, name: str) -> dict:
        providers = self.config.get("providers") or {}
        if name not in providers:
            raise KeyError(name)
        removed = providers.pop(name)
        if self.config.get("default_provider") == name:
            self.config["default_provider"] = next(iter(providers), "")
        self.save_config()
        self.reload_providers()
        return {"removed": name, "model": removed.get("model", "")}

    def probe_provider(self, name: str) -> dict:
        provider = self.providers.get(name)
        if provider is None:
            raise KeyError(name)
        try:
            caps = probe_capabilities(provider.spec, timeout=10).as_dict()
        except Exception as exc:
            return {"name": name, "reachable": False, "notes": _short_reason(exc)}
        return {"name": name, **caps}

    def reload_providers(self) -> None:
        self.providers = self._build_providers()

    # -- fleet host management (config-backed, hot-applied) ----------------
    _LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
    _HOST_EDITABLE = ("host", "port", "user", "key_path", "password_env", "jump",
                      "use_login_shell", "policy", "tags", "metrics_cmd")

    def _host_state(self, label: str) -> dict:
        """One row for runtime.state(): identity, policy and cached liveness."""
        spec = self.ssh.get_host(label)
        return {
            "label": label,
            "host": spec.host,
            "policy": (getattr(spec, "policy", "") or "standard"),
            "tags": list(getattr(spec, "tags", []) or []),
            "online": self.fleet.online(label),
        }

    def _host_spec_dict(self, spec: HostSpec) -> dict:
        """Config-file shape of a host (round-trips through HostSpec(**entry))."""
        return {
            "label": spec.label, "host": spec.host, "port": spec.port, "user": spec.user,
            "key_path": spec.key_path, "password_env": spec.password_env, "jump": spec.jump,
            "use_login_shell": spec.use_login_shell,
            "policy": (getattr(spec, "policy", "") or "standard"),
            "tags": list(getattr(spec, "tags", []) or []),
            "metrics_cmd": (getattr(spec, "metrics_cmd", "") or ""),
        }

    def _validate_host_fields(self, fields: dict) -> dict:
        """Validate the editable host fields; returns clean HostSpec kwargs."""
        out: dict = {}
        address = str(fields.get("host") or "").strip()
        if not address or any(c.isspace() for c in address):
            raise ValueError("'host' must be a non-empty address without whitespace")
        out["host"] = address
        try:
            port = int(fields.get("port", 22))
        except (TypeError, ValueError):
            raise ValueError("'port' must be an integer")
        if not 1 <= port <= 65535:
            raise ValueError("'port' must be within 1..65535")
        out["port"] = port
        out["user"] = str(fields.get("user") or "").strip()
        out["key_path"] = str(fields.get("key_path") or "").strip()
        out["password_env"] = str(fields.get("password_env") or "").strip()
        out["jump"] = str(fields.get("jump") or "").strip()
        policy = str(fields.get("policy") or "standard").strip().lower()
        if policy not in POLICIES:
            raise ValueError(f"unknown policy '{policy}'. known: {', '.join(POLICIES)}")
        out["policy"] = policy
        tags = fields.get("tags") or []
        if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
            raise ValueError("'tags' must be a list of strings")
        cleaned: list[str] = []
        for tag in tags:
            tag = tag.strip()
            if tag and tag not in cleaned:
                cleaned.append(tag)
        out["tags"] = cleaned
        # metrics_cmd is free-form on purpose, but it executes remotely with the
        # host's credentials: the fleet module gate-checks it and demands a
        # one-shot approval unless it is provably read-only. See fleet.py.
        out["metrics_cmd"] = str(fields.get("metrics_cmd") or "").strip()
        return out

    def add_host(self, body: dict) -> dict:
        """Register a host: validate, hot-add to the hub, persist to agentd.json."""
        label = str(body.get("label") or "").strip()
        if not self._LABEL_RE.match(label):
            raise ValueError("host label must start with a letter and use letters, digits, "
                             "'-', '_' or '.' (max 64 chars)")
        if label == "self":
            raise ValueError("'self' is reserved for the local host and cannot be registered")
        hosts = list(self.config.get("ssh_hosts") or [])
        if any(h.get("label") == label for h in hosts):
            raise ValueError(f"a host named '{label}' already exists")
        fields = self._validate_host_fields(body)
        if fields["jump"] and fields["jump"] not in self.ssh.hosts():
            raise ValueError(f"jump host '{fields['jump']}' is not registered")
        spec = HostSpec(label=label, **fields)
        self.config["ssh_hosts"] = hosts + [self._host_spec_dict(spec)]
        self.ssh.add_host(spec)     # hot: usable without a restart
        self.save_config()
        return self._host_spec_dict(spec)

    def update_host(self, label: str, body: dict) -> dict:
        """Partially update a host; unknown fields are refused, not ignored."""
        if label not in self.ssh.hosts():
            raise SSHError(f"unknown host '{label}'")
        unknown = [k for k in body if k not in self._HOST_EDITABLE and k != "label"]
        if unknown:
            raise ValueError(f"unknown field(s): {', '.join(sorted(unknown))}")
        merged = {**self._host_spec_dict(self.ssh.get_host(label)),
                  **{k: body[k] for k in body if k in self._HOST_EDITABLE}}
        fields = self._validate_host_fields(merged)
        if fields["jump"] and fields["jump"] not in self.ssh.hosts() and fields["jump"] != label:
            raise ValueError(f"jump host '{fields['jump']}' is not registered")
        spec = HostSpec(label=label, **fields)
        hosts = [h for h in (self.config.get("ssh_hosts") or []) if h.get("label") != label]
        self.config["ssh_hosts"] = hosts + [self._host_spec_dict(spec)]
        self.ssh.add_host(spec)     # hot: replaces the old spec in place
        self.save_config()
        return self._host_spec_dict(spec)

    def remove_host(self, label: str) -> dict:
        """Forget a host in config and hub; running sessions are never killed."""
        if label == "self":
            raise ValueError("the 'self' host cannot be removed")
        hosts = list(self.config.get("ssh_hosts") or [])
        remaining = [h for h in hosts if h.get("label") != label]
        in_config = len(remaining) != len(hosts)
        in_hub = label in self.ssh.hosts()
        if not in_config and not in_hub:
            raise KeyError(label)
        if in_config and not remaining:
            raise ValueError("cannot remove the last host in the fleet")
        self.config["ssh_hosts"] = remaining
        if in_hub:
            self.ssh.remove_host(label)
        self.save_config()
        return {"ok": True, "removed": label}

    def save_config(self) -> None:
        """Persist the runtime config atomically (0600) keeping one backup.

        The same file the service loads on restart, so a seat added in the UI
        survives a reboot by construction rather than by convention.
        """
        path = config_path(os.environ.get("AGENTD_CONFIG"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = path.with_suffix(path.suffix + ".bak")
            try:
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                os.chmod(backup, 0o600)
            except OSError:
                pass
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.config, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def _build_kernel(self, enabled: bool) -> JitRLKernel:
        cfg = self.config.get("kernel", {})
        return JitRLKernel(store=self.store, beta=float(cfg.get("beta", 1.0)),
                           gamma=float(cfg.get("gamma", 0.95)), top_k=int(cfg.get("top_k", 8)),
                           ngram=int(cfg.get("ngram", 2)), min_sim=float(cfg.get("min_sim", 0.02)),
                           exploration_prob=float(cfg.get("epsilon", 0.05)),
                           track_credit=bool(cfg.get("track_credit", True)),
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
        """Blocking approval request surfaced to connected clients.

        Sessions run on `asyncio.to_thread` worker threads, so there is no
        running event loop here — the previous asyncio.Queue waiter raised
        RuntimeError on every call and silently denied all pending commands.
        A threading.Event is safe to wait on from the worker thread and to set
        from the HTTP handler thread; the event loop never blocks.
        """
        if os.environ.get("AGENTD_AUTO_APPROVE") == "1":
            self.publish({"type": "approval", "auto": True, **payload})
            return True
        token = uuid.uuid4().hex[:12]
        record = {"token": token, "payload": payload,
                  "event": threading.Event(), "decided": None}
        self.pending_approvals[token] = record
        session = self._current_session()
        if session:
            session.pending_approvals[token] = record
        self.publish({"type": "approval_required", "token": token, **payload})
        timeout = float(os.environ.get("AGENTD_APPROVAL_TIMEOUT", "300"))
        record["event"].wait(timeout)
        # Remove the record whichever way it ends: a timeout must not leave a
        # phantom approval card, and a resolved one no longer needs a slot.
        self.pending_approvals.pop(token, None)
        if session:
            session.pending_approvals.pop(token, None)
        return record["decided"] is True

    def resolve_approval(self, token: str, approved: bool) -> bool:
        record = self.pending_approvals.get(token)
        if record is None:
            return False
        record["decided"] = approved
        record["event"].set()
        self.publish({"type": "approval_resolved", "token": token, "approved": approved})
        self.pending_approvals.pop(token, None)
        for session in self.sessions.values():
            session.pending_approvals.pop(token, None)
        return True

    def _current_session(self):
        return getattr(self._local, "session", None)

    # -- sessions ---------------------------------------------------------
    def build_task(self, task_name: str, *, kind: str = "task", host: str = ""):
        """Resolve a session's environment.

        `kind="task"` runs a scripted built-in task (measurable, closed action
        set). `kind="ops"` runs an open goal through the tool bus -- real SSH
        and local calls, same loop and same safety gate.
        """
        if kind == "ops":
            from .envs.tool_task import ToolTask

            goal = (task_name or "").strip()
            if not goal:
                raise ValueError("an ops session needs a goal to work on")
            hosts = self.ssh.hosts()
            return ToolTask(goal_text=goal, hosts=hosts,
                            default_host=(host or (hosts[0] if hosts else "")))
        if kind not in ("task", "auto"):
            raise ValueError(f"unknown session kind '{kind}'. use 'task' or 'ops'")
        factory = BUILTIN_TASKS.get(task_name)
        if factory is None:
            raise KeyError(
                f"unknown task '{task_name}'. available: {', '.join(sorted(BUILTIN_TASKS))} "
                "(or pass kind='ops' to run an open goal through the tools)")
        return factory()

    def _council_members(self, primary, requested: list[str] | None) -> list:
        """The seats at the table.

        Caller-picked names win (a typo is refused rather than silently
        shrinking the council); otherwise the primary plus the next configured
        providers, capped so a deliberation cannot quietly multiply token cost.
        """
        if requested:
            missing = [n for n in requested if n not in self.providers]
            if missing:
                raise ValueError(
                    f"unknown council member(s): {', '.join(missing)}. "
                    f"configured: {', '.join(self.providers) or 'none'}")
            crew = [primary]
            for name in requested:
                provider = self.providers[name]
                if all(provider is not member for member in crew):
                    crew.append(provider)
            return crew
        cap = max(2, int((self.config.get("council") or {}).get("max_members", 3)))
        crew = [primary]
        for provider in self.providers.values():
            if len(crew) >= cap:
                break
            if all(provider is not member for member in crew):
                crew.append(provider)
        return crew

    def create_session(self, task_name: str, *, provider: str = "", learning: bool = True,
                       max_steps: int = 12, council: bool = False,
                       members: list[str] | None = None, kind: str = "task",
                       host: str = "") -> Session:
        task_obj = self.build_task(task_name, kind=kind, host=host)
        model = self.default_provider(provider)
        kernel = self._build_kernel(learning)
        crew = self._council_members(model, members) if council else [model]
        deliberates = bool(council) and len(crew) > 1
        loop = AgentLoop(
            kernel, model, self.bus, context=ContextBuilder(), ledger=Ledger(),
            council=Council(kernel, crew, trigger=TriggerPolicy(max_members=len(crew)))
            if deliberates else None,
            config=LoopConfig(max_steps=max_steps, deliberate=deliberates),
            approver=self._ask_human, ssh=self.ssh,
        )
        session = Session(id=uuid.uuid4().hex[:10], task=task_name, loop=loop,
                          task_obj=task_obj)
        self.sessions[session.id] = session
        self._prune_sessions()
        return session

    def _prune_sessions(self) -> None:
        """Evict finished sessions past the TTL, then oldest-finished overage.

        Queued/running sessions are never evicted: dropping one would orphan a
        live episode. If every slot is busy the dict grows past the cap instead
        of killing work.
        """
        now = time.time()
        for sid in [sid for sid, s in self.sessions.items()
                    if s.status not in ("queued", "running")
                    and now - s.created_at > self.session_ttl_s]:
            del self.sessions[sid]
        while len(self.sessions) > self.max_sessions:
            for sid, s in self.sessions.items():
                if s.status not in ("queued", "running"):
                    del self.sessions[sid]
                    break
            else:
                break

    def run_session(self, session: Session) -> dict:
        self._local.session = session
        session.status = "running"
        self.store.add_trajectory_event(session.id, "status", json.dumps({"status": "running", "task": session.task}, ensure_ascii=False))
        self.publish({"type": "session_started", "session": session.id, "task": session.task})
        try:
            task = session.task_obj if session.task_obj is not None else BUILTIN_TASKS[session.task]()
            report = session.loop.run(task)
            session.report = report.as_dict()
            session.status = "success" if report.success else "failed"
            # Keep the high-value parts of the loop trace in the same replayable
            # stream as lifecycle events. Raw provider prompts and credentials
            # are never copied here; tool output is bounded by the trace itself.
            for trace in session.report.get("trace", []):
                self.store.add_trajectory_event(session.id, "state", trace.get("state", ""))
                self.store.add_trajectory_event(
                    session.id, "observation", trace.get("tool_output", "") or "",
                )
                self.store.add_trajectory_event(
                    session.id, "tool", json.dumps({
                        "action": trace.get("chosen_action", ""),
                        "risk": trace.get("risk", ""), "ok": trace.get("ok", True),
                    }, ensure_ascii=False),
                )
            for council in session.report.get("council", []):
                self.store.add_trajectory_event(
                    session.id, "proposal", json.dumps(council, ensure_ascii=False, default=str),
                )
            self.store.add_trajectory_event(session.id, "decision", json.dumps(session.report, ensure_ascii=False, default=str))
            self.store.add_trajectory_event(session.id, "reward", json.dumps({"success": bool(report.success), "steps": len(report.steps)}, ensure_ascii=False))
        except Exception as exc:
            session.status = "error"
            session.report = {"error": _short_reason(exc)}
            self.store.add_trajectory_event(session.id, "reflection", session.report["error"])
            self.publish({"type": "session_error", "session": session.id, "error": _short_reason(exc)})
        finally:
            self._local.session = None
        self.publish({"type": "session_finished", "session": session.id, "status": session.status,
                      "report": session.report})
        self.store.add_trajectory_event(session.id, "status", json.dumps({"status": session.status}, ensure_ascii=False))
        return session.report or {}

    def trajectory(self, run_id: str, limit: int = 500) -> list[dict]:
        return [event.as_dict() for event in self.store.trajectory_for(run_id, limit)]

    def task_templates(self) -> list[dict]:
        return templates_from_config(self.config)

    def save_task_templates(self, templates: list[dict]) -> list[dict]:
        if not isinstance(templates, list) or len(templates) > 100:
            raise ValueError("task_templates must be a list with at most 100 entries")
        clean = []
        for item in templates:
            if not isinstance(item, dict) or not str(item.get("id", "")).strip():
                raise ValueError("each task template needs an id")
            clean.append({str(k): v for k, v in item.items()})
        self.config["task_templates"] = clean
        self.save_config()
        return clean

    def decision_model(self) -> dict:
        value = self.config.get("decision_model")
        return dict(value) if isinstance(value, dict) else {"enabled": False, "provider": "jev", "fallback": "council"}

    def save_decision_model(self, body: dict) -> dict:
        if not isinstance(body, dict):
            raise ValueError("decision model must be an object")
        fallback = str(body.get("fallback", "council"))
        if fallback not in {"council", "human"}:
            raise ValueError("fallback must be council or human")
        value = {"enabled": bool(body.get("enabled", False)), "provider": "jev",
                 "endpoint": str(body.get("endpoint", "")).strip(),
                 "model": str(body.get("model", "")).strip(), "fallback": fallback}
        if value["enabled"] and (not value["endpoint"] or not value["model"]):
            raise ValueError("enabled decision model needs endpoint and model")
        self.config["decision_model"] = value
        self.save_config()
        return value

    def publish(self, event: dict) -> None:
        event["ts"] = time.time()
        loop = self.main_loop
        with self._subscribers_lock:
            subscribers = list(self.subscribers)
        for queue in subscribers:
            if loop is not None:
                try:
                    # asyncio.Queue.put_nowait is not safe against an awaiting
                    # consumer on the loop thread; marshalling the put through
                    # call_soon_threadsafe keeps it on the thread that owns the
                    # queue. It is also safe when called from the loop itself.
                    loop.call_soon_threadsafe(self._offer, queue, event)
                    continue
                except RuntimeError:
                    pass    # loop closed mid-shutdown: fall back to a direct put
            self._offer(queue, event)

    def _offer(self, queue: asyncio.Queue, event: dict) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # A slow subscriber must not stall the agent loop, but a silent
            # drop is indistinguishable from a bug — count it where state()
            # can report it.
            self.dropped_events += 1
            log.warning("subscriber queue full, event dropped (total dropped: %d)",
                        self.dropped_events)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        if self.main_loop is None:
            try:
                self.main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass    # no loop (CLI/tests): direct puts are the only option anyway
        with self._subscribers_lock:
            self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._subscribers_lock:
            if queue in self.subscribers:
                self.subscribers.remove(queue)

    # -- snapshot for the UI ---------------------------------------------
    def state(self) -> dict:
        caps = {}
        for name, provider in self.providers.items():
            try:
                caps[name] = probe_capabilities(provider.spec, timeout=8).as_dict()
            except Exception as exc:
                caps[name] = {"label": name, "reachable": False, "notes": _short_reason(exc)}
        return {
            "providers": {name: {"model": p.spec.model, "base_url": p.spec.base_url, "tier": p.spec.tier}
                          for name, p in self.providers.items()},
            "capabilities": caps,
            "kernel": self.kernel.stats(),
            "tools": {"count": len(self.bus.names()), "by_source": self.bus.by_source()},
            # Structured host rows (label/policy/tags/liveness) so the console
            # dropdown and the fleet page share one source of truth.
            "hosts": [self._host_state(label) for label in self.ssh.hosts()],
            "tasks": sorted(BUILTIN_TASKS),
            "extensions": self.extensions,
            "sessions": [{"id": s.id, "task": s.task, "status": s.status} for s in self.sessions.values()],
            "session_count": len(self.sessions),
            "max_sessions": self.max_sessions,
            "events": {"subscribers": len(self.subscribers), "dropped": self.dropped_events},
            "db": self.db_path,
        }

    def close(self) -> None:
        self.fleet.stop_polling()
        self.pty.stop_reaper()
        for server in self._mcp_servers.values():
            server.stop()
        self.ssh.close()


def _short_reason(exc: BaseException) -> str:
    """One human-readable line for a client; full detail stays server-side.

    Exception class names plus raw messages leak internal paths and stack hints
    into API responses and the event stream, while the operator only needs the
    outcome ("connection refused", "unknown host").
    """
    lines = str(exc).strip().splitlines()
    return (lines[0] if lines else "unknown error")[:200]


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
