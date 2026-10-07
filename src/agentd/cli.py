"""agentd command line.

API keys are only ever read from the environment or a 0600 config file — never
from argv, because argv ends up in shell history and in `ps` output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

from . import __version__


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import demo, render

    print(render(demo(episodes=args.episodes, seed=args.seed)))
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from .providers import ProviderSpec, probe_capabilities

    api_key = os.environ.get(args.api_key_env or "AGENTD_API_KEY", "")
    if not api_key:
        print(f"note: ${args.api_key_env or 'AGENTD_API_KEY'} is empty; probing without an Authorization header", file=sys.stderr)
    spec = ProviderSpec(label=args.label, model=args.model, base_url=args.base_url, api_key=api_key,
                        timeout_s=args.timeout)
    caps = probe_capabilities(spec, use_cache=False)
    print(json.dumps(caps.as_dict(), indent=2, ensure_ascii=False))
    if not caps.reachable:
        print("\nendpoint unreachable — fix base_url/network before trying the agent", file=sys.stderr)
        return 2
    if caps.mode.value == "token":
        print("\n=> JitRL token mode available: candidate scores come straight from top_logprobs.")
    elif caps.mode.value == "n_sample":
        print("\n=> degraded to k-sample voting (logprobs present, no candidate mass).")
    else:
        print("\n=> degraded to verbalised grading (no logprobs at all).")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0]))
    for dep, hint in (("ssh", "SSH tools (system OpenSSH)"), ("scp", "file transfer (system OpenSSH)")):
        found = shutil.which(dep) is not None
        checks.append((dep, found, found and "on PATH" or f"install {dep}"))
    for mod, hint in (("fastapi", "HTTP API (pip install 'agentd[api]')"), ("uvicorn", "server")):
        try:
            __import__(mod)
            checks.append((mod, True, "installed"))
        except ImportError:
            checks.append((mod, False, hint))
    try:
        from .kernel import JitRLKernel, Step, Store

        store = Store()
        kernel = JitRLKernel(store=store)
        episode_id = store.start_episode("doctor", "doctor")
        store.add_step(Step(episode_id=episode_id, t=0, state="s", state_fp="fp", action="a",
                            action_fp="a", ret=1.0))
        roundtrip = store.steps_for_episode(episode_id)[0].ret == 1.0
        store.finish_episode(episode_id, True, 1.0, "")
        checks.append(("kernel roundtrip", roundtrip, f"episodes={kernel.stats()['episodes']} steps={kernel.stats()['steps']}"))
    except Exception as exc:
        checks.append(("kernel import", False, f"{type(exc).__name__}: {exc}"))

    db = os.path.expanduser(args.db)
    parent = os.path.dirname(db) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        probe_file = os.path.join(parent, ".agentd-write-test")
        with open(probe_file, "w"):
            pass
        os.remove(probe_file)
        checks.append((f"data dir writable ({parent})", True, "ok"))
    except OSError as exc:
        checks.append((f"data dir writable ({parent})", False, exc.strerror or str(exc)))
    env_keys = [k for k in os.environ if k.startswith("AGENTD_")]
    checks.append(("config from env", True, ", ".join(env_keys) or "AGENTD_* unset"))

    width = max(len(name) for name, _, _ in checks)
    failures = 0
    for name, ok, detail in checks:
        print(f"[{'ok ' if ok else 'FAIL'}] {name:<{width}}  {detail}")
        failures += 0 if ok else 1
    print(f"\nagentd {__version__} — {'ready' if not failures else f'{failures} problem(s)'}")
    return 1 if failures else 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from .kernel import Store

    store = Store(os.path.expanduser(args.db))
    print(json.dumps({
        "db": args.db,
        "episodes": store.count_episodes(),
        "steps": store.count_steps(),
    }, indent=2))
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    from .kernel import Store
    from .kernel.pack import diff_packs, export_pack, import_pack, load_pack, save_pack, to_markdown

    store = Store(os.path.expanduser(args.db))
    source = {"model": args.model or os.environ.get("AGENTD_MODEL", ""),
              "host": args.host or os.environ.get("AGENTD_HOST", "")}

    if args.action == "list":
        rows = store.db.execute(
            "SELECT task, COUNT(*) AS n, SUM(success) AS wins FROM episodes GROUP BY task ORDER BY task"
        ).fetchall()
        if not rows:
            print("memory is empty")
        for row in rows:
            win_rate = (row["wins"] or 0) / row["n"] if row["n"] else 0.0
            print(f"{row['task']:<34} episodes={row['n']:<4} win_rate={win_rate:.2f}")
        print(f"\n{store.count_steps()} experience rows in {args.db}")
        return 0

    if args.action == "export":
        pack = export_pack(store, task=args.task, source=source)
        written = save_pack(pack, args.out or f"memory-{int(time.time())}.agentdmem")
        print(json.dumps(written, indent=2, ensure_ascii=False))
        print(f"\nreadable view: {written['markdown']}")
        return 0

    if args.action == "import":
        pack = load_pack(args.pack)
        report = import_pack(store, pack, skip_tasks=args.skip or [], current_source=source)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        for warning in report["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
        return 0

    if args.action == "diff":
        left = load_pack(args.pack) if args.pack else export_pack(store, source=source)
        right = load_pack(args.other)
        print(json.dumps(diff_packs(left, right), indent=2, ensure_ascii=False))
        return 0

    if args.action == "show":
        pack = load_pack(args.pack)
        print(to_markdown(pack))
        return 0

    print("usage: agentd memory {list|export|import|diff|show}", file=sys.stderr)
    return 2


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import _real_provider_or_none, bench, render, web_bench

    if args.suite == "web":
        result = web_bench(episodes=args.episodes, policy=args.policy, seed=args.seed,
                           noise=args.noise, beta=args.beta, gamma=args.gamma)
    else:
        provider = _real_provider_or_none()
        model_note = ""
        if provider is None:
            model_note = "(no AGENTD_BASE_URL set → using the built-in synthetic model)"
        result = bench(episodes=args.episodes, beta=args.beta, noise=args.noise, seed=args.seed,
                       max_steps=args.steps, council=args.council, provider=provider)
    if args.suite != "web" and "model_note" in dir() and model_note:
        print(f"note: {model_note}", file=sys.stderr)
    print(render(result))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .bench import _real_provider_or_none, build_loop
    from .envs.ops_tasks import BUILTIN_TASKS
    from .kernel import Store

    factory = BUILTIN_TASKS.get(args.task)
    if factory is None:
        print(f"unknown task '{args.task}'. available: {', '.join(sorted(BUILTIN_TASKS))}", file=sys.stderr)
        return 2
    provider = _real_provider_or_none()
    store_path = os.path.expanduser(args.db)
    store = Store(store_path)
    loop = build_loop(store, enabled=not args.no_memory, beta=args.beta, noise=args.noise,
                      seed=args.seed, max_steps=args.steps, council=args.council, provider=provider)
    report = loop.run(factory())
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    print(f"\nepisode stored in {store_path} "
          f"({'success' if report.success else 'failure'}, {len(report.steps)} steps)", file=sys.stderr)
    return 0 if report.success else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    from .runtime import Runtime, load_config
    from .viewer import tunnel_command

    if args.host not in ("127.0.0.1", "localhost", "::1") and os.environ.get("AGENTD_ALLOW_PUBLIC") != "1":
        print(f"refusing to bind {args.host}: agentd can execute commands on your servers.\n"
              "Set AGENTD_ALLOW_PUBLIC=1 only if you know what that costs you.", file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError:
        print("pip install uvicorn", file=sys.stderr)
        return 1

    runtime = Runtime(load_config(args.config))
    from .api import create_app

    app = create_app(runtime)
    print(f"agentd api  http://{args.host}:{args.port}  (loopback only)")
    print(f"memory      {runtime.db_path}")
    print(f"providers   {', '.join(runtime.providers) or 'NONE — set AGENTD_BASE_URL/AGENTD_MODEL'}")
    print(f"tools       {len(runtime.bus.names())} registered")
    if args.host in ("127.0.0.1", "localhost"):
        print("from your laptop:  " + tunnel_command("user@your-server", port=args.ssh_port))
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
    return 0


def _cmd_viewer(args: argparse.Namespace) -> int:
    from .runtime import Runtime, load_config
    from .viewer import setup_via_ssh, viewer_status

    runtime = Runtime(load_config(args.config))
    if args.action == "status":
        print(json.dumps(viewer_status(runtime), indent=2))
        return 0
    if args.action == "url":
        from .viewer import viewer_url

        token = runtime.vnc_tokens.issue("cli")
        print(viewer_url(token=token.value))
        print(f"(valid for {token.as_dict()['expires_in_s']}s)", file=sys.stderr)
        return 0
    if not args.host:
        print("viewer setup needs --host <label>", file=sys.stderr)
        return 2
    report = setup_via_ssh(runtime.ssh, args.host, approved=args.apply)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("blocked"):
        print("\nhalf of these commands change the host: re-run with --apply after you have read them",
              file=sys.stderr)
        return 3
    return 0 if report.get("ready") else 1


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import serve
    from .runtime import Runtime, load_config

    runtime = Runtime(load_config(args.config))
    if args.handshake:
        print(json.dumps({"protocolVersion": "2025-06-18", "serverInfo": {"name": "agentd"},
                          "tools": len(runtime.bus.names(include_hidden=True)),
                          "resources": 4}, indent=2))
        return 0
    print("agentd mcp server on stdio (newline-delimited JSON-RPC). Ctrl-D to stop.", file=sys.stderr)
    try:
        return serve(runtime, runtime.bus)
    finally:
        runtime.close()


def _runtime(args: argparse.Namespace):
    from .runtime import Runtime, load_config

    return Runtime(load_config(getattr(args, "config", None)))


def _cmd_host(args: argparse.Namespace) -> int:
    from .envs.ssh_env import HostSpec

    runtime = _runtime(args)
    if args.action == "list":
        if not runtime.ssh.hosts():
            print("no hosts configured — add an \"ssh_hosts\" entry to agentd.json")
        for label in runtime.ssh.hosts():
            spec = runtime.ssh.get_host(label)
            print(f"{label:<16} {spec.target()}:{spec.port}  jump={spec.jump or '-'}  "
                  f"key={spec.key_path or '-'}  pw_env={spec.password_env or '-'}")
        return 0

    if args.action == "add":
        if not (args.host and args.label):
            print("host add needs --label and --host", file=sys.stderr)
            return 2
        spec = HostSpec(label=args.label, host=args.host, port=args.port, user=args.user,
                        key_path=args.key or "", password_env=args.password_env or "",
                        jump=args.jump or "")
        current = {h["label"]: h for h in (runtime.config.get("ssh_hosts") or [])}
        current[spec.label] = {k: v for k, v in spec.__dict__.items() if k in
                               ("host", "port", "user", "key_path", "password_env", "jump",
                                "use_login_shell")}
        runtime.config["ssh_hosts"] = list(current.values())
        path = os.path.expanduser(args.config or os.environ.get(
            "AGENTD_CONFIG", "~/.config/agentd/agentd.json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(runtime.config, fh, indent=2, ensure_ascii=False)
        os.chmod(path, 0o600)
        print(f"saved host '{spec.label}' to {path} (0600)")
        print("secrets are stored as an ENV VAR NAME, never as a value")
        return 0

    if args.action == "probe":
        if not args.label:
            print("host probe needs --label", file=sys.stderr)
            return 2
        try:
            facts = runtime.ssh.probe(args.label)
        except Exception as exc:
            print(f"probe failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(facts, indent=2, ensure_ascii=False))
        verdict = "can host WebArena" if facts.get("webarena_viable") else \
                  "cannot host WebArena (needs x86_64 + 16GB RAM + 80GB disk + Docker)"
        print(f"\n{args.label}: {verdict}", file=sys.stderr)
        return 0
    return 2


def _cmd_schedule(args: argparse.Namespace) -> int:
    from datetime import datetime

    from .scheduler import Cron, CronError, Job, ScheduleStore, Scheduler

    runtime = _runtime(args)
    directory = os.path.dirname(os.path.expanduser(runtime.db_path))
    store = ScheduleStore(directory)

    if args.action == "list":
        jobs = store.load()
        if not jobs:
            print("no scheduled jobs — agentd schedule add --name ... --cron ... --task ...")
        for job in jobs:
            try:
                nxt = job.schedule().next_after(datetime.now())
            except CronError as exc:
                nxt = f"invalid cron: {exc}"
            except Exception:
                nxt = "?"
            print(f"{job.name:<14} {job.cron:<16} task={job.task:<14} "
                  f"enabled={job.enabled} last={job.last_status or '-'} next={nxt}")
        return 0

    if args.action == "add":
        if not (args.name and args.cron and args.task):
            print("schedule add needs --name --cron --task", file=sys.stderr)
            return 2
        try:
            Cron(args.cron)
        except CronError as exc:
            print(f"rejecting bad schedule: {exc}", file=sys.stderr)
            return 2
        job = Job(name=args.name, cron=args.cron, task=args.task, provider=args.provider,
                  learning=not args.no_memory, max_steps=args.steps, council=args.council)
        store.upsert(job)
        print(json.dumps(job.as_dict(), indent=2))
        return 0

    if args.action == "remove":
        return 0 if store.remove(args.name or "") else 2

    if args.action == "history":
        path = store.history
        if not path.exists():
            print("no runs yet")
            return 0
        for line in path.read_text(encoding="utf-8").splitlines()[-(args.limit or 20):]:
            print(line)
        return 0

    if args.action == "run-now":
        job = next((j for j in store.load() if j.name == args.name), None)
        if job is None:
            print(f"no such job: {args.name}", file=sys.stderr)
            return 2
        scheduler = Scheduler(runtime, store)
        print(json.dumps(scheduler.run_job(job), indent=2, ensure_ascii=False))
        return 0

    if args.action == "daemon":
        scheduler = Scheduler(runtime, store, tick_seconds=args.tick)
        print(f"scheduler watching {store.path} every {args.tick}s (Ctrl-C to stop)")
        try:
            scheduler.start_background().join()
        except KeyboardInterrupt:
            scheduler.stop()
        return 0
    return 2


def _cmd_backend(args: argparse.Namespace) -> int:
    from .backends import probe_local

    if args.host:
        from .backends.deepgemm import probe_remote

        runtime = _runtime(args)
        verdict = probe_remote(runtime.ssh, args.host)
    else:
        verdict = probe_local()
    print(json.dumps(verdict.as_dict(), indent=2, ensure_ascii=False))
    if verdict.eligible:
        print("\neligible: DeepGEMM applies here. Follow the plan above.", file=sys.stderr)
    else:
        print("\nnot eligible for DeepGEMM — that is expected on most machines, and agentd "
              "does not need it.", file=sys.stderr)
    return 0 if verdict.eligible else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"agentd {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the keyless test-time-RL demo (memory on vs off)")
    d.add_argument("--episodes", type=int, default=40)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=_cmd_demo)

    pr = sub.add_parser("probe", help="ask any OpenAI-compatible endpoint what it can do")
    pr.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8080/v1")
    pr.add_argument("--model", default="local")
    pr.add_argument("--label", default="probe")
    pr.add_argument("--api-key-env", default="AGENTD_API_KEY")
    pr.add_argument("--timeout", type=int, default=30)
    pr.set_defaults(func=_cmd_probe)

    doc = sub.add_parser("doctor", help="check this machine is ready to run agentd")
    doc.add_argument("--db", default="~/.config/agentd/memory.db")
    doc.set_defaults(func=_cmd_doctor)

    st = sub.add_parser("stats", help="summarise a memory database")
    st.add_argument("--db", default="~/.config/agentd/memory.db")
    st.set_defaults(func=_cmd_stats)

    mem = sub.add_parser("memory", help="list/export/import/diff/show shareable memory packs")
    mem.add_argument("action", choices=["list", "export", "import", "diff", "show"])
    mem.add_argument("--db", default="~/.config/agentd/memory.db")
    mem.add_argument("--pack", help="path to a .agentdmem file")
    mem.add_argument("--other", help="second pack for diff")
    mem.add_argument("--out", help="output path for export")
    mem.add_argument("--task", default=None, help="restrict export to one task")
    mem.add_argument("--skip", action="append", default=[], help="task to skip on import (repeatable)")
    mem.add_argument("--model", default="", help="provenance: model that produced this memory")
    mem.add_argument("--host", default="", help="provenance: host the work ran on")
    mem.set_defaults(func=_cmd_memory)

    bn = sub.add_parser("bench", help="measure learning ON vs OFF on the built-in task suite")
    bn.add_argument("--episodes", type=int, default=40)
    bn.add_argument("--steps", type=int, default=6)
    bn.add_argument("--beta", type=float, default=1.0)
    bn.add_argument("--noise", type=float, default=0.22)
    bn.add_argument("--seed", type=int, default=7)
    bn.add_argument("--council", action="store_true", help="enable conditional multi-model deliberation")
    bn.add_argument("--suite", default="ops", choices=["ops", "web"],
                    help="ops = scripted tasks (fast, no browser); web = real pages via Chromium")
    bn.add_argument("--policy", default="heuristic", choices=["heuristic", "model"],
                    help="web suite policy: lexical baseline (keyless) or your configured endpoint")
    bn.add_argument("--gamma", type=float, default=0.95)
    bn.add_argument("--json", action="store_true", help="also dump raw numbers")
    bn.set_defaults(func=_cmd_bench)

    rn = sub.add_parser("run", help="run one episode of a built-in task and store the experience")
    rn.add_argument("--task", default="nginx-down", choices=sorted(_task_names()))
    rn.add_argument("--steps", type=int, default=8)
    rn.add_argument("--beta", type=float, default=1.0)
    rn.add_argument("--noise", type=float, default=0.22)
    rn.add_argument("--seed", type=int, default=7)
    rn.add_argument("--council", action="store_true")
    rn.add_argument("--no-memory", action="store_true", help="control arm: ignore stored experience")
    rn.add_argument("--db", default="~/.config/agentd/memory.db")
    rn.set_defaults(func=_cmd_run)

    sv = sub.add_parser("serve", help="run the HTTP API + SSE on loopback (for the desktop shell)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=int(os.environ.get("AGENTD_PORT", "8765")))
    sv.add_argument("--ssh-port", type=int, default=22)
    sv.add_argument("--config", default=None)
    sv.add_argument("--log-level", default="info")
    sv.set_defaults(func=_cmd_serve)

    vw = sub.add_parser("viewer", help="install/inspect the Xvfb+x11vnc+noVNC screen path")
    vw.add_argument("action", choices=["setup", "status", "url"])
    vw.add_argument("--host", default="", help="configured ssh host label")
    vw.add_argument("--config", default=None)
    vw.add_argument("--apply", action="store_true", help="approve host-modifying install commands")
    vw.set_defaults(func=_cmd_viewer)

    mc = sub.add_parser("mcp", help="expose agentd as an MCP server over stdio")
    mc.add_argument("--config", default=None)
    mc.add_argument("--handshake", action="store_true", help="print capability summary and exit")
    mc.set_defaults(func=_cmd_mcp)

    ho = sub.add_parser("host", help="manage SSH hosts the agent may operate on")
    ho.add_argument("action", choices=["list", "add", "probe"])
    ho.add_argument("--label", default="")
    ho.add_argument("--host", default="")
    ho.add_argument("--port", type=int, default=22)
    ho.add_argument("--user", default="")
    ho.add_argument("--key", default="", help="path to a private key")
    ho.add_argument("--password-env", default="", help="NAME of an env var holding the password")
    ho.add_argument("--jump", default="", help="bastion host label")
    ho.add_argument("--config", default=None)
    ho.set_defaults(func=_cmd_host)

    sc = sub.add_parser("schedule", help="recurring jobs:巡检/备份校验/日志排查")
    sc.add_argument("action", choices=["list", "add", "remove", "run-now", "history", "daemon"])
    sc.add_argument("--name", default="")
    sc.add_argument("--cron", default="")
    sc.add_argument("--task", default="")
    sc.add_argument("--provider", default="")
    sc.add_argument("--steps", type=int, default=12)
    sc.add_argument("--council", action="store_true")
    sc.add_argument("--no-memory", action="store_true")
    sc.add_argument("--tick", type=int, default=30)
    sc.add_argument("--limit", type=int, default=20)
    sc.add_argument("--config", default=None)
    sc.set_defaults(func=_cmd_schedule)

    be = sub.add_parser("backend", help="check whether DeepGEMM-class kernels apply to this hardware")
    be.add_argument("action", nargs="?", default="probe", choices=["probe"])
    be.add_argument("--host", default="", help="probe a configured ssh host instead of locally")
    be.add_argument("--config", default=None)
    be.set_defaults(func=_cmd_backend)
    return p


def _task_names() -> list[str]:
    from .envs.ops_tasks import BUILTIN_TASKS

    return sorted(BUILTIN_TASKS)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
