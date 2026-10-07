"""Memory packs: export, share, and merge learned experience as plain text.

Two design rules, both borrowed from what users complain about in agent tools:

* **No hidden state.** A pack renders to human-readable Markdown, so a teammate
  can literally read what the agent learned before trusting it.
* **Provenance is part of the data.** Advantage estimates learned from one model
  or one host do not automatically transfer, so the pack records what produced it
  and import warns when the current setup differs.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .store import Step, Store

FORMAT = "agentd-memory/1"


def _checksum(body: dict) -> str:
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def export_pack(store: Store, task: str | None = None, source: dict | None = None,
                include_states: bool = True) -> dict:
    if task:
        rows = store.db.execute("SELECT id FROM episodes WHERE task=?", (task,)).fetchall()
    else:
        rows = store.db.execute("SELECT id FROM episodes ORDER BY id").fetchall()
    episode_ids = [r["id"] for r in rows]

    episodes: list[dict] = []
    for episode_id in episode_ids:
        episode = store.get_episode(episode_id)
        if episode is None:
            continue
        steps = [
            {
                "t": s.t, "action": s.action, "scope": s.scope, "ret": round(s.ret, 6),
                "reward": s.reward, "z": s.z, "adv": s.adv, "z_prime": s.z_prime,
                **({"state": s.state} if include_states else {}),
            }
            for s in store.steps_for_episode(episode_id)
        ]
        if not steps:
            continue
        episodes.append({
            "task": episode.task, "goal": episode.goal, "success": episode.success,
            "score": episode.score, "analysis": episode.analysis, "steps": steps,
        })

    body = {
        "format": FORMAT,
        "created": time.time(),
        "source": source or {},
        "stats": {"episodes": len(episodes),
                  "steps": sum(len(e["steps"]) for e in episodes)},
        "episodes": episodes,
    }
    body["checksum"] = _checksum({k: v for k, v in body.items() if k != "checksum"})
    return body


def to_markdown(pack: dict) -> str:
    """Human-readable view: what the agent believes works, per task and state."""
    lines = [
        f"# agentd memory pack",
        "",
        f"- format: `{pack.get('format')}`",
        f"- episodes: {pack['stats']['episodes']}  steps: {pack['stats']['steps']}",
        f"- source: {json.dumps(pack.get('source', {}), ensure_ascii=False)}",
        f"- checksum: `{pack.get('checksum')}`",
        "",
    ]
    for episode in pack["episodes"]:
        verdict = {True: "success", False: "failed", None: "unknown"}[episode.get("success")]
        lines.append(f"## {episode['task']} — {verdict}")
        if episode.get("analysis"):
            lines.append(f"> {episode['analysis']}")
        lines.append("")
        by_state: dict[str, list[dict]] = {}
        for step in episode["steps"]:
            by_state.setdefault(step.get("state", "(no state)"), []).append(step)
        for state, steps in by_state.items():
            lines.append(f"- state: `{state[:110]}`")
            for step in sorted(steps, key=lambda s: s["ret"], reverse=True):
                scope = f" @{step['scope']}" if step.get("scope") else ""
                # `ret` is the discounted return *after* this action, so an action that
                # merely preceded a success can look good. reward>0 marks causality.
                cause = "  ← earned the terminal reward" if step.get("reward") else ""
                lines.append(f"  - **{step['action']}**{scope} → return {step['ret']:+.3f}"
                             f" (reward {step.get('reward', 0.0):+.2f}){cause}")
        lines.append("")
    return "\n".join(lines)


def verify_pack(pack: dict) -> bool:
    """Recompute the checksum. Never trust a caller-supplied `checksum_ok` flag."""
    body = {k: v for k, v in pack.items() if k not in ("checksum", "checksum_ok")}
    return pack.get("checksum") == _checksum(body)


def save_pack(pack: dict, path: str | Path, markdown: bool = True) -> dict:
    path = Path(path).expanduser()
    if path.suffix not in (".agentdmem", ".json"):
        path = path.with_suffix(".agentdmem")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    written = {"pack": str(path)}
    if markdown:
        md_path = path.with_suffix(".md")
        md_path.write_text(to_markdown(pack), encoding="utf-8")
        written["markdown"] = str(md_path)
    return written


def load_pack(path: str | Path) -> dict:
    raw = Path(path).expanduser().read_text(encoding="utf-8")
    pack = json.loads(raw)
    if pack.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported pack format {pack.get('format')!r} (expected {FORMAT!r})")
    pack["checksum_ok"] = verify_pack(pack)
    return pack


def _existing_keys(store: Store) -> set[tuple[str, str, str]]:
    rows = store.db.execute(
        "SELECT e.task AS task, s.state_fp AS sf, s.action_fp AS af FROM steps s"
        " JOIN episodes e ON e.id = s.episode_id"
    ).fetchall()
    return {(r["task"], r["sf"], r["af"]) for r in rows}


def import_pack(store: Store, pack: dict, *, skip_tasks: list[str] | None = None,
                current_source: dict | None = None) -> dict:
    """Merge a pack into a live store. Idempotent: re-importing adds nothing."""
    report = {"episodes_added": 0, "episodes_skipped": 0, "steps_added": 0, "steps_skipped": 0,
              "tasks_skipped": [], "warnings": [], "checksum_ok": verify_pack(pack)}
    if not report["checksum_ok"]:
        report["warnings"].append("checksum mismatch: the file was edited or truncated after export")

    if current_source and pack.get("source"):
        for key in ("model", "host"):
            mine, theirs = current_source.get(key), pack["source"].get(key)
            if mine and theirs and mine != theirs:
                report["warnings"].append(
                    f"pack was learned with {key}={theirs!r}, you are running {key}={mine!r}; "
                    "advantage estimates may not transfer")

    known = _existing_keys(store)
    known_tasks = {row["task"] for row in store.db.execute("SELECT DISTINCT task FROM episodes")}

    for episode in pack.get("episodes", []):
        task = episode.get("task", "")
        if skip_tasks and task in skip_tasks:
            report["tasks_skipped"].append(task)
            continue

        planned = [
            (task, fingerprint_of(step.get("state", "")), normalize(step["action"]), step)
            for step in episode.get("steps", [])
            if step.get("action")
        ]
        novel = [item for item in planned if (item[0], item[1], item[2]) not in known]
        report["steps_skipped"] += len(planned) - len(novel)
        if not novel:
            report["episodes_skipped"] += 1
            continue

        episode_id = store.start_episode(task, episode.get("goal", ""),
                                         meta={"imported_from": pack.get("checksum"),
                                               "source": pack.get("source", {})})
        report["episodes_added"] += 1
        for _task, state_fp, action_fp, step in novel:
            known.add((_task, state_fp, action_fp))
            store.add_step(Step(
                episode_id=episode_id, t=step.get("t", 0), state=step.get("state", ""),
                state_fp=state_fp, action=step["action"], action_fp=action_fp,
                scope=step.get("scope", ""), z=step.get("z"), adv=step.get("adv"),
                z_prime=step.get("z_prime"), reward=step.get("reward", 0.0), ret=step.get("ret", 0.0),
            ))
            report["steps_added"] += 1
        store.finish_episode(episode_id, episode.get("success"), episode.get("score"),
                             episode.get("analysis", ""))

    report["tasks_already_known"] = sorted(known_tasks)
    return report


def diff_packs(a: dict, b: dict) -> dict:
    def action_returns(pack: dict) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for episode in pack.get("episodes", []):
            for step in episode.get("steps", []):
                out.setdefault(episode["task"], {}).setdefault(normalize(step["action"]), step["ret"])
        return out

    left, right = action_returns(a), action_returns(b)
    shared = set(left) & set(right)
    return {
        "only_in_a": sorted(set(left) - shared),
        "only_in_b": sorted(set(right) - shared),
        "tasks": {
            task: {
                "a_actions": len(left[task]), "b_actions": len(right[task]),
                "only_in_a": sorted(set(left[task]) - set(right[task])),
                "only_in_b": sorted(set(right[task]) - set(left[task])),
            }
            for task in sorted(shared)
        },
    }


def normalize(text: str) -> str:
    from .state import normalize_action

    return normalize_action(text)


def fingerprint_of(text: str) -> str:
    from .state import fingerprint

    return fingerprint(text)
