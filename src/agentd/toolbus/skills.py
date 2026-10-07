"""Skill packs: a directory of `SKILL.md` files becomes callable tools.

The body of the file is the procedure a model should follow; it is deliberately
*not* pasted into every prompt. Only the one-line description goes into the
catalog, and the body is returned when the skill is actually invoked — that is
where most of the token saving in a tool-heavy agent comes from.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .spec import RISK_DANGEROUS, RISK_READ, RISK_WRITE, ToolResult, ToolSpec

BODY_LIMIT = 6_000


@dataclass
class Skill:
    name: str
    description: str
    when: str
    run: str
    args: list[str]
    risk: str
    body: str
    path: Path

    @property
    def parameters(self) -> dict:
        props = {a: {"type": "string"} for a in self.args}
        return {"type": "object", "properties": props, "required": list(self.args)}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal flat YAML frontmatter: `key: value` lines between --- markers."""
    if not text.startswith("---"):
        return {}, text.strip()
    try:
        _, block, body = text.split("---", 2)
    except ValueError:
        return {}, text.strip()
    meta: dict[str, object] = {}
    for line in block.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key in ("args", "arguments"):
            meta[key] = [a.strip() for a in value.strip("[]").split(",") if a.strip()]
        else:
            meta[key] = value
    return meta, body.strip()


def load_skill_file(path: Path) -> Skill:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    name = str(meta.get("name") or path.parent.name)
    return Skill(
        name=name,
        description=str(meta.get("description") or body.split("\n\n")[0][:160]),
        when=str(meta.get("when") or ""),
        run=str(meta.get("run") or ""),
        args=list(meta.get("args") or []),
        risk=str(meta.get("risk") or (RISK_WRITE if meta.get("run") else RISK_READ)),
        body=body,
        path=path,
    )


def _make_handler(skill: Skill):
    def run_skill(ctx=None, **kwargs) -> ToolResult:
        from ..safety.gate import classify, initial_readonly

        instructions = skill.body[:BODY_LIMIT]
        if not skill.run:
            return ToolResult(ok=True, output=instructions, risk=RISK_READ,
                              data={"skill": skill.name, "kind": "instructions-only"})
        missing = [a for a in skill.args if not kwargs.get(a)]
        if missing:
            return ToolResult(ok=False, error=f"skill '{skill.name}' missing args: {missing}")
        rendered = skill.run
        for key, value in kwargs.items():
            rendered = rendered.replace("{" + key + "}", shlex.quote(str(value)))
        if "{" in rendered and "}" in rendered:
            return ToolResult(ok=False, error=f"skill '{skill.name}' has unsubstituted placeholders left")

        # A skill pack is third-party code: same gate as any other shell tool.
        verdict = classify(rendered)
        human_ok = bool(ctx and ctx.approved)
        if verdict.hard_blocked and not human_ok:
            return ToolResult(ok=False, risk=RISK_DANGEROUS,
                              error=f"skill '{skill.name}' refused by safety gate: {'; '.join(verdict.reasons)}")
        if not human_ok and not initial_readonly(rendered):
            return ToolResult(
                ok=False, risk=RISK_WRITE,
                error=(f"skill '{skill.name}' runs a non-read-only command ({rendered[:120]}); "
                       "requires approval=true after the user confirms"),
            )
        try:
            proc = subprocess.run(rendered, shell=True, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"skill '{skill.name}' timed out after 180s")
        except OSError as exc:
            return ToolResult(ok=False, error=str(exc))
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return ToolResult(
            ok=proc.returncode == 0,
            output=f"[instructions]\n{instructions}\n\n[result]\n{out[:4000]}",
            data={"skill": skill.name, "rc": proc.returncode, "level": verdict.level},
            risk=skill.risk,
        )

    return run_skill


def load_skill_dir(bus, directory: str | Path, namespace: str = "skill") -> dict:
    directory = Path(directory).expanduser()
    report = {"loaded": [], "errors": {}, "skipped": []}
    if not directory.is_dir():
        report["skipped"].append(f"{directory} does not exist")
        return report

    for md in sorted(directory.glob("*/SKILL.md")):
        try:
            skill = load_skill_file(md)
        except Exception as exc:
            report["errors"][md.parent.name] = f"{type(exc).__name__}: {exc}"
            continue
        description = skill.description if not skill.when else f"{skill.description} — use when: {skill.when}"
        bus.register(
            ToolSpec(
                name=f"{namespace}.{skill.name}",
                description=description[:300],
                parameters=skill.parameters,
                source="skill",
                risk=skill.risk,
                handler=_make_handler(skill),
                examples=[],
            ),
            replace=True,
        )
        report["loaded"].append(f"{namespace}.{skill.name}")

    return report
