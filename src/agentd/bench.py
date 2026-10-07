"""`agentd bench` — the before/after curves, measured, not asserted.

Runs the same task suite twice with an identical model and identical sampling
stream, differing only in whether the JitRL memory is consulted. With no API key
configured it runs against a built-in synthetic model and labels the output as
such, so a contributor can reproduce the mechanism on a laptop in one command.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field

from .context import ContextBuilder, Ledger, PriceTable
from .council import Council, TriggerPolicy
from .envs.ops_tasks import BUILTIN_TASKS, nginx_down, approve_merge_request
from .kernel import JitRLKernel, Store
from .loop import AgentLoop, LoopConfig
from .providers import DecodeMode, MockProvider, OpenAICompatProvider, ProviderSpec
from .toolbus import ToolBus, register_builtins


def op_suite():
    return [nginx_down, approve_merge_request]


@dataclass
class SyntheticModel(MockProvider):
    """A deliberately bad operator model plus sampling noise (temperature proxy)."""

    def __init__(self, weights: dict[str, float], noise: float = 0.22, seed: int = 7):
        super().__init__(ProviderSpec(label="synthetic", model="synthetic", kind="synthetic"),
                         weights=weights, mode=DecodeMode.TOKEN,
                         critiques=["try the state-advancing action; do not stall, do not reboot"] * 400)
        self.rng = random.Random(seed)
        self.noise = noise

    def choose(self, system: str, user: str, candidates: list[str]):
        choice = super().choose(system, user, candidates)
        choice.z = {k: v + self.rng.gauss(0.0, self.noise) for k, v in choice.z.items()}
        return choice


SYNTHETIC_WEIGHTS = {
    "reboot the whole host": 0.70, "systemctl restart nginx": 0.30, "tail the nginx error log": 0.20,
    "run apt upgrade": 0.60, "curl the healthcheck endpoint": 0.35, "edit the nginx config file": 0.20,
    "force push the source branch": 0.65, "click approve on the green merge request": 0.30,
    "open the pipeline tab": 0.25, "click merge": 0.40, "revert the approval": 0.45,
    "close the merge request": 0.30,
}


def _real_provider_or_none():
    base_url = os.environ.get("AGENTD_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        return None
    return OpenAICompatProvider(ProviderSpec(
        label=os.environ.get("AGENTD_MODEL", "remote"),
        model=os.environ.get("AGENTD_MODEL", "gpt-4o-mini"),
        base_url=base_url,
        api_key=os.environ.get("AGENTD_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        temperature=float(os.environ.get("AGENTD_TEMPERATURE", "0.7")),
        decode_mode=DecodeMode(os.environ["AGENTD_DECODE_MODE"]) if os.environ.get("AGENTD_DECODE_MODE") else None,
    ))


def build_loop(store: Store, *, enabled: bool, beta: float, noise: float, seed: int,
               max_steps: int, council: bool, provider=None):
    kernel = JitRLKernel(store=store, gamma=float(os.environ.get("AGENTD_GAMMA", "0.95")),
                         beta=beta, enabled=enabled, seed=seed,
                         exploration_prob=float(os.environ.get("AGENTD_EPSILON", "0.05")))
    bus = ToolBus()
    register_builtins(bus)
    model = provider or SyntheticModel(SYNTHETIC_WEIGHTS, noise=noise, seed=seed)
    # Every council member must be equally weak, otherwise the measured gain is an
    # artifact of handing the council an oracle. Same weights, different luck.
    members = [model, SyntheticModel(SYNTHETIC_WEIGHTS, noise=noise, seed=seed + 1),
               SyntheticModel(SYNTHETIC_WEIGHTS, noise=noise, seed=seed + 2)]
    return AgentLoop(
        kernel, model, bus, context=ContextBuilder(),
        ledger=Ledger(price=PriceTable(input=float(os.environ.get("AGENTD_PRICE_IN", "0")),
                                       output=float(os.environ.get("AGENTD_PRICE_OUT", "0")))),
        council=Council(kernel, members, trigger=TriggerPolicy()) if council else None,
        config=LoopConfig(max_steps=max_steps, deliberate=council),
    )


def run_arm(*, enabled: bool, episodes: int, tasks, beta: float, noise: float, seed: int,
            max_steps: int, council: bool, provider=None) -> dict:
    store = Store()
    loop = build_loop(store, enabled=enabled, beta=beta, noise=noise, seed=seed,
                      max_steps=max_steps, council=council, provider=provider)
    series: dict[str, list[int]] = {}
    flat: list[int] = []
    t0 = time.time()
    for i in range(episodes):
        factory = tasks[i % len(tasks)]
        name = factory().name
        report = loop.run(factory())
        flat.append(int(bool(report.success)))
        series.setdefault(name, []).append(int(bool(report.success)))
    half = max(1, len(flat) // 2)
    return {
        "episodes": len(flat), "successes": sum(flat),
        "rate": round(sum(flat) / max(len(flat), 1), 3),
        "first_half_rate": round(sum(flat[:half]) / half, 3),
        "second_half_rate": round(sum(flat[half:]) / max(len(flat) - half, 1), 3),
        "per_task": {k: {"successes": sum(v), "episodes": len(v),
                         "rate": round(sum(v) / max(len(v), 1), 3)} for k, v in series.items()},
        "curve": flat,
        "memory_rows": store.count_steps(),
        "tokens": loop.ledger.summary(),
        "wall_seconds": round(time.time() - t0, 2),
    }


def bench(*, episodes: int = 40, tasks=None, beta: float = 1.0, noise: float = 0.22,
          seed: int = 7, max_steps: int = 6, council: bool = False,
          provider=None) -> dict:
    tasks = tasks or op_suite()
    on = run_arm(enabled=True, episodes=episodes, tasks=tasks, beta=beta, noise=noise,
                 seed=seed, max_steps=max_steps, council=council, provider=provider)
    off = run_arm(enabled=False, episodes=episodes, tasks=tasks, beta=0.0, noise=noise,
                  seed=seed, max_steps=max_steps, council=False, provider=provider)
    return {
        "model": (provider.label if provider else "synthetic (keyless demo)"),
        "episodes_per_arm": episodes, "beta": beta, "noise": noise, "seed": seed,
        "memory_on": on, "memory_off": off,
        "delta_rate": round(on["rate"] - off["rate"], 3),
        "delta_second_half": round(on["second_half_rate"] - off["second_half_rate"], 3),
        "extra_tokens_on_arm": on["tokens"]["total_tokens"] - off["tokens"]["total_tokens"],
    }


def web_bench(*, episodes: int = 30, policy: str = "heuristic", seed: int = 7,
              noise: float = 0.18, beta: float = 1.0, gamma: float = 0.95,
              epsilon: float = 0.05, headless: bool = True) -> dict:
    """Learning ON/OFF on real pages, driven by Chromium through Playwright.

    `policy="heuristic"` needs no API key; `policy="model"` uses whatever endpoint
    AGENTD_BASE_URL points at. The result says which one produced the numbers, so a
    lexical baseline can never be mistaken for a language model.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("web suite needs playwright: pip install 'agentd[browser]' "
                           "then `python -m playwright install chromium`") from exc

    from .envs.browser_env import BrowserEnv
    from .envs.web_tasks import web_suite
    from .providers.heuristic import HeuristicModel

    def make_policy():
        if policy == "heuristic":
            return HeuristicModel(seed=seed, noise=noise)
        real = _real_provider_or_none()
        if real is None:
            raise RuntimeError("policy=model needs AGENTD_BASE_URL/AGENTD_MODEL/AGENTD_API_KEY")
        return real

    tasks = web_suite()
    arms: dict[str, dict] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        for arm, enabled in (("memory_off", False), ("memory_on", True)):
            page = browser.new_page()
            store = Store()
            kernel = JitRLKernel(store=store, gamma=gamma, beta=beta if enabled else 0.0,
                                 enabled=enabled, seed=seed, exploration_prob=epsilon)
            bus = ToolBus()
            register_builtins(bus)
            loop = AgentLoop(kernel, make_policy(), bus, context=ContextBuilder(),
                             ledger=Ledger(), config=LoopConfig(max_steps=14, deliberate=False,
                                                                reflect=False))
            flat: list[int] = []
            per_task: dict[str, list[int]] = {}
            t0 = time.time()
            for i in range(episodes):
                task = tasks[i % len(tasks)]
                env = BrowserEnv(page, task)
                report = loop.run(env)
                flat.append(int(bool(report.success)))
                per_task.setdefault(task.name, []).append(int(bool(report.success)))
            half = max(1, len(flat) // 2)
            arms[arm] = {
                "episodes": len(flat), "successes": sum(flat),
                "rate": round(sum(flat) / max(len(flat), 1), 3),
                "first_half_rate": round(sum(flat[:half]) / half, 3),
                "second_half_rate": round(sum(flat[half:]) / max(len(flat) - half, 1), 3),
                "per_task": {name: {"successes": sum(v), "episodes": len(v),
                                    "rate": round(sum(v) / len(v), 3)} for name, v in per_task.items()},
                "curve": flat, "memory_rows": store.count_steps(),
                "tokens": loop.ledger.summary(), "wall_seconds": round(time.time() - t0, 2),
            }
            page.close()
        browser.close()

    on, off = arms["memory_on"], arms["memory_off"]
    return {
        "suite": "web", "policy": policy, "seed": seed, "episodes_per_arm": episodes,
        "model": f"{policy} (no language model involved)" if policy == "heuristic" else policy,
        "memory_on": on, "memory_off": off,
        "delta_rate": round(on["rate"] - off["rate"], 3),
        "delta_second_half": round(on["second_half_rate"] - off["second_half_rate"], 3),
    }


def render(result: dict) -> str:
    lines = [
        f"agentd bench — model: {result['model']}  episodes/arm: {result['episodes_per_arm']}",
        "",
    ]
    for arm, label in (("memory_off", "learning OFF (control)"), ("memory_on", "learning ON  (JitRL)")):
        block = result[arm]
        spark = "".join("█" if v else "·" for v in block["curve"])
        lines += [
            f"{label:24s} {block['successes']:3d}/{block['episodes']} ({block['rate']*100:5.1f}%)"
            f"   1st half {block['first_half_rate']*100:5.1f}%   2nd half {block['second_half_rate']*100:5.1f}%",
            f"   {spark}",
            f"   memory rows {block['memory_rows']:5d}   tokens {block['tokens']['total_tokens']:8d}"
            f"   cache hit {block['tokens']['cache_hit_pct']:5.1f}%",
        ]
    lines += [
        "",
        f"delta rate          {result['delta_rate']:+.3f}",
        f"delta second half   {result['delta_second_half']:+.3f}",
        f"extra tokens spent  {result['extra_tokens_on_arm']:+d}   (memory costs no inference; it only re-ranks)",
        "",
        "per task (learning ON):",
    ]
    for name, stats in result["memory_on"]["per_task"].items():
        lines.append(f"  - {name:<26} {stats['successes']}/{stats['episodes']} ({stats['rate']*100:.1f}%)")
    return "\n".join(lines)
