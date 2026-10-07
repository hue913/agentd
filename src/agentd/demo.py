"""Keyless self-demo: watch test-time RL work without training anything.

`agentd demo` runs the same decision loop twice on a synthetic task family —
memory off (the control arm) and memory on — against a deliberately noisy fake
model. The noisy model occasionally stumbles onto the right action; the kernel
is what turns those lucky hits into a habit. No API key, no network.
"""

from __future__ import annotations

import random

from .kernel import JitRLKernel, Store
from .providers.base import Choice, DecodeMode, Provider, ProviderSpec, ScoreSpace, Usage

# Each task is a chain of states. `options` are the numbered candidates the model
# sees; `right` is what actually advances the task. The fake model below prefers
# the wrong option on average, so any improvement must come from memory.
TASKS: dict[str, dict] = {
    "buy-cheap-laptop": {
        "steps": [
            {
                "state": "one-stop shop, laptop results page, sort and filter controls visible",
                "options": ["click the cheapest laptop row", "open the site settings page", "scroll down once"],
                "right": "click the cheapest laptop row",
                "base": {"click the cheapest laptop row": 0.35, "open the site settings page": 0.30, "scroll down once": 0.45},
            },
            {
                "state": "laptop detail page with an Add to cart button and a price",
                "options": ["click add to cart", "write a product review", "open the site settings page"],
                "right": "click add to cart",
                "base": {"click add to cart": 0.40, "write a product review": 0.25, "open the site settings page": 0.50},
            },
            {
                "state": "shopping cart page showing one item and a Proceed to Checkout button",
                "options": ["proceed to checkout", "empty the cart", "keep shopping"],
                "right": "proceed to checkout",
                "base": {"proceed to checkout": 0.45, "empty the cart": 0.30, "keep shopping": 0.40},
            },
        ],
    },
    "restart-failing-service": {
        "steps": [
            {
                "state": "ssh session on app-01, nginx reported as down by the healthcheck",
                "options": ["systemctl restart nginx", "tail the nginx error log", "reboot the whole host"],
                "right": "systemctl restart nginx",
                "base": {"systemctl restart nginx": 0.30, "tail the nginx error log": 0.50, "reboot the whole host": 0.25},
            },
            {
                "state": "nginx restarted on app-01, healthcheck endpoint pending",
                "options": ["curl the healthcheck endpoint", "edit the config file", "run apt upgrade"],
                "right": "curl the healthcheck endpoint",
                "base": {"curl the healthcheck endpoint": 0.40, "edit the config file": 0.35, "run apt upgrade": 0.30},
            },
        ],
    },
}


class NoisyModel(Provider):
    """A weak-but-honest stand-in: prefers the wrong option, sometimes guesses right."""

    def __init__(self, noise: float = 0.22, seed: int = 7):
        super().__init__(ProviderSpec(label="noisy-fake", model="noisy-fake", kind="mock"))
        self.rng = random.Random(seed)
        self.noise = noise

    def choose(self, system: str, user: str, candidates: list[str]) -> Choice:
        base = None
        for task in TASKS.values():
            for step in task["steps"]:
                if step["state"] in user and set(step["options"]) == set(candidates):
                    base = step["base"]
        z = {c: (base or {}).get(c, 0.3) + self.rng.gauss(0.0, self.noise) for c in candidates}
        return Choice(z=z, raw_text="", mode=DecodeMode.TOKEN, score_space=ScoreSpace.LOGPROB,
                      usage=Usage(prompt_tokens=len(user) // 4, completion_tokens=1))


def run_task(kernel: JitRLKernel, model: Provider, task_name: str) -> bool:
    task = TASKS[task_name]
    rec = kernel.begin(task_name, task_name)
    prompt_hint = "Answer with exactly one digit."
    for i, step in enumerate(task["steps"]):
        decision = kernel.decide(step["state"], step["options"], model.choose(prompt_hint, step["state"], step["options"]).z,
                                 mode="token")
        chosen = decision.chosen_action
        success_step = chosen == step["right"]
        rec.record(step["state"], chosen, z=decision.rerank.chosen.z, adv=decision.rerank.chosen.advantage,
                   z_prime=decision.rerank.chosen.z_prime, chosen_index=decision.rerank.chosen_index,
                   reward=1.0 if (success_step and i == len(task["steps"]) - 1) else 0.0)
        if not success_step:
            rec.finish(success=False, score=0.0, analysis=f"derailed at step {i + 1} by choosing '{chosen}'")
            return False
    rec.finish(success=True, score=1.0, analysis="task completed; keep the actions that advanced it")
    return True


def roll(arm: str, episodes: int, seed: int) -> list[int]:
    # The kernel seed matters: epsilon-exploration draws from this rng, so an
    # unseeded kernel makes the headline curve unreproducible run to run.
    kernel = JitRLKernel(store=Store(), beta=1.0 if arm == "on" else 0.0,
                         enabled=(arm == "on"), gamma=0.5, seed=seed)
    model = NoisyModel(seed=seed)
    names = list(TASKS)
    outcomes: list[int] = []
    for e in range(episodes):
        task_name = names[e % len(names)]
        outcomes.append(int(run_task(kernel, model, task_name)))
    return outcomes


def moving(seq: list[int], w: int = 5) -> list[float]:
    return [round(sum(seq[max(0, i - w + 1) : i + 1]) / len(seq[max(0, i - w + 1) : i + 1]), 3) for i in range(len(seq))]


def demo(episodes: int = 40, seed: int = 7) -> dict:
    off = roll("off", episodes, seed)
    on = roll("on", episodes, seed)
    half = max(1, episodes // 2)
    return {
        "episodes": episodes,
        "memory_off": {"successes": sum(off), "first_half": sum(off[:half]) / half,
                       "second_half": sum(off[half:]) / max(1, episodes - half), "curve": off},
        "memory_on": {"successes": sum(on), "first_half": sum(on[:half]) / half,
                      "second_half": sum(on[half:]) / max(1, episodes - half), "curve": on},
        "moving_avg_off": moving(off),
        "moving_avg_on": moving(on),
    }


def render(result: dict) -> str:
    lines = [
        "agentd demo — JitRL-style test-time RL, no API key, no training",
        f"tasks: {len(TASKS)} families, episodes per arm: {result['episodes']}",
        "",
    ]
    for arm, key in (("memory OFF (control)", "memory_off"), ("memory ON  (JitRL)", "memory_on")):
        block = result[key]
        spark = "".join("█" if v else "·" for v in block["curve"])
        lines += [
            f"{arm:22s} successes {block['successes']:2d}/{result['episodes']}"
            f"   first half {block['first_half']:.2f}   second half {block['second_half']:.2f}",
            f"   {spark}",
        ]
    lines += [
        "",
        "reading: the fake model prefers the wrong option on average, so a flat",
        "control arm is expected. The learning arm only improves by retrieving",
        "past (state, action, return) triplets and biasing the next token by the",
        "estimated advantage: z' = z + beta * A.",
    ]
    return "\n".join(lines)
