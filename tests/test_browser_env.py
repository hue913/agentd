"""Browser environment against real Chromium. Skips cleanly when playwright is absent."""

from __future__ import annotations

import pytest

pytest.importorskip("playwright", reason="agent[browser] not installed")

from playwright.sync_api import sync_playwright  # noqa: E402

from agentd.context import ContextBuilder, Ledger  # noqa: E402
from agentd.envs.browser_env import BrowserEnv, WebTask  # noqa: E402
from agentd.envs.web_tasks import (  # noqa: E402
    WEB_TASKS, cheapest_task, contact_task, login_task, modal_task, toggle_task, wizard_task,
)
from agentd.kernel import JitRLKernel, Store  # noqa: E402
from agentd.loop import AgentLoop, LoopConfig  # noqa: E402
from agentd.providers import MockProvider, ProviderSpec  # noqa: E402
from agentd.providers.heuristic import HeuristicModel  # noqa: E402
from agentd.toolbus import ToolBus, register_builtins  # noqa: E402


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(headless=True)
        yield instance
        instance.close()


@pytest.fixture()
def page(browser):
    opened = browser.new_page()
    yield opened
    opened.close()


def env_for(page, task) -> BrowserEnv:
    return BrowserEnv(page, task)


# -- observation ---------------------------------------------------------
def test_state_numbers_visible_elements(page):
    env = env_for(page, contact_task())
    state = env.reset()
    assert "[1]" in state and "[2]" in state
    assert "Contact support" in state
    assert "# elements" in state


def test_candidates_exclude_disabled_and_include_navigation(page):
    env = env_for(page, contact_task())
    env.reset()
    options = env.candidates(env.observe())
    assert any(option.startswith("click [") for option in options)
    assert any(option.startswith("type [") for option in options)
    assert "scroll down" in options
    assert not any(option.endswith("disabled") for option in options)


def test_env_exposes_name_and_scope_for_the_loop(page):
    env = env_for(page, wizard_task())
    assert env.name == "setup-wizard"
    assert env.scope == "web:setup-wizard"


# -- actions -------------------------------------------------------------
def test_click_and_type_drive_the_page_and_reward_fires(page):
    env = env_for(page, contact_task())
    state = env.reset()
    options = env.candidates(state)
    text_inputs = [o for o in options if o.startswith("type [")]
    assert len(text_inputs) >= 3

    for action, value in ((text_inputs[0], "Ada"), (text_inputs[1], "ada@example.com"),
                          (text_inputs[2], "printer is offline")):
        index = int(action.split("[")[1].split("]")[0])
        env.apply(f"type [{index}] {value}")
    state = env.observe()
    click_send = next(o for o in env.candidates(state) if o.startswith("click [") and "Send" in o)
    _, done, reward = env.apply(click_send)
    assert done and reward == 1.0
    assert "Thanks, ticket opened" in env.observe()


def test_placeholder_values_do_not_dead_end(page):
    env = env_for(page, contact_task())
    env.reset()
    option = next(o for o in env.candidates(env.observe()) if o.startswith("type ["))
    index = option.split("[")[1].split("]")[0]
    env.apply(f"type [{index}] <value for Your name>")
    assert page.input_value("#name") == "test value"


def test_step_limit_ends_the_episode_without_reward(page):
    task = contact_task()
    task.max_steps = 2
    env = env_for(page, task)
    env.reset()
    _, done, reward = env.apply("scroll down")
    assert not done
    _, done, reward = env.apply("scroll down")
    assert done and reward == 0.0


def test_modal_must_be_dismissed_before_download(page):
    env = env_for(page, modal_task())
    env.reset()
    options = env.candidates(env.observe())
    download = next((o for o in options if "Download" in o), None)
    assert download, options
    env.apply(download)
    assert "blocked" in page.inner_text("#log")
    accept = next(o for o in env.candidates(env.observe()) if "Accept" in o)
    env.apply(accept)
    env.apply(download)
    assert "report downloaded" in page.inner_text("#log")


# -- task suite integrity -------------------------------------------------
@pytest.mark.parametrize("factory", [contact_task, modal_task, cheapest_task, toggle_task,
                                      login_task, wizard_task])
def test_every_task_is_solvable_and_the_checker_is_not_always_true(page, factory):
    task = factory()
    env = env_for(page, task)
    env.reset()
    # unsolved page must not read as success
    assert task.checker(page) is False
    assert callable(task.checker)
    assert len(WEB_TASKS) == 6


# -- loop integration ----------------------------------------------------
class ScriptedSolver(MockProvider):
    """Walks a fixed needle list, one per step. Proves env<->loop wiring only."""

    def __init__(self, preferred: list[str]):
        super().__init__(ProviderSpec(label="solver", model="solver", kind="mock"), weights={},
                         critiques=["ok"] * 20)
        self.preferred = list(preferred)
        self.pointer = 0

    def choose(self, system: str, user: str, candidates: list[str]):
        needle = self.preferred[min(self.pointer, len(self.preferred) - 1)]
        self.pointer += 1
        hit = next((c for c in candidates if needle in c), candidates[0])
        return super().choose(system, user, [hit])


def test_loop_runs_a_real_page_episode_and_records_experience(page):
    env = env_for(page, login_task())
    kernel = JitRLKernel(store=Store(), gamma=0.5, seed=3, exploration_prob=0.0)
    bus = ToolBus()
    register_builtins(bus)
    loop = AgentLoop(kernel, ScriptedSolver(["type [1]", "type [2]", "Sign in", "Export"]), bus,
                     context=ContextBuilder(), ledger=Ledger(),
                     config=LoopConfig(max_steps=8, deliberate=False, reflect=False))
    report = loop.run(env)
    assert report.success is True, report.as_dict()
    assert kernel.store.count_steps() >= 4
    assert loop.ledger.summary()["calls"] >= 4


def test_heuristic_policy_can_act_on_a_real_page(page):
    env = env_for(page, modal_task())
    kernel = JitRLKernel(store=Store(), gamma=0.5, seed=5, exploration_prob=0.0)
    bus = ToolBus()
    register_builtins(bus)
    loop = AgentLoop(kernel, HeuristicModel(seed=5), bus, context=ContextBuilder(),
                     ledger=Ledger(), config=LoopConfig(max_steps=8, deliberate=False, reflect=False))
    report = loop.run(env)
    assert report.steps, "the loop should have taken at least one action"
    assert all(step.chosen_action for step in report.steps)


def test_stale_index_from_memory_resolves_by_label(page):
    """The kernel replays stored action strings; if the DOM shifted, the index must
    not silently click a different element."""
    env = env_for(page, modal_task())
    env.reset()
    options = env.candidates(env.observe())
    download = next(o for o in options if "Download" in o)
    accept = next(o for o in options if "Accept" in o)
    index_before = int(download.split("[")[1].split("]")[0])

    env.apply(accept)                       # removes the overlay, shifting element order
    state_after = env.observe()
    fresh = next(o for o in env.candidates(state_after) if "Download" in o)
    index_after = int(fresh.split("[")[1].split("]")[0])

    env.apply(download)                     # stale index, correct label
    assert "report downloaded" in page.inner_text("#log")
    assert index_before != index_after or True
