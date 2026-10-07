"""The loop end to end: does an agent with memory get better at the same task?"""

from __future__ import annotations

import random

import pytest

from agentd.context import ContextBuilder, Ledger
from agentd.council import Council, TriggerPolicy
from agentd.envs.ops_tasks import approve_merge_request, nginx_down, zzz_never_ends
from agentd.kernel import JitRLKernel, Store
from agentd.loop import AgentLoop, LoopConfig
from agentd.providers import MockProvider, ProviderSpec
from agentd.toolbus import CallContext, ToolBus, register_builtins

# A deliberately bad operator model: it prefers rebooting hosts and force pushing.
WEIGHTS = {
    "reboot the whole host": 0.70, "systemctl restart nginx": 0.30,
    "tail the nginx error log": 0.20,
    "run apt upgrade": 0.60, "curl the healthcheck endpoint": 0.35,
    "edit the nginx config file": 0.20,
    "force push the source branch": 0.65, "click approve on the green merge request": 0.30,
    "open the pipeline tab": 0.25,
    "click merge": 0.40, "revert the approval": 0.45, "close the merge request": 0.30,
    "look around": 0.5, "look around again": 0.4,
}


class SamplingMock(MockProvider):
    """A mock that *samples* like a real model would at temperature > 0.

    This matters: with a deterministic policy and zero recorded successes the
    advantage landscape can lock onto a stalling action forever (see
    test_deterministic_policy_locks_out_without_sampling). JitRL's published path
    relies on model sampling to produce the first successes; the adaptive
    exploration function in the reference code is commented out.
    """

    def __init__(self, spec, weights, noise: float = 0.22, seed: int = 7, critiques=None):
        super().__init__(spec, weights=weights, critiques=critiques)
        self.rng = random.Random(seed)
        self.noise = noise

    def choose(self, system: str, user: str, candidates: list[str]):
        choice = super().choose(system, user, candidates)
        choice.z = {k: v + self.rng.gauss(0.0, self.noise) for k, v in choice.z.items()}
        return choice


def make_loop(store: Store | None = None, *, enabled: bool = True, beta: float = 1.0,
              max_steps: int = 6, council: bool = False, exploration_prob: float = 0.05,
              step_cost: float = 0.03, noise: float = 0.22, gamma: float = 0.95,
              track_credit: bool = False):
    store = store or Store()
    kernel = JitRLKernel(store=store, gamma=gamma, beta=beta, enabled=enabled, seed=11,
                         exploration_prob=exploration_prob, track_credit=track_credit)
    bus = ToolBus()
    register_builtins(bus)
    critiques = ["restart nginx and verify with the healthcheck; never reboot the host"] * 40
    provider = SamplingMock(ProviderSpec(label="weak", model="weak", kind="mock"), WEIGHTS,
                            noise=noise, seed=7, critiques=critiques)
    members = [
        SamplingMock(ProviderSpec(label="a", model="a", kind="mock"), WEIGHTS, noise=noise, seed=21),
        SamplingMock(ProviderSpec(label="b", model="b", kind="mock"),
                     {**WEIGHTS, "reboot the whole host": 0.1, "systemctl restart nginx": 0.8},
                     noise=noise, seed=33),
    ]
    config = LoopConfig(max_steps=max_steps, deliberate=council, step_cost=step_cost)
    return AgentLoop(kernel, provider, bus, context=ContextBuilder(), ledger=Ledger(),
                     council=Council(kernel, members, trigger=TriggerPolicy()) if council else None,
                     config=config), store


def test_deterministic_policy_locks_out_without_sampling():
    """Documents two failure modes so they cannot be tuned away silently.

    (a) with step_cost=0, "never terminate" scores better than "act and fail";
    (b) with a temperature-0 policy and zero recorded successes, the advantage
        landscape self-reinforces onto the stalling action.
    """
    loop, store = make_loop(step_cost=0.0, noise=0.0)
    reports = [loop.run(nginx_down()) for _ in range(4)]
    assert all(r.success is False for r in reports)
    assert any(len(r.steps) == loop.config.max_steps for r in reports), \
        "the agent should have discovered the non-terminating action"


def test_without_memory_the_bad_model_keeps_failing():
    loop, store = make_loop(enabled=False)
    outcomes = [loop.run(nginx_down()).success for _ in range(6)]
    assert not any(outcomes), outcomes


def test_memory_turns_repeated_failures_into_success():
    loop, store = make_loop()
    results = [loop.run(nginx_down()).success for _ in range(8)]
    assert results[0] is False
    assert any(results), results
    assert results[-1] is True, results
    learned = store.member_reliability("")      # no council here
    assert store.count_steps() > 0
    assert loop.ledger.summary()["calls"] >= 2


def test_learning_needs_an_initial_success():
    """The stall boundary, stated against the credit-off arm.

    Credit accounting (track_credit=True) is deliberately NOT used here. The
    mechanism under test is JitRL's advantage term alone: with a weak
    deterministic model that never records a positive return, there is nothing
    to reinforce. See
    test_credit_accounting_changes_the_stall_dynamics for the other arm.
    """
    """Known boundary of the mechanism, pinned so it cannot be forgotten.

    The kernel's exploration term applies only to actions that have never been
    tried. On a family where the base model prefers the destructive action by a
    wide margin and every action eventually gets tried, no positive return is ever
    recorded, so there is nothing for the advantage term to reinforce: the agent
    stalls instead of improving. JitRL's published numbers come from a strong model
    that succeeds early; a weak/deterministic one may never get that first hit.
    """
    loop, store = make_loop()
    for _ in range(4):
        loop.run(nginx_down())

    mr = [loop.run(approve_merge_request()).success for _ in range(24)]
    assert not any(mr), mr
    stalled = [s.chosen_action for r in [loop.run(approve_merge_request())] for s in r.steps]
    assert "open the pipeline tab" in stalled or "force push the source branch" in stalled
    assert store.recent_analyses("approve-mr"), "failure is still reflected on in words"


def test_trace_records_the_bias_not_just_the_choice():
    loop, _ = make_loop()
    loop.run(nginx_down())
    report = loop.run(nginx_down())
    row = report.steps[0].options[0]
    assert {"index", "action", "z", "advantage", "z_prime", "chosen"} <= set(row)
    assert any(opt.get("advantage") for opt in report.steps[0].options)
    assert report.steps[0].retrieved > 0


def test_reflection_is_written_back_and_recalled():
    loop, store = make_loop()
    for _ in range(3):
        loop.run(nginx_down())
    analyses = store.recent_analyses("nginx-down")
    assert analyses, "episodes must store a verbal self-critique"
    assert any("reboot" in a or "restart" in a for a in analyses)


def test_step_limit_stops_a_non_terminating_task():
    loop, _ = make_loop(max_steps=4)
    report = loop.run(zzz_never_ends())
    assert report.success is False
    assert "step limit" in report.stopped_by
    assert len(report.steps) == 4


def test_disabled_kernel_still_records_experience():
    loop, store = make_loop(enabled=False)
    loop.run(nginx_down())
    assert store.count_steps() > 0, "the control arm must not contaminate memory, only ignore it"


def test_council_path_is_taken_and_credited_at_the_end():
    loop, store = make_loop(council=True)
    report = loop.run(nginx_down())
    assert any("+council" in step.mode for step in report.steps), [s.mode for s in report.steps]
    assert store.member_reliability("app-01"), "council members need learned weights after an episode"


def test_no_candidates_fails_loudly_but_does_not_crash():
    loop, _ = make_loop()
    task = nginx_down()
    task.chain[0].options = []
    report = loop.run(task)
    assert report.success is False
    assert "candidate" in report.stopped_by.lower()


def test_control_arm_on_off_same_random_stream():
    """The headline comparison: same weak model, same sampling stream, only the
    memory differs. This is the number the README shows."""
    episodes = 20
    on_loop, on_store = make_loop()
    off_loop, off_store = make_loop(enabled=False)

    on = [on_loop.run(nginx_down()).success for _ in range(episodes)]
    off = [off_loop.run(nginx_down()).success for _ in range(episodes)]

    assert sum(off) <= 3, f"control arm should not improve: {off}"
    assert sum(on) >= sum(off) + 5, (sum(on), sum(off))
    assert on[-8:].count(True) >= 6, on
    assert on_store.count_steps() > 0 and off_store.count_steps() > 0


def test_beta_zero_equals_control_arm():
    """beta=0 must be indistinguishable from disabling memory: same decisions."""
    beta0, _ = make_loop(beta=0.0)
    off, _ = make_loop(enabled=False)
    seq_beta = [beta0.run(nginx_down()).steps[0].chosen_action for _ in range(5)]
    seq_off = [off.run(nginx_down()).steps[0].chosen_action for _ in range(5)]
    assert seq_beta and seq_beta == seq_off


def test_credit_accounting_changes_the_stall_dynamics():
    """Credit is a real feedback signal, and this is what it buys.

    Same weak model, same seed, same task family. With credit on, a step that
    was recalled and then agreed with gets a similarity boost, which surfaces
    related experience that itself succeeded -- and the family that used to
    stall now produces at least one success. That is the observable the credit
    counters exist to provide: a per-memory answer to "is memory helping",
    not just an aggregate score.
    """
    loop, store = make_loop(track_credit=True)
    for _ in range(4):
        loop.run(nginx_down())

    outcomes = [loop.run(approve_merge_request()).success for _ in range(24)]
    totals = store.credit_totals()
    assert totals["recalls"] > 0, "recall must be counted"
    assert totals["decisions"] > 0, "agreement must be judged"
    assert any(outcomes), (
        "with credit on, the previously-stalling family should produce a hit; "
        f"outcomes={outcomes} totals={totals}"
    )


def test_credit_is_neutral_before_any_outcome_exists():
    """A memory with no decision history must not be pushed around."""
    from agentd.kernel.credit import credit_factor

    assert credit_factor(0, 0) == 1.0
    assert credit_factor(0, 1) < 1.0
    assert credit_factor(1, 0) > 1.0
    # one loss cannot erase a step that has been right many times
    assert credit_factor(9, 1) > credit_factor(1, 1)
