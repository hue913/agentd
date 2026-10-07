"""Council: conditional triggering, learned member weighting, preserved dissent."""

from __future__ import annotations

import pytest

from agentd.council import Council, TriggerPolicy
from agentd.kernel import JitRLKernel, Store
from agentd.providers import MockProvider, ProviderSpec

RESTART = "systemctl restart nginx"
REBOOT = "reboot the whole host"
STATE = "ssh session on app-01, nginx is down and the healthcheck is failing"
CANDIDATES = [RESTART, REBOOT]


def member(label: str, weights: dict[str, float], critiques: list[str] | None = None) -> MockProvider:
    return MockProvider(ProviderSpec(label=label, model=label, kind="mock"),
                        weights=weights, critiques=critiques)


def kernel_with(store: Store | None = None) -> JitRLKernel:
    return JitRLKernel(store=store or Store(), gamma=0.5, beta=1.0, seed=5)


def train(store: Store, task: str = "fix nginx", state: str = STATE,
          actions: tuple[str, ...] = (RESTART,), scope: str = "app-01") -> None:
    kernel = JitRLKernel(store=store, gamma=0.5, seed=4)
    for _ in range(2):
        rec = kernel.begin(task, task)
        rec.record(state, REBOOT, z=0.2, reward=0.0, scope=scope)
        for action in actions:
            rec.record(state, action, z=0.5, reward=1.0, scope=scope)
        rec.finish(success=True, score=1.0, analysis="restart, never reboot")


# -- triggering -----------------------------------------------------------
def test_confident_primary_skips_deliberation_entirely():
    calls: list = []
    a = member("a", {RESTART: 0.9, REBOOT: 0.1}, )
    a.calls = calls
    council = Council(kernel_with(), [a], trigger=TriggerPolicy(margin_threshold=0.12))
    outcome = council.decide(STATE, CANDIDATES)
    assert outcome.used is False and "confident" in outcome.reason
    assert len(calls) == 1                      # no draft fan-out, no critique
    assert outcome.decision.chosen_action == RESTART


def test_thin_margin_triggers_the_council():
    a = member("a", {RESTART: 0.51, REBOOT: 0.49})
    b = member("b", {RESTART: 0.4, REBOOT: 0.6})
    outcome = Council(kernel_with(), [a, b]).decide(STATE, CANDIDATES)
    assert outcome.used and "margin" in outcome.reason
    assert {p.member for p in outcome.proposals} == {"a", "b"}
    assert outcome.usage.calls >= 2


def test_dangerous_risk_triggers_even_with_a_confident_primary():
    a = member("a", {RESTART: 0.9, REBOOT: 0.1})
    b = member("b", {RESTART: 0.1, REBOOT: 0.9})
    outcome = Council(kernel_with(), [a, b]).decide(STATE, CANDIDATES, risk="dangerous")
    assert outcome.used and outcome.reason == "action risk=dangerous"


def test_pinned_task_triggers():
    a = member("a", {RESTART: 0.9, REBOOT: 0.1})
    b = member("b", {RESTART: 0.9, REBOOT: 0.1})
    outcome = Council(kernel_with(), [a, b],
                      trigger=TriggerPolicy(pin_tasks=("fix nginx",))).decide(
                          STATE, CANDIDATES, task="fix nginx")
    assert outcome.used and "pinned" in outcome.reason


# -- learned routing ------------------------------------------------------
def test_reliability_history_decides_which_model_is_believed():
    store = Store()
    council = Council(kernel_with(store),
                      [member("a", {RESTART: 0.9, REBOOT: 0.1}),
                       member("b", {RESTART: 0.1, REBOOT: 0.9})])

    # a is right most of the time: 9 wins / 10 trials -> (9+1)/(10+2) = 0.83
    for _ in range(9):
        store.record_member_outcome("a", "app-01", won=True, ret=1.0)
    store.record_member_outcome("a", "app-01", won=False, ret=0.0)
    # b is usually wrong: 2 wins / 10 trials -> (2+1)/(10+2) = 0.25
    for _ in range(2):
        store.record_member_outcome("b", "app-01", won=True, ret=1.0)
    for _ in range(8):
        store.record_member_outcome("b", "app-01", won=False, ret=0.0)

    outcome = council.decide(STATE, CANDIDATES, scope="app-01", risk="dangerous")
    assert outcome.weights["a"] == pytest.approx(0.8333, abs=1e-3)
    assert outcome.weights["b"] == pytest.approx(0.25, abs=1e-3)
    assert outcome.decision.chosen_action == RESTART

    flipped = Store()
    for _ in range(9):
        flipped.record_member_outcome("b", "app-01", won=True, ret=1.0)
    flipped.record_member_outcome("b", "app-01", won=False, ret=0.0)
    for _ in range(2):
        flipped.record_member_outcome("a", "app-01", won=True, ret=1.0)
    for _ in range(8):
        flipped.record_member_outcome("a", "app-01", won=False, ret=0.0)
    outcome2 = Council(kernel_with(flipped),
                      [member("a", {RESTART: 0.9, REBOOT: 0.1}),
                       member("b", {RESTART: 0.1, REBOOT: 0.9})]).decide(
                          STATE, CANDIDATES, scope="app-01", risk="dangerous")
    assert outcome2.decision.chosen_action == REBOOT


def test_memory_still_beats_the_council_average():
    """The kernel is the final say: retrieved experience outranks a vote."""
    store = Store()
    train(store)
    council = Council(kernel_with(store),
                      [member("a", {RESTART: 0.45, REBOOT: 0.55}),
                       member("b", {RESTART: 0.40, REBOOT: 0.60})])
    outcome = council.decide(STATE, CANDIDATES, scope="app-01", risk="dangerous")
    assert outcome.decision.chosen_action == RESTART, "memory should flip the plurality vote"


# -- dissent --------------------------------------------------------------
def test_split_vote_is_recorded_as_a_risk_not_merged_away():
    store = Store()
    episode_id = store.start_episode("fix nginx", "fix nginx")
    council = Council(kernel_with(store),
                      [member("a", {RESTART: 0.9, REBOOT: 0.1}),
                       member("b", {RESTART: 0.1, REBOOT: 0.9})])
    outcome = council.decide(STATE, CANDIDATES, scope="app-01", risk="dangerous",
                             task="fix nginx", episode_id=episode_id)
    assert outcome.consensus is False
    split = [r for r in outcome.risks if r["stance"] == "split"]
    assert split and RESTART in split[0]["note"] and REBOOT in split[0]["note"]
    assert store.risks_for(episode_id), "dissent must be persisted for the UI"


def test_objection_lowers_the_targeted_action():
    critic = json_objection(target="a", severity="high", note="reboot masks the real failure")
    council = Council(kernel_with(),
                      [member("a", {RESTART: 0.7, REBOOT: 0.3}),
                       member("b", {RESTART: 0.3, REBOOT: 0.7}, critiques=[critic])])
    outcome = council.decide(STATE, CANDIDATES, scope="app-01", risk="dangerous")
    assert outcome.objections and outcome.objections[0].stance == "object"
    assert outcome.decision.chosen_action == REBOOT


def test_self_critique_is_ignored():
    council = Council(kernel_with(),
                      [member("a", {RESTART: 0.7, REBOOT: 0.3},
                              critiques=[json_objection(target="a", severity="high")]),
                       member("b", {RESTART: 0.3, REBOOT: 0.7})])
    outcome = council.decide(STATE, CANDIDATES, scope="app-01", risk="dangerous")
    assert [o for o in outcome.objections if o.member == o.target_member] == []


def test_providers_without_text_channel_still_deliberate():
    class Silent(MockProvider):
        def text(self, prompt: str):
            raise NotImplementedError("no free-text channel")

    a = Silent(ProviderSpec(label="a", model="a", kind="mock"), weights={RESTART: 0.9, REBOOT: 0.1})
    b = member("b", {RESTART: 0.2, REBOOT: 0.8})
    outcome = Council(kernel_with(), [a, b]).decide(STATE, CANDIDATES, risk="dangerous")
    assert outcome.used and len(outcome.proposals) == 2 and outcome.objections == []


def test_credit_updates_reliability():
    store = Store()
    council = Council(kernel_with(store),
                      [member("a", {RESTART: 0.9, REBOOT: 0.05}),
                       member("b", {RESTART: 0.4, REBOOT: 0.6})])
    outcome = council.decide(STATE, CANDIDATES, scope="app-01", risk="dangerous")
    assert outcome.decision.chosen_action == RESTART
    council.credit(outcome, success=True)

    weights = store.member_reliability("app-01")
    assert set(weights) == {"a"}, "only the executed proposal has evidence"
    assert weights["a"] == pytest.approx(2 / 3)


def test_dissent_is_neither_rewarded_nor_punished():
    store = Store()
    council = Council(kernel_with(store),
                      [member("a", {RESTART: 0.9, REBOOT: 0.05}),
                       member("b", {RESTART: 0.4, REBOOT: 0.6})])
    outcome = council.decide(STATE, CANDIDATES, scope="app-01", risk="dangerous")
    council.credit(outcome, success=False)
    weights = store.member_reliability("app-01")
    assert set(weights) == {"a"}
    assert weights["a"] == pytest.approx(1 / 3)


def json_objection(target: str, severity: str = "medium", note: str = "bad idea") -> str:
    import json

    return json.dumps({"objections": [{"target_member": target, "stance": "object",
                                       "note": note, "severity": severity}]})
