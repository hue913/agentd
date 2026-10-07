"""Kernel behaviour: does retrieved experience actually flip the decision?"""

from __future__ import annotations

import math
import random

import pytest

from agentd.kernel import (
    JitRLKernel, Retriever, Store, estimate, normalize_action, ngrams, normalize_state, rerank,
    tokenize,
)

STATE = "shopping product page with an Add to cart button and a search box at the top"
SIMILAR_STATE = "shopping product page with an Add to cart button and a filter panel"
DISTINCT_STATE = "gitlab merge request list with an approve button"

CLICK = "click [add to cart]"
SCROLL = "scroll down"
NEWBIE = "type laptop into search box"


def seed_experience(kernel: JitRLKernel, scope: str = "") -> None:
    """Two past episodes where clicking added-to-cart succeeded and scrolling did not.

    Reward is sparse and lands on the step that ends the task, so the discounted
    return at gamma=0.5 is scroll=0.5, click=1.0.
    """
    for state in (STATE, SIMILAR_STATE):
        rec = kernel.begin("buy item", "buy item")
        rec.record(state, SCROLL, z=0.9, reward=0.0, scope=scope)
        rec.record(state, CLICK, z=0.4, reward=1.0, scope=scope)
        rec.finish(success=True, score=1.0, analysis="clicking the cart button finished the task")


def test_similarity_prefers_lexically_close_states():
    store = Store()
    seed_experience(JitRLKernel(store=store, gamma=0.5))
    r = Retriever(store, ngram=2, top_k=4)
    near = r.neighbors(STATE)
    far = r.neighbors(DISTINCT_STATE)
    assert near and near[0].sim > 0.5
    assert (not far) or far[0].sim < near[0].sim


def test_advantage_uses_recomputed_baseline_like_the_reference():
    store = Store()
    seed_experience(JitRLKernel(store=store, gamma=0.5))
    neighbors = Retriever(store).neighbors(STATE)
    adv = estimate([CLICK, SCROLL, NEWBIE], neighbors, exploration_prob=0.0)

    assert adv.baseline_before == pytest.approx(0.75)
    # the unseen action pinned to 0 pulls the baseline down to 0.5
    assert adv.baseline == pytest.approx(0.5)
    assert adv.raw[CLICK] == pytest.approx(0.5)
    assert adv.raw[SCROLL] == pytest.approx(0.0)
    assert adv.raw[NEWBIE] == pytest.approx(-0.5)
    assert adv.normalized[CLICK] == pytest.approx(1.0)


def test_exploration_is_stochastic_not_always_on():
    store = Store()
    seed_experience(JitRLKernel(store=store, gamma=0.5))
    neighbors = Retriever(store).neighbors(STATE)

    never = estimate([CLICK, SCROLL, NEWBIE], neighbors, exploration_prob=0.0, rng=random.Random(1))
    assert never.explored == []
    assert never.action_means[NEWBIE] == 0.0

    always = estimate([CLICK, SCROLL, NEWBIE], neighbors, exploration_prob=1.0, alpha=1.0,
                      rng=random.Random(1))
    assert always.explored == [normalize_action(NEWBIE)]
    # baseline_before 0.75 + alpha/count(4) = 1.0 -> the untried action is worth trying
    assert always.action_means[NEWBIE] == pytest.approx(1.0)
    assert always.raw[NEWBIE] > 0


def test_scope_restricts_which_history_counts():
    store = Store()
    seed_experience(JitRLKernel(store=store, gamma=0.5), scope="app-01:/var/log")
    neighbors = Retriever(store).neighbors(STATE)
    assert neighbors and all(n.scope == "app-01:/var/log" for n in neighbors)

    same = estimate([CLICK, SCROLL], neighbors, exploration_prob=0.0, scope="app-01:/var/log")
    other = estimate([CLICK, SCROLL], neighbors, exploration_prob=0.0, scope="gitlab:/repo")
    assert same.baseline == pytest.approx(0.75)
    # no neighbour shares the scope, so estimation falls back to the full set
    assert other.baseline == pytest.approx(0.75)
    assert same.raw[CLICK] == other.raw[CLICK]


def test_memory_on_overrides_a_wrong_but_confident_model():
    store = Store()
    kernel = JitRLKernel(store=store, beta=1.0, seed=3)
    seed_experience(kernel)
    decision = kernel.decide(STATE, [CLICK, SCROLL], z={CLICK: 0.4, SCROLL: 0.9})
    assert decision.chosen_action == CLICK
    assert decision.rerank.mode == "token"


def test_beta_zero_is_the_control_arm_and_keeps_the_model_choice():
    store = Store()
    kernel = JitRLKernel(store=store, beta=0.0, seed=3)
    seed_experience(kernel)
    decision = kernel.decide(STATE, [CLICK, SCROLL], z={CLICK: 0.4, SCROLL: 0.9})
    assert decision.chosen_action == SCROLL


def test_disabled_kernel_never_consults_memory():
    store = Store()
    kernel = JitRLKernel(store=store, enabled=False)
    seed_experience(kernel)
    decision = kernel.decide(STATE, [CLICK, SCROLL], z={CLICK: 0.4, SCROLL: 0.9})
    assert decision.neighbors == []
    assert decision.chosen_action == SCROLL


def test_discounted_return_backpropagates():
    store = Store()
    kernel = JitRLKernel(store=store, gamma=0.5)
    rec = kernel.begin("t", "t")
    rec.record("s0", "a0", reward=0.0)
    rec.record("s1", "a1", reward=0.0)
    rec.record("s2", "a2", reward=1.0)
    rec.finish(success=True, score=1.0)
    steps = store.steps_for_episode(rec.episode_id)
    assert [s.ret for s in steps] == pytest.approx([0.25, 0.5, 1.0])


def test_gamma_default_matches_webarena_not_jericho():
    assert JitRLKernel(store=Store()).gamma == 0.95


def test_wipe_removes_everything_for_the_control_condition():
    store = Store()
    kernel = JitRLKernel(store=store, gamma=0.5)
    seed_experience(kernel)
    assert store.count_steps() > 0
    store.wipe()
    assert store.count_steps() == 0
    assert kernel.decide(STATE, [CLICK, SCROLL], z={CLICK: 0.4, SCROLL: 0.9}).chosen_action == SCROLL


def test_llm_analysis_is_recallable_per_task():
    store = Store()
    kernel = JitRLKernel(store=store, gamma=0.5)
    rec = kernel.begin("buy item", "buy item")
    rec.record(STATE, CLICK, reward=1.0)
    rec.finish(success=True, score=1.0, analysis="avoid the login modal next time")
    assert kernel.analyses_for("buy item") == ["avoid the login modal next time"]
    assert kernel.analyses_for("other task") == []


def test_state_normalisation_strips_volatile_element_ids():
    a = normalize_state("button [ref=12] says Buy")
    b = normalize_state("button [ref=9999] says Buy")
    assert a == b
    assert ngrams(tokenize(a), 2) == ngrams(tokenize(b), 2)


def test_rerank_with_no_candidates_raises():
    with pytest.raises(ValueError):
        rerank([], {}, {})


def test_probability_space_matches_reference_formula():
    logprob = math.log(0.25)
    result = rerank(["do a", "do b"], z={"do a": math.exp(logprob), "do b": 0.6},
                     advantages={"do a": 1.0, "do b": -1.0}, beta=1.0)
    assert result.chosen_index == 1
    assert result.options[0].z_prime == pytest.approx(1.25)
    assert result.options[1].z_prime == pytest.approx(-0.4)
