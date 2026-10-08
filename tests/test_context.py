"""Token-thriftiness machinery: stable prefix, compaction, honest ledger, routing."""

from __future__ import annotations

from agentd.context import (
    ContextBuilder, Ledger, PriceTable, TierRouter, compact_observation, estimate_tokens,
)
from agentd.providers import Usage

AXTREE = """RootWebArea One Stop Shop
  navigation
    link Cart
    link Sign In
  main
    button Add to cart
    button Add to cart
    button Add to cart
    img data:image/png;base64,AAAABBBBCCCCDDDDeeee0000
    text 0123456789abcdef0123456789abcdef
  contentinfo
    link Cart
"""


def test_compaction_removes_noise_and_duplicates():
    compacted, stats = compact_observation(AXTREE, budget_chars=5_000)
    assert compacted.count("button Add to cart") == 1
    assert "data:image" not in compacted and "[image omitted]" in compacted
    assert "0123456789abcdef0123456789abcdef" not in compacted and "[id]" in compacted
    assert stats["dropped_duplicate_lines"] >= 1
    assert stats["saved_pct"] > 0 and stats["kept_chars"] < stats["orig_chars"]


def test_truncation_keeps_both_ends():
    long_text = "\n".join(f"line {i} some distinct content here" for i in range(600))
    compacted, stats = compact_observation(long_text, budget_chars=800)
    assert len(compacted) <= 900
    assert compacted.startswith("line 0")
    assert "line 599" in compacted
    assert "chars omitted" in compacted
    assert stats["kept_chars"] < stats["orig_chars"]


def test_empty_and_oversized_inputs_are_safe():
    assert compact_observation("") == ("", {"orig_chars": 0, "kept_chars": 0, "saved_pct": 0.0})
    kept, stats = compact_observation("tiny", budget_chars=100)
    assert kept == "tiny" and stats["saved_pct"] == 0.0


def test_prefix_is_stable_across_steps_so_caching_can_work():
    builder = ContextBuilder()
    common = dict(instructions="Fix the server without destroying data.", catalog="ssh.exec: run a command")
    first = builder.build(state="nginx is down on app-01", history=[], recall=[], **common)
    second = builder.build(state="nginx restarted, healthcheck pending",
                           history=["ran systemctl restart nginx"], recall=["restart worked before"],
                           **common)
    assert first.prefix_hash == second.prefix_hash
    assert first.system == second.system
    # volatility order: history -> state -> recall
    assert second.user.index("recent steps") < second.user.index("current state") < second.user.index("what worked here before")
    assert first.est_prompt_tokens == estimate_tokens(first.system + first.user)


def test_history_char_budget_bounds_the_prompt():
    builder = ContextBuilder(history_budget_chars=400)
    plan = builder.build(instructions="i", catalog="c", recall=[],
                         history=[f"step {i} did something distinct and long enough to add up" for i in range(50)],
                         state="s")
    assert "earlier steps omitted" in plan.user
    assert len(plan.user) < 1200


def test_history_budget_never_triggers_for_normal_episodes():
    builder = ContextBuilder()
    plan = builder.build(instructions="i", catalog="c", recall=[],
                         history=[f"chose action {i} -> ok" for i in range(12)], state="s")
    assert "earlier steps omitted" not in plan.user
    assert "1. chose action 0" in plan.user


def test_ledger_reports_cache_hits_and_costs():
    ledger = Ledger(price=PriceTable(input=2.0, output=8.0, cache_read=0.2))
    ledger.record(Usage(prompt_tokens=10_000, completion_tokens=50, cached_tokens=8_000, calls=1))
    row = ledger.per_step[0]
    assert row["cache_hit_pct"] == 80.0
    # 2000 fresh * 2/M + 8000 cached * 0.2/M + 50 * 8/M
    assert row["usd"] == round((2000 * 2 + 8000 * 0.2 + 50 * 8) / 1e6, 6)

    summary = ledger.summary()
    assert summary["prompt_tokens"] == 10_000 and summary["cached_tokens"] == 8_000
    assert summary["total_tokens"] == 10_050 and summary["calls"] == 1


def test_router_escalates_only_when_needed():
    router = TierRouter(cheap="qwen-small", strong="gemini", escalate_margin=0.15)
    assert router.pick({"a": 0.8, "b": 0.1}) == "qwen-small"
    assert router.pick({"a": 0.5, "b": 0.45}) == "gemini"
    assert router.pick({"a": 0.9, "b": 0.05}, risk="dangerous") == "gemini"
    assert router.total_steps == 3 and router.escalations == 2


def test_router_without_two_tiers_is_a_no_op():
    assert TierRouter(cheap="only").pick({"a": 0.5, "b": 0.5}) == "only"
    assert TierRouter().pick({"a": 0.5}) == ""
