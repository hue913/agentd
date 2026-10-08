"""KV-cache ordering regressions: the prefix a provider cache can actually hit.

The complaint that motivated this file: provider-side prompt caches key on the
longest common byte prefix, and the old assembly (recall -> history -> state)
rewrote the head of the user block every single step, so the cache never hit.
"""

from __future__ import annotations

from agentd.context import ContextBuilder
from agentd.envs.ops_tasks import nginx_down
from agentd.kernel import JitRLKernel, Store
from agentd.loop import AgentLoop, LoopConfig
from agentd.providers import MockProvider, ProviderSpec
from agentd.toolbus import ToolBus, register_builtins


def test_recall_sits_after_state_in_the_user_block():
    builder = ContextBuilder()
    plan = builder.build(instructions="Task: fix nginx", catalog="ssh.exec: run",
                         recall=["restart worked before"],
                         history=["ran tail -n 50 /var/log/nginx/error.log"],
                         state="nginx is down")
    assert plan.user.index("# recent steps") < plan.user.index("# current state") \
        < plan.user.index("# what worked here before")


def test_user_is_byte_prefix_stable_as_history_grows():
    """Step N's user string must be a byte prefix of step N+1's when nothing but
    the appended history entry differs and the history block is the whole tail.

    With a state/recall block present, the byte-identical region is the fixed
    system prefix plus the entire history section -- that is the assertion
    test_history_section_is_byte_prefix_stable below. A full-user prefix can
    only hold when nothing follows the growing block, which is what this test
    pins.
    """
    builder = ContextBuilder()
    history = [f"chose action {i} -> ok" for i in range(6)]
    prev = builder.build(instructions="i", catalog="c", recall=[], history=history[:1], state="")
    for k in range(2, 6):
        plan = builder.build(instructions="i", catalog="c", recall=[], history=history[:k], state="")
        assert plan.user.startswith(prev.user), (k, prev.user, plan.user)
        prev = plan


def test_history_section_is_byte_prefix_stable_with_state_and_recall():
    builder = ContextBuilder()
    state = "nginx is down on app-01"
    recall = ["restart worked before"]
    common = dict(instructions="i", catalog="c", recall=recall, state=state)
    prev = builder.build(history=[], **common)
    for k in range(1, 6):
        plan = builder.build(history=[f"chose action {i} -> ok" for i in range(k)], **common)
        section = _history_section(prev.user)
        if section:
            # everything up to and including the history section + its separator
            # is byte-identical: exactly the region a provider prefix cache retains
            assert plan.user.startswith(section), (k, section, plan.user)
        prev = plan


def _history_section(user: str) -> str:
    """The history block without its trailing separator: the only region that is
    byte-stable when the next step appends an entry."""
    start = user.find("# recent steps")
    if start < 0:
        return ""
    end = user.find("\n\n# ", start + 1)
    return user if end < 0 else user[:end]


class CountingKernel(JitRLKernel):
    """JitRLKernel that counts recall queries -- the loop must ask once per episode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyses_calls = 0

    def analyses_for(self, task: str) -> list[str]:
        self.analyses_calls += 1
        return super().analyses_for(task)


def test_loop_queries_recall_once_per_episode():
    store = Store()
    kernel = CountingKernel(store=store, gamma=0.5, seed=11)
    bus = ToolBus()
    register_builtins(bus)
    provider = MockProvider(ProviderSpec(label="m", model="m", kind="mock"),
                            weights={"systemctl restart nginx": 0.3,
                                     "reboot the whole host": 0.7})
    loop = AgentLoop(kernel, provider, bus, config=LoopConfig(max_steps=4))
    loop.run(nginx_down())
    assert kernel.analyses_calls == 1, "recall must be frozen once per episode, not per step"
