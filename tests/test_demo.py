"""The keyless demo is the first thing a newcomer runs, so its headline claim
-- memory turns lucky hits into a habit -- has to hold, not just render.

It did not. `run_task` rewarded only the final step, so every correct
intermediate move was recorded as a failure, those states only ever collected a
return of 0, and the learning arm scored 0/40 while the identical mechanism
under `agentd bench` scored 21/40. Nothing caught it because the demo had no
test at all.
"""

from agentd.demo import demo


def test_demo_learning_arm_beats_the_control_arm():
    result = demo(episodes=20, seed=7)
    assert result["memory_on"]["successes"] > result["memory_off"]["successes"], (
        f"memory must help: off={result['memory_off']['successes']} "
        f"on={result['memory_on']['successes']}"
    )


def test_demo_improves_within_the_run_not_just_at_the_start():
    """The point is that experience accumulates: the second half must beat the first."""
    result = demo(episodes=20, seed=7)
    on = result["memory_on"]
    assert on["second_half"] > on["first_half"], (
        f"learning arm did not improve over time: first={on['first_half']} second={on['second_half']}"
    )


def test_demo_is_deterministic_for_a_seed():
    """Unseeded noise would make the headline curve unreproducible run to run."""
    assert demo(episodes=10, seed=7) == demo(episodes=10, seed=7)


def test_control_arm_is_the_plain_model_and_stays_flat():
    """The control arm is beta=0. It must not drift upward on its own."""
    result = demo(episodes=20, seed=7)
    off = result["memory_off"]
    assert off["successes"] <= len(off["curve"])
    assert off["second_half"] <= off["first_half"] + 0.25