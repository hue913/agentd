"""Screen capture, change detection, and gated synthetic input.

Split deliberately in two:

* Pure-logic tests (change detection maths, allow-lists, bounds) always run, so
  CI without an X server still covers the part most likely to be wrong.
* Live tests need `xwd`/`convert`/`xdotool` and a reachable display; they skip
  cleanly otherwise rather than failing a build that has no X server.

The change-detection tests encode a measurement, not an assumption: two grabs of
a static display produce DIFFERENT raw `xwd` bytes (256 bytes of colour map
move every time) but byte-identical `convert` output. Anything that regresses to
hashing the raw file will show up here as a permanent "changed".
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

from agentd.screen.actions import SAFE_KEYS, ScreenActionError, ScreenActions
from agentd.screen.capture import PROBE_H, PROBE_W, ScreenCapture
from agentd.screen.perceive import Perceiver

DISPLAY = os.environ.get("AGENTD_DISPLAY", ":99")


def _live() -> bool:
    """True only when the tools AND a reachable display are both present.

    Checking binaries alone is not enough: a CI image can have xwd installed
    with no X server, and then every live test fails instead of skipping.
    """
    if not (shutil.which("xwd") and shutil.which("convert") and shutil.which("xdotool")):
        return False
    probe = subprocess.run(["xdpyinfo"], capture_output=True, timeout=5,
                           env={**os.environ, "DISPLAY": DISPLAY})
    return probe.returncode == 0


requires_display = pytest.mark.skipif(not _live(), reason="needs xwd/convert/xdotool")


# --- pure logic ------------------------------------------------------------

def test_probe_geometry_is_what_the_converter_is_asked_for():
    assert PROBE_W * PROBE_H == 16000


def test_diff_reports_no_change_for_identical_probes():
    probe = bytes([100] * (PROBE_W * PROBE_H))
    ratio, box = ScreenCapture._diff(probe, probe)
    assert ratio == 0.0
    assert box is None


def test_diff_locates_the_changed_region():
    prev = bytearray([0] * (PROBE_W * PROBE_H))
    cur = bytearray(prev)
    for y in range(10, 20):
        for x in range(30, 40):
            cur[y * PROBE_W + x] = 255
    ratio, box = ScreenCapture._diff(bytes(prev), bytes(cur))
    assert ratio == pytest.approx(100 / (PROBE_W * PROBE_H), rel=0.01)
    assert box == (30, 10, 40, 20), box


def test_diff_tolerates_sub_threshold_noise():
    """Cursor blink and clock ticks produce 1-2 level changes; those are noise."""
    prev = bytes([128] * (PROBE_W * PROBE_H))
    cur = bytes([130] * (PROBE_W * PROBE_H))  # 2 levels, under the tolerance of 8
    ratio, box = ScreenCapture._diff(prev, cur)
    assert ratio == 0.0 and box is None


def test_diff_against_no_previous_frame_is_a_full_change():
    ratio, box = ScreenCapture._diff(b"", bytes(PROBE_W * PROBE_H))
    assert ratio == 1.0
    assert box == (0, 0, PROBE_W, PROBE_H)


def test_ink_ratio_uses_the_modal_value_as_background():
    # 90% background, 10% content
    probe = bytes([20] * 900 + [220] * 100)
    assert ScreenCapture._ink_ratio(probe) == pytest.approx(0.1, abs=0.01)


def test_a_uniform_frame_is_completely_blank():
    assert ScreenCapture._ink_ratio(bytes([42] * (PROBE_W * PROBE_H))) == 0.0


# --- input guards (no display needed) --------------------------------------

@pytest.mark.parametrize("name", sorted(SAFE_KEYS))
def test_allow_listed_keys_pass_validation(name):
    ScreenActions(display="none").key(name)  # may fail on xdotool, must not raise ScreenActionError


@pytest.mark.parametrize("name", ["ctrl+del", "ctrl+alt+backspace", "; rm -rf /", "", "A" * 40,
                                   "super+q", "f13", "xdotool"])
def test_keys_outside_the_allow_list_are_refused(name):
    with pytest.raises(ScreenActionError):
        ScreenActions(display="none").key(name)


def test_allow_list_excludes_the_obviously_dangerous():
    for dangerous in ("ctrl+del", "ctrl+alt+t", "super", "ctrl+w"):
        # 'super' is present for window switching; everything else must not be.
        if dangerous != "super":
            assert dangerous not in SAFE_KEYS, dangerous


@pytest.mark.parametrize("x,y", [(9999, 9999), (-1, 10), (10, -1), (1440, 100), (100, 900)])
def test_out_of_bounds_clicks_are_refused(x, y):
    with pytest.raises(ScreenActionError):
        ScreenActions(display="none", width=1440, height=900).click(x, y)


def test_bad_button_is_refused():
    with pytest.raises(ScreenActionError):
        ScreenActions(display="none").click(10, 10, button=9)


def test_oversized_typing_is_refused():
    with pytest.raises(ScreenActionError):
        ScreenActions(display="none").type_text("x" * 5000)


# --- perception honesty ----------------------------------------------------

def test_perceiver_states_plainly_when_no_vision_model_is_attached():
    """The single most important property here: never imply it saw something."""
    from agentd.screen.capture import Frame

    frame = Frame(ts=0.0, sha="deadbeef", changed=True, width=1440, height=900,
                  mode="grayscale", ink_ratio=0.2, blank=False, diff_ratio=1.0)
    obs = Perceiver(describe=None).observe(frame)
    assert obs.vision is None
    assert "NOT been described" in obs.vision_skipped_reason
    assert "no vision model configured" in obs.text


def test_perceiver_uses_the_vision_model_when_one_is_supplied():
    from agentd.screen.capture import Frame

    frame = Frame(ts=0.0, sha="x", changed=True, width=1440, height=900, mode="grayscale",
                  ink_ratio=0.2, blank=False, diff_ratio=0.1, jpeg_path="/dev/null")
    seen = {}

    def describe(path, question):
        seen["path"] = path
        return "a terminal window showing a system dashboard"

    obs = Perceiver(describe=describe, model_name="fake-vlm").observe(frame)
    assert obs.vision is not None
    assert obs.vision["model"] == "fake-vlm"
    assert "system dashboard" in obs.text
    assert seen["path"] == "/dev/null"


def test_perceiver_reports_a_failed_vision_call_instead_of_hiding_it():
    from agentd.screen.capture import Frame

    frame = Frame(ts=0.0, sha="x", changed=True, width=1440, height=900, mode="grayscale",
                  ink_ratio=0.2, blank=False, diff_ratio=0.1, jpeg_path="/dev/null")

    def boom(path, question):
        raise TimeoutError("upstream timed out")

    obs = Perceiver(describe=boom).observe(frame)
    assert obs.vision is None
    assert "vision call failed" in obs.vision_skipped_reason
    assert "TimeoutError" in obs.vision_skipped_reason


def test_perceiver_skips_the_vision_call_on_a_blank_screen():
    from agentd.screen.capture import Frame

    frame = Frame(ts=0.0, sha="x", changed=True, width=1440, height=900, mode="grayscale",
                  ink_ratio=0.0, blank=True, diff_ratio=1.0, jpeg_path="/dev/null")
    calls = []

    def describe(path, question):
        calls.append(path)
        return "should not happen"

    obs = Perceiver(describe=describe).observe(frame)
    assert calls == [], "a blank screen must not cost a vision call"
    assert obs.vision is None


# --- live ------------------------------------------------------------------

@requires_display
def test_capture_produces_a_decodable_jpeg():
    cap = ScreenCapture(display=DISPLAY)
    try:
        frame = cap.grab()
        assert frame.jpeg_bytes > 0
        assert os.path.getsize(frame.jpeg_path) == frame.jpeg_bytes
        head = open(frame.jpeg_path, "rb").read(3)
        assert head == b"\xff\xd8\xff", "not a JPEG"
    finally:
        cap.cleanup()


@requires_display
def test_a_static_display_is_reported_as_unchanged():
    """The regression guard for hashing raw xwd output instead of the probe."""
    cap = ScreenCapture(display=DISPLAY)
    try:
        cap.grab(want_image=False)
        time.sleep(0.6)
        second = cap.grab(want_image=False)
        assert second.changed is False, (
            f"idle screen reported as changed (diff={second.diff_ratio}); "
            "the probe hash is not stable"
        )
        assert second.diff_ratio == 0.0
    finally:
        cap.cleanup()


@requires_display
def test_window_list_parses_the_real_xwininfo_layout():
    result = ScreenActions(display=DISPLAY).window_list()
    if result.get("error"):
        pytest.skip(result["error"])
    for w in result["windows"]:
        assert w["id"].startswith("0x")
        assert w["w"] > 1 and w["h"] > 1, "1x1 openbox helpers should be filtered out"


@requires_display
def test_click_actually_moves_the_pointer():
    actions = ScreenActions(display=DISPLAY)
    actions.move(200, 150)
    import subprocess

    pos = subprocess.run(["xdotool", "getmouselocation"], capture_output=True, timeout=5,
                         env={**os.environ, "DISPLAY": DISPLAY})
    assert b"200" in pos.stdout and b"150" in pos.stdout, pos.stdout
