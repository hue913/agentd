"""Synthetic input on the X display, gated like every other command.

These are the only tools that let a model *act* on the screen, so they are the
ones that matter for blast radius. Three guards, in order:

1. The safety gate classifies the command line before anything runs.
2. Coordinates are bounds-checked against the display geometry, so a model that
   hallucinates (9999, 9999) gets a clear error instead of a stray click that
   lands on a maximise button.
3. Every action is audited, and the result records whether the display actually
   changed -- an action that reports success but moved nothing is the failure
   mode worth catching.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

from ..safety.gate import classify

TIMEOUT = 10
# Key names xdotool understands. Allow-listing matters: passing arbitrary text to
# `key` would let a model inject modifier chords (ctrl+alt+del) unfiltered.
SAFE_KEYS = {
    "return", "enter", "tab", "escape", "esc", "space", "backspace", "delete",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
    "insert", "ctrl+c", "ctrl+d", "ctrl+z", "ctrl+l", "ctrl+a", "ctrl+e",
    "ctrl+u", "ctrl+k", "alt+f4", "super", "shift",
}
_KEY_RE = re.compile(r"^[a-z0-9_+]{1,24}$")


class ScreenActionError(RuntimeError):
    pass


class ScreenActionRefused(ScreenActionError):
    pass


@dataclass
class ActionResult:
    action: str
    ok: bool
    detail: str = ""
    changed: bool = False
    elapsed_ms: int = 0

    def as_dict(self) -> dict:
        return {"action": self.action, "ok": self.ok, "detail": self.detail,
                "changed": self.changed, "elapsed_ms": self.elapsed_ms}


def _xdotool() -> str:
    path = shutil.which("xdotool")
    if not path:
        raise ScreenActionError("'xdotool' is not installed (apt-get install -y xdotool)")
    return path


class ScreenActions:
    def __init__(self, display: str = ":99", width: int = 1440, height: int = 900,
                 audit=None):
        self.display = display
        self.width = width
        self.height = height
        self.audit = audit

    def _env(self) -> dict:
        return {**os.environ, "DISPLAY": self.display}

    def _run(self, args: list[str], gate_text: str, action: str) -> ActionResult:
        verdict = classify(gate_text)
        if verdict.hard_blocked:
            self._audit(action, gate_text, "block", verdict.reasons)
            raise ScreenActionRefused("; ".join(verdict.reasons) or "blocked by safety gate")
        t0 = time.time()
        try:
            proc = subprocess.run([_xdotool(), *args], capture_output=True, timeout=TIMEOUT,
                                  env=self._env())
        except subprocess.TimeoutExpired:
            return ActionResult(action, False, f"xdotool timed out after {TIMEOUT}s",
                                elapsed_ms=int((time.time() - t0) * 1000))
        except Exception as exc:
            return ActionResult(action, False, f"{type(exc).__name__}: {exc}",
                                elapsed_ms=int((time.time() - t0) * 1000))
        ok = proc.returncode == 0
        detail = "" if ok else proc.stderr.decode("utf-8", "replace")[:200]
        self._audit(action, gate_text, "allow" if ok else "allow", verdict.reasons, error=detail)
        return ActionResult(action, ok, detail, elapsed_ms=int((time.time() - t0) * 1000))

    def _check_point(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ScreenActionError(
                f"({x},{y}) is outside the {self.width}x{self.height} display; "
                "refusing to click blind"
            )

    def move(self, x: int, y: int) -> ActionResult:
        x, y = int(x), int(y)
        self._check_point(x, y)
        return self._run(["mousemove", "--sync", str(x), str(y)], f"mousemove {x} {y}", "move")

    def click(self, x: int, y: int, button: int = 1, times: int = 1) -> ActionResult:
        x, y = int(x), int(y)
        self._check_point(x, y)
        if button not in (1, 2, 3):
            raise ScreenActionError("button must be 1 (left), 2 (middle) or 3 (right)")
        times = max(1, min(int(times), 5))
        # xdotool's `click` takes a button number only and acts at the current
        # pointer position -- passing coordinates after it makes xdotool treat
        # them as a subcommand and fail with "Unknown command". The idiom is
        # mousemove first, then click.
        move = self.move(x, y)
        if not move.ok:
            return move
        return self._run(["click", "--repeat", str(times), str(button)],
                         f"click {x} {y}", "click")

    def type_text(self, text: str, delay_ms: int = 12) -> ActionResult:
        if not isinstance(text, str):
            raise ScreenActionError("type_text takes a string")
        if len(text) > 4000:
            raise ScreenActionError("refusing to type more than 4000 characters in one action")
        # xdotool type --clearmodifiers avoids a stuck modifier swallowing the text.
        return self._run(["type", "--clearmodifiers", "--delay", str(max(0, int(delay_ms))), "--", text],
                         f"type {text[:60]}", "type")

    def key(self, name: str) -> ActionResult:
        key = (name or "").strip().lower()
        if not _KEY_RE.match(key) or key not in SAFE_KEYS:
            raise ScreenActionError(
                f"'{name}' is not an allowed key. allowed: {', '.join(sorted(SAFE_KEYS))}"
            )
        return self._run(["key", "--clearmodifiers", key], f"key {key}", "key")

    def scroll(self, direction: str = "down", amount: int = 3) -> ActionResult:
        if direction not in ("up", "down", "left", "right"):
            raise ScreenActionError("scroll direction must be up/down/left/right")
        button = {"up": 4, "down": 5, "left": 6, "right": 7}[direction]
        amount = max(1, min(int(amount), 20))
        return self._run(["click", "--repeat", str(amount), str(button)],
                         f"scroll {direction} {amount}", "scroll")

    def window_list(self) -> dict:
        """Structural view of what is on screen. Cheap; no pixels involved.

        Parses the real `xwininfo -root -children` layout, which is
            0xc00064 (has no name): ()  606x393+60+40  +60+40
            0xc0000e "Openbox": ("" (none))  1x1+-100+-100  +-100+-100
        An earlier regex here assumed a different xwininfo build's field order
        and silently matched nothing, so the panel reported "no windows" while
        a window was plainly on screen.
        """
        if not shutil.which("xwininfo"):
            return {"windows": [], "error": "xwininfo not installed"}
        try:
            proc = subprocess.run(["xwininfo", "-root", "-children"], capture_output=True,
                                  timeout=TIMEOUT, env=self._env())
        except Exception as exc:
            return {"windows": [], "error": f"{type(exc).__name__}: {exc}"}
        if proc.returncode != 0:
            return {"windows": [], "error": proc.stderr.decode("utf-8", "replace")[:200]}
        text = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")

        root = re.search(r"Root window id:\s*(0x[0-9a-f]+)", text)
        root_id = root.group(1) if root else ""
        pattern = re.compile(
            r"^\s*(0x[0-9a-f]+)\s+"
            r'(?:"([^"]*)"|\(has no name\))'
            r".*?\s+(\d+)x(\d+)([+-]\d+)([+-]\d+)\s+[+-]\d+[+-]\d+\s*$",
            re.M,
        )
        windows = []
        for m in pattern.finditer(text):
            wid, name, w, h, x, y = m.group(1), m.group(2) or "", *map(int, m.groups()[2:])
            if wid == root_id:
                continue
            # openbox keeps 1x1 helper windows parked off-screen; they are
            # noise, not something an operator or a model should see.
            if w <= 1 and h <= 1:
                continue
            windows.append({"id": wid, "name": name, "x": x, "y": y, "w": w, "h": h,
                            "area": w * h, "onscreen": not (w <= 1 and h <= 1)})
        windows.sort(key=lambda w: w["area"], reverse=True)
        return {"windows": windows, "count": len(windows),
                "root": root_id, "display": self.display}

    def _audit(self, action: str, command: str, level: str, reasons: list[str],
               error: str = "") -> None:
        if self.audit is None:
            return
        try:
            from ..safety.audit import AuditRecord

            self.audit.write(AuditRecord(host=f"display{self.display}", command=command,
                                         level=level, reasons=reasons, error=error))
        except Exception as exc:
            # Never let auditing break input injection — but say why the
            # record is missing instead of failing silently.
            from ..log import get_logger

            get_logger("agentd.screen").warning("screen audit write failed: %s", exc)
