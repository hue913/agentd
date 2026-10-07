"""Turn a captured frame into an observation a text model can actually use.

The rule this module is built around: never invent a description of the screen.
If no vision model is wired up, the observation says exactly that and stops.
An agent that believes it "saw" a dialog it never saw will act on a fiction,
which is far worse than knowing it is blind.

What a text-only model still gets, for free and truthfully:
  * geometry and display mode
  * how much of the screen is non-background ("ink"), and whether it is blank
  * the bounding box of what changed since the last look, in real pixels
  * the window list with geometry and mapped state

That is enough to drive a scripted workflow and to decide *whether* to spend a
vision call, which is the expensive part.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from .actions import ScreenActions
from .capture import PROBE_H, PROBE_W, Frame

# Below this, a vision call is not worth its cost on a metered link.
MIN_INK_FOR_VISION = 0.001
# Frames that changed by less than this fraction are treated as noise: cursor
# blink, a clock ticking once a second.
MIN_DIFF_FOR_VISION = 0.0002


@dataclass
class Observation:
    ts: float
    changed: bool
    blank: bool
    text: str
    vision: dict | None = None
    vision_skipped_reason: str = ""
    windows: list = None
    frame: dict = None

    def as_dict(self) -> dict:
        return {
            "ts": self.ts, "changed": self.changed, "blank": self.blank,
            "text": self.text, "vision": self.vision,
            "vision_skipped_reason": self.vision_skipped_reason,
            "windows": self.windows or [], "frame": self.frame,
        }


class Perceiver:
    def __init__(self, actions: ScreenActions | None = None, describe=None,
                 model_name: str = ""):
        self.actions = actions
        # describe(jpeg_path, question) -> str | None. Left injectable so a
        # vision provider can be attached without this module knowing which one.
        self.describe = describe
        self.model_name = model_name

    @property
    def has_vision(self) -> bool:
        return callable(self.describe)

    @staticmethod
    def _scale_to_real(bbox, real_w: int, real_h: int) -> list[int] | None:
        if not bbox:
            return None
        x0, y0, x1, y1 = bbox
        sx = real_w / PROBE_W
        sy = real_h / PROBE_H
        return [int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)]

    def observe(self, frame: Frame, real_width: int = 1440, real_height: int = 900,
                question: str = "Describe what is on this screen and what a user should do next.") -> Observation:
        windows = []
        if self.actions is not None:
            windows = self.actions.window_list().get("windows", [])

        lines = [
            f"screen {real_width}x{real_height}, captured {time.strftime('%H:%M:%S')}",
            f"display content: {'BLANK (no window manager or no window)' if frame.blank else 'has content'}",
            f"ink ratio: {frame.ink_ratio:.4f} of the frame is non-background",
        ]
        if frame.changed and frame.bbox:
            box = self._scale_to_real(frame.bbox, real_width, real_height)
            lines.append(f"changed since last look: {frame.diff_ratio:.4f} of the frame, region {box}")
        elif not frame.changed:
            lines.append("unchanged since the previous observation")
        if windows:
            shown = [w for w in windows if w.get("area", 0) > 0]
            lines.append(f"windows on display: {len(windows)}")
            for w in shown[:5]:
                label = w.get("name") or "(unnamed)"
                lines.append(f"  - {label} id={w['id']} {w['w']}x{w['h']} at ({w['x']},{w['y']})")
        else:
            lines.append("windows on display: none reported")

        vision = None
        skipped = ""
        if not self.has_vision:
            skipped = ("no vision model configured: this is a structural reading only. "
                       "The screen contents have NOT been described.")
            lines.append(skipped)
        elif frame.blank:
            skipped = "screen is blank; no point paying for a vision call"
        elif frame.diff_ratio < MIN_DIFF_FOR_VISION and not frame.changed:
            skipped = "nothing changed since the last look; skipping the vision call"
        elif frame.ink_ratio < MIN_INK_FOR_VISION:
            skipped = "frame is effectively blank; skipping the vision call"
        else:
            if not frame.jpeg_path or not Path(frame.jpeg_path).exists():
                skipped = "frame image missing; cannot describe"
            else:
                try:
                    text = self.describe(frame.jpeg_path, question)
                    if text:
                        vision = {"model": self.model_name, "question": question,
                                  "answer": text, "jpeg_bytes": frame.jpeg_bytes}
                        lines.append(f"vision ({self.model_name}): {text}")
                    else:
                        skipped = "vision model returned nothing"
                except Exception as exc:
                    # A failed vision call is reported, never swallowed into a
                    # confident-looking empty answer.
                    skipped = f"vision call failed: {type(exc).__name__}: {exc}"

        return Observation(
            ts=frame.ts, changed=frame.changed, blank=frame.blank,
            text="\n".join(lines), vision=vision, vision_skipped_reason=skipped,
            windows=windows, frame=frame.as_dict(),
        )
