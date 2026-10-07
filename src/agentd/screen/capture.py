"""Frame capture from the Xvfb display, and change detection that does not lie.

The measurement that shaped this module
---------------------------------------
The obvious change detector -- hash the bytes `xwd` writes -- is wrong. Two
grabs of a completely idle screen produce different files: 256 bytes in the
colour-map region differ every time, so a raw hash reports "changed" forever
and every frame would be billed to a vision model.

Hashing `convert`'s *normalised* output is stable: two grabs of the idle screen
produced byte-identical output, and a single xdotool click changed it. That is
the basis used here, and it is why capture always goes through convert even
when the caller only wants a change verdict.

Nothing here needs numpy or Pillow. The stats are computed over a 160x100
grayscale probe (16 kB) in pure Python, which keeps the perception loop cheap
enough to run on a 1.9 GB box.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DISPLAY = os.environ.get("AGENTD_DISPLAY", ":99")
CAPTURE_TIMEOUT = 15
CONVERT_TIMEOUT = 20

# Small grayscale probe used for hashing and stats. 16 kB per frame.
PROBE_W, PROBE_H = 160, 100
# Resolution handed to a vision model. The host link is ~10 Mbps, so 1024 wide
# at q60 lands around 60-100 kB -- the difference between a usable loop and an
# unusable one.
MODEL_W = 1024
MODEL_QUALITY = 60

# A frame counts as blank when this fraction of pixels are background. Used to
# tell "the desktop is empty" apart from "a window is on screen".
BLANK_RATIO = 0.995


class CaptureError(RuntimeError):
    pass


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise CaptureError(
            f"'{name}' is not installed. On Debian/Ubuntu: "
            f"apt-get install -y --no-install-recommends x11-apps imagemagick"
        )
    return path


@dataclass
class Frame:
    ts: float
    sha: str                       # hash of the normalised probe
    changed: bool
    width: int
    height: int
    mode: str                      # grayscale | rgb
    ink_ratio: float               # fraction of non-background pixels
    blank: bool
    diff_ratio: float = 0.0        # fraction of probe pixels that differ
    bbox: tuple[int, int, int, int] | None = None   # changed region, probe coords
    jpeg_path: str | None = None
    jpeg_bytes: int = 0
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "ts": self.ts, "sha": self.sha[:16], "changed": self.changed,
            "width": self.width, "height": self.height, "mode": self.mode,
            "ink_ratio": round(self.ink_ratio, 4), "blank": self.blank,
            "diff_ratio": round(self.diff_ratio, 4), "bbox": self.bbox,
            "jpeg_bytes": self.jpeg_bytes, "notes": self.notes,
        }


@dataclass
class ScreenCapture:
    display: str = DISPLAY
    model_width: int = MODEL_W
    quality: int = MODEL_QUALITY
    _last_probe: bytes | None = field(default=None, repr=False)
    _last_sha: str = ""
    _tmpdir: str = field(default="", repr=False)

    def __post_init__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="agentd-screen-")
        self._last_probe = None

    # -- low level --------------------------------------------------------
    def _xwd_bytes(self) -> bytes:
        try:
            proc = subprocess.run(
                [_tool("xwd"), "-root", "-silent", "-display", self.display],
                capture_output=True, timeout=CAPTURE_TIMEOUT,
                env={**os.environ, "DISPLAY": self.display},
            )
        except subprocess.TimeoutExpired as exc:
            raise CaptureError(f"xwd timed out after {CAPTURE_TIMEOUT}s") from exc
        if proc.returncode != 0 or not proc.stdout:
            raise CaptureError(
                f"xwd failed on {self.display}: {proc.stderr.decode('utf-8', 'replace')[:200]}"
            )
        return proc.stdout

    def _convert(self, xwd_path: Path, args: list[str]) -> bytes:
        try:
            proc = subprocess.run(
                [_tool("convert"), f"xwd:{xwd_path}", *args],
                capture_output=True, timeout=CONVERT_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise CaptureError(f"convert timed out after {CONVERT_TIMEOUT}s") from exc
        if proc.returncode != 0:
            raise CaptureError(
                f"convert failed: {proc.stderr.decode('utf-8', 'replace')[:200]}"
            )
        return proc.stdout

    # -- probe ------------------------------------------------------------
    def _probe(self, xwd_path: Path) -> bytes:
        """Normalised 160x100 grayscale. Stable for an unchanged screen."""
        data = self._convert(xwd_path, [
            "-resize", f"{PROBE_W}x{PROBE_H}!", "-colorspace", "Gray", "-depth", "8", "gray:-",
        ])
        if len(data) != PROBE_W * PROBE_H:
            raise CaptureError(f"probe size {len(data)} != {PROBE_W * PROBE_H}")
        return data

    @staticmethod
    def _ink_ratio(probe: bytes) -> float:
        """Fraction of pixels that are not the background value."""
        if not probe:
            return 0.0
        background = max(set(probe), key=probe.count)  # modal value == desktop colour
        non_bg = sum(1 for b in probe if abs(b - background) > 8)
        return non_bg / len(probe)

    @staticmethod
    def _diff(prev: bytes, cur: bytes, tolerance: int = 8) -> tuple[float, tuple | None]:
        if not prev or len(prev) != len(cur):
            return 1.0, (0, 0, PROBE_W, PROBE_H)
        changed = [i for i, (a, b) in enumerate(zip(prev, cur)) if abs(a - b) > tolerance]
        if not changed:
            return 0.0, None
        xs = [i % PROBE_W for i in changed]
        ys = [i // PROBE_W for i in changed]
        return len(changed) / len(cur), (min(xs), min(ys), max(xs) + 1, max(ys) + 1)

    # -- public -----------------------------------------------------------
    def grab(self, want_image: bool = True) -> Frame:
        """One frame. `want_image=False` skips the model-resolution JPEG."""
        xwd_bytes = self._xwd_bytes()
        tmp = Path(self._tmpdir) / "frame.xwd"
        tmp.write_bytes(xwd_bytes)

        probe = self._probe(tmp)
        sha = hashlib.sha256(probe).hexdigest()
        changed = sha != self._last_sha
        diff_ratio, bbox = self._diff(self._last_probe or b"", probe) if self._last_probe else (1.0, None)

        ink = self._ink_ratio(probe)
        frame = Frame(
            ts=time.time(), sha=sha, changed=changed,
            width=PROBE_W * 9, height=PROBE_H * 9,   # scaled to the real display
            mode="grayscale", ink_ratio=ink, blank=ink < 1 - BLANK_RATIO,
            diff_ratio=diff_ratio, bbox=bbox,
        )

        if want_image:
            out = Path(self._tmpdir) / "frame.jpg"
            self._convert(tmp, [
                "-resize", f"{self.model_width}x>", "-quality", str(self.quality), f"jpg:{out}",
            ])
            frame.jpeg_path = str(out)
            frame.jpeg_bytes = out.stat().st_size
            if frame.blank:
                frame.notes = "screen is blank: no window manager or no window on the display"

        self._last_probe = probe
        self._last_sha = sha
        return frame

    def peek(self) -> Frame:
        """Cheap verdict only. Never produces an image, never bills a model."""
        return self.grab(want_image=False)

    def reset(self) -> None:
        self._last_probe = None
        self._last_sha = ""

    def cleanup(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)
