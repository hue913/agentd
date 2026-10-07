"""Server-screen perception: capture, change detection, structural observation,
and gated synthetic input. See capture.py for the measurement that shaped the
change detector."""
from .actions import ActionResult, ScreenActionError, ScreenActionRefused, ScreenActions
from .capture import CaptureError, Frame, ScreenCapture
from .perceive import Observation, Perceiver

__all__ = [
    "ActionResult", "CaptureError", "Frame", "Observation", "Perceiver",
    "ScreenActionError", "ScreenActionRefused", "ScreenActions", "ScreenCapture",
]
