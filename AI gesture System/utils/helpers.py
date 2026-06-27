"""Small, dependency-light helpers shared across the project.

Keeping these here avoids circular imports between core modules and the
action layer. Everything in this module is pure / side-effect free.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """Rolling-average frames-per-second meter.

    Call :meth:`tick` once per processed frame; read :attr:`fps` any time.
    A rolling window (instead of a single instantaneous delta) keeps the
    on-screen number stable enough to read.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window)
        self.fps: float = 0.0

    def tick(self) -> float:
        """Record a frame timestamp and update :attr:`fps`. Returns current FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self.fps = (len(self._timestamps) - 1) / elapsed
        return self.fps
