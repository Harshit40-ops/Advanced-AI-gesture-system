"""Temporal smoothing helpers for stable, spam-free gesture control.

Three small, independent tools:

* :class:`GestureStabilizer` -- N-frame confirmation. A raw per-frame gesture
  must persist for ``confirm_frames`` consecutive frames before it becomes the
  "stable" gesture, killing single-frame flicker. Also tracks how long the
  stable gesture has been held (for hold-to-trigger gestures like the 3-second
  open-palm toggle).
* :class:`Cooldown` -- per-key rate limiter so a confirmed gesture's action
  doesn't fire every frame.
* :class:`EMAFilter` -- exponential moving average for smooth cursor motion.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple


class GestureStabilizer:
    """Confirm a gesture only after it persists for N consecutive frames."""

    def __init__(self, confirm_frames: int = 3) -> None:
        """
        Args:
            confirm_frames: Consecutive frames a raw gesture must repeat before
                it is accepted as the new stable gesture.
        """
        self._confirm = max(1, confirm_frames)
        self._candidate = "none"
        self._count = 0
        self._stable = "none"
        self._stable_since = time.time()

    def update(self, raw_gesture: str) -> str:
        """Feed one frame's raw gesture; return the current stable gesture."""
        if raw_gesture == self._candidate:
            self._count += 1
        else:
            self._candidate = raw_gesture
            self._count = 1

        if self._count >= self._confirm and self._stable != self._candidate:
            self._stable = self._candidate
            self._stable_since = time.time()
        return self._stable

    @property
    def stable(self) -> str:
        """The currently confirmed gesture (``"none"`` if unconfirmed)."""
        return self._stable

    def held_seconds(self) -> float:
        """Seconds the current stable gesture has been continuously held."""
        return time.time() - self._stable_since


class Cooldown:
    """Per-key minimum interval between successive triggers."""

    def __init__(self, seconds: float = 0.7) -> None:
        """
        Args:
            seconds: Minimum time between triggers of the same key.
        """
        self._seconds = seconds
        self._last: Dict[str, float] = {}

    def ready(self, key: str) -> bool:
        """True if ``key`` hasn't fired within the cooldown window."""
        return time.time() - self._last.get(key, 0.0) >= self._seconds

    def mark(self, key: str) -> None:
        """Stamp ``key`` as having just fired."""
        self._last[key] = time.time()


class EMAFilter:
    """Exponential moving average over 2-D points (for cursor smoothing)."""

    def __init__(self, alpha: float = 0.4) -> None:
        """
        Args:
            alpha: Smoothing factor in (0, 1]. Higher = more responsive but
                jitterier; lower = smoother but laggier.
        """
        self._alpha = alpha
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        """Return the smoothed (x, y) for a new raw point."""
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self._alpha * x + (1 - self._alpha) * self._x
            self._y = self._alpha * y + (1 - self._alpha) * self._y
        return self._x, self._y

    def reset(self) -> None:
        """Forget history so the next point starts fresh (no glide-in)."""
        self._x = self._y = None
