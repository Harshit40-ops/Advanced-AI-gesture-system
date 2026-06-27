"""Superhero-style power FX: glowing fingertip trails + energy-blast shockwaves.

* :class:`TrailFX` -- persistent decaying buffer that leaves glowing motion
  trails behind every fingertip (ambient flair on all gestures).
* :class:`EnergyBlast` -- expanding shockwave projectiles fired by a fast open-
  palm thrust ("force push" / Hadouken), travelling in the thrust direction.

Pure NumPy + OpenCV.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]

_TIPS = (4, 8, 12, 16, 20)
# Per-finger trail colours (BGR): thumb, index, middle, ring, pinky.
_TIP_COLORS = [(80, 180, 255), (60, 230, 255), (120, 255, 200),
               (255, 220, 80), (255, 130, 90)]


class TrailFX:
    """Glowing motion trails following the fingertips.

    The persistent buffer is kept at 1/``scale`` resolution: decay, draw, and
    blur all happen on the small buffer (cheap), then it's upscaled and added.
    This avoids a costly full-frame GaussianBlur every frame.
    """

    def __init__(self, decay: float = 0.80, scale: int = 3) -> None:
        self._decay = decay
        self._scale = scale
        self._buf: np.ndarray | None = None

    def render(self, frame: np.ndarray, hands: Sequence) -> np.ndarray:
        h, w = frame.shape[:2]
        sh, sw = max(2, h // self._scale), max(2, w // self._scale)
        if self._buf is None or self._buf.shape[:2] != (sh, sw):
            self._buf = np.zeros((sh, sw, 3), np.uint8)
        self._buf = cv2.convertScaleAbs(self._buf, alpha=self._decay)  # fade
        for hand in hands:
            for c, idx in zip(_TIP_COLORS, _TIPS):
                x = int(hand.pixels[idx][0]) // self._scale
                y = int(hand.pixels[idx][1]) // self._scale
                cv2.circle(self._buf, (x, y), 2, c, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(self._buf, (0, 0), sigmaX=1.5)
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.add(frame, glow)


class _Blast:
    __slots__ = ("pos", "vel", "age", "life", "r0")

    def __init__(self, pos, vel, r0):
        self.pos = np.array(pos, np.float32)
        self.vel = np.array(vel, np.float32)
        self.age = 0.0
        self.life = 0.7
        self.r0 = r0


class EnergyBlast:
    """Manages expanding energy-blast shockwaves fired from the hand."""

    CORE = (255, 255, 255)
    HOT = (120, 240, 255)
    EDGE = (40, 150, 255)

    def __init__(self) -> None:
        self._blasts: List[_Blast] = []

    def fire(self, pos: Point, direction: np.ndarray, radius: float) -> None:
        """Spawn a blast at ``pos`` travelling along unit ``direction``."""
        self._blasts.append(_Blast(pos, direction * 1100.0, radius * 0.7))

    def render(self, frame: np.ndarray, dt: float) -> np.ndarray:
        if not self._blasts:
            return frame
        soft = np.zeros_like(frame)
        crisp = np.zeros_like(frame)
        alive: List[_Blast] = []
        for b in self._blasts:
            b.age += dt
            b.pos = b.pos + b.vel * dt
            f = b.age / b.life            # 0..1 progress
            if f >= 1.0:
                continue
            alive.append(b)
            c = (int(b.pos[0]), int(b.pos[1]))
            rr = int(b.r0 * (1.0 + 3.0 * f))
            alpha = (1.0 - f)
            # expanding shock ring + bright travelling core
            cv2.circle(soft, c, rr, _s(self.EDGE, alpha), max(2, int(8 * alpha)),
                       cv2.LINE_AA)
            cv2.circle(soft, c, int(b.r0 * (1 - 0.5 * f)), _s(self.HOT, alpha),
                       -1, cv2.LINE_AA)
            cv2.circle(crisp, c, rr, _s(self.CORE, alpha), 2, cv2.LINE_AA)
            cv2.circle(crisp, c, max(2, int(b.r0 * (1 - 0.6 * f))), self.CORE,
                       -1, cv2.LINE_AA)
        self._blasts = alive

        h, w = frame.shape[:2]
        small = cv2.resize(soft, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=6)
        bloom = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        frame = cv2.add(frame, bloom)
        frame = cv2.add(frame, crisp)
        return frame


def _s(c, b):
    return (min(255, int(c[0] * b)), min(255, int(c[1] * b)), min(255, int(c[2] * b)))
