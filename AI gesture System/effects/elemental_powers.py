"""Per-gesture elemental powers: fire, ice, and lightning around the hand.

Mapping (single hand):
    fist  -> fire     (flames erupting from the palm and fingertips)
    peace -> ice      (crystalline shards + frost sparkles on the fingertips)
    rock  -> lightning (electric arcs crackling across the fingers)

All hands' effects for a frame are drawn into one soft (bloom) + one crisp
buffer and composited once, so multiple hands stay cheap. Pure NumPy + OpenCV.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np

# Fingertip + palm landmark indices.
_TIPS = (4, 8, 12, 16, 20)
_PALM = (0, 5, 9, 13, 17)

# gesture -> element
POWERS = {"fist": "fire", "peace": "ice", "rock": "lightning"}


def _lerp(a, b, f):
    return (int(a[0] + (b[0] - a[0]) * f),
            int(a[1] + (b[1] - a[1]) * f),
            int(a[2] + (b[2] - a[2]) * f))


class ElementalPowers:
    """Renders fire/ice/lightning around hands based on their gesture."""

    # Palette (BGR).
    F_HOT = (130, 235, 255)
    F_ORANGE = (15, 130, 255)
    F_RED = (5, 45, 205)
    ICE = (255, 230, 150)
    ICE_HOT = (255, 255, 255)
    BOLT = (255, 230, 150)

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._rng = np.random.default_rng(5)

    def render(
        self, frame: np.ndarray,
        hand_gestures: Sequence[Tuple[object, str]],
    ) -> np.ndarray:
        """Draw the elemental effect for each (hand, gesture) with a power."""
        active = [(h, POWERS[g]) for h, g in hand_gestures if g in POWERS]
        if not active:
            return frame

        t = time.perf_counter() - self._t0
        soft = np.zeros_like(frame)
        crisp = np.zeros_like(frame)
        for hand, element in active:
            px = hand.pixels
            tips = [tuple(int(v) for v in px[i]) for i in _TIPS]
            palm = tuple(int(v) for v in np.mean([px[i] for i in _PALM], axis=0))
            scale = float(np.linalg.norm(px[9] - px[0]))  # palm size px
            if element == "fire":
                self._fire(soft, crisp, tips, palm, scale, t)
            elif element == "ice":
                self._ice(soft, crisp, tips, scale, t)
            elif element == "lightning":
                self._lightning(soft, crisp, tips, palm, t)

        h, w = frame.shape[:2]
        small = cv2.resize(soft, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=6)
        bloom = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        frame = cv2.addWeighted(frame, 1.0, bloom, 1.0, 0)
        frame = cv2.addWeighted(frame, 1.0, crisp, 1.0, 0)
        return frame

    # -- fire ------------------------------------------------------------

    def _fire_col(self, f):
        if f < 0.4:
            return _lerp(self.F_HOT, self.F_ORANGE, f / 0.4)
        return _lerp(self.F_ORANGE, self.F_RED, (f - 0.4) / 0.6)

    def _fire(self, soft, crisp, tips, palm, scale, t):
        sources = list(tips) + [palm]
        for (sx, sy) in sources:
            for _ in range(3):
                sway = self._rng.uniform(-0.45, 0.45)
                fh = scale * self._rng.uniform(0.9, 2.0)   # tall tongues
                steps = 10
                for s in range(steps):
                    f = s / steps
                    # curve the tongue sideways as it rises
                    px = int(sx + sway * fh * (f ** 1.5))
                    py = int(sy - fh * f)                  # rise upward
                    rad = max(1, int((1 - f) ** 1.7 * scale * 0.16))  # thin taper
                    cv2.circle(soft, (px, py), rad, self._fire_col(f), -1, cv2.LINE_AA)
                    if s < 3:                              # crisp hot base
                        cv2.circle(crisp, (px, py), max(1, rad // 2),
                                   self._fire_col(f * 0.35), -1, cv2.LINE_AA)

    # -- ice -------------------------------------------------------------

    def _ice(self, soft, crisp, tips, scale, t):
        for (tx, ty) in tips:
            cv2.circle(soft, (tx, ty), int(scale * 0.35), self.ICE, -1, cv2.LINE_AA)
            for k in range(6):                          # crystalline shards
                a = 2 * np.pi * k / 6 + t * 0.5
                L = scale * (0.4 + 0.25 * np.sin(t * 4 + k))
                ex = int(tx + L * np.cos(a))
                ey = int(ty + L * np.sin(a))
                cv2.line(crisp, (tx, ty), (ex, ey), self.ICE_HOT, 1, cv2.LINE_AA)
                # little barbs for a frosty look
                mx, my = (tx + ex) // 2, (ty + ey) // 2
                cv2.line(crisp, (mx, my),
                         (int(mx + 6 * np.cos(a + 1.2)), int(my + 6 * np.sin(a + 1.2))),
                         self.ICE, 1, cv2.LINE_AA)
            for _ in range(4):                          # frost sparkles
                p = (int(tx + self._rng.uniform(-scale, scale) * 0.5),
                     int(ty + self._rng.uniform(-scale, scale) * 0.5))
                cv2.circle(crisp, p, 1, self.ICE_HOT, -1, cv2.LINE_AA)

    # -- lightning -------------------------------------------------------

    def _bolt(self, buf, p0, p1, width):
        p0 = np.array(p0, np.float32)
        p1 = np.array(p1, np.float32)
        n = np.array([-(p1 - p0)[1], (p1 - p0)[0]], np.float32)
        nl = np.linalg.norm(n) or 1.0
        n /= nl
        pts = [p0]
        segs = 5
        for s in range(1, segs):
            base = p0 + (p1 - p0) * (s / segs)
            pts.append(base + n * self._rng.uniform(-12, 12))
        pts.append(p1)
        cv2.polylines(buf, [np.array(pts, np.int32)], False, self.BOLT,
                      width, cv2.LINE_AA)

    def _lightning(self, soft, crisp, tips, palm, t):
        # arcs across consecutive fingertips
        for i in range(len(tips) - 1):
            self._bolt(soft, tips[i], tips[i + 1], 4)
            self._bolt(crisp, tips[i], tips[i + 1], 1)
        # arcs from palm out to each fingertip
        for tp in tips:
            self._bolt(soft, palm, tp, 4)
            self._bolt(crisp, palm, tp, 1)
