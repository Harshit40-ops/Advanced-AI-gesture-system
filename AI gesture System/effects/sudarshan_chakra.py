"""Sudarshan Chakra -- a divine flaming discus spinning on a fingertip.

Styled after the classic depiction: an ornate metallic wheel seen in tilted
perspective (a flattened ellipse), wreathed in fire, spinning fast above the
raised index finger. Layers: golden bloom, a roaring fire ring, an engraved
spinning wheel (rim + glyph studs + spokes + inner hub), and flying sparks.

Pure NumPy + OpenCV. Public API mirrors the other effects:
``render(frame, center, radius, intensity) -> frame``.
"""

from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Color = Tuple[int, int, int]


def _scale(c: Color, b: float) -> Color:
    return (min(255, int(c[0] * b)), min(255, int(c[1] * b)),
            min(255, int(c[2] * b)))


def _lerp(a: Color, b: Color, f: float) -> Color:
    return (int(a[0] + (b[0] - a[0]) * f),
            int(a[1] + (b[1] - a[1]) * f),
            int(a[2] + (b[2] - a[2]) * f))


class SudarshanChakra:
    """Renders an animated tilted, flaming, spinning chakra at a point."""

    # Palette (BGR).
    HOT = (130, 235, 255)   # warm yellow (flame base)
    YELLOW = (60, 230, 255)
    ORANGE = (15, 130, 255)
    RED = (5, 45, 205)      # deep red (flame tip)
    GOLD = (50, 200, 255)
    WHITE = (255, 255, 255)

    SPIN = 8.0              # spin speed (rad/s)
    TILT = 0.42            # ellipse minor/major ratio (perspective tilt)
    N_FLAMES = 50
    N_SPOKES = 22
    N_STUDS = 22

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._rng = np.random.default_rng(11)

    # -- public ----------------------------------------------------------

    def render(
        self, frame: np.ndarray, center: Point, radius: float,
        intensity: float = 1.0,
    ) -> np.ndarray:
        if intensity <= 0.02 or radius < 6:
            return frame
        intensity = float(min(1.0, intensity))
        t = time.perf_counter() - self._t0
        flick = 0.85 + 0.15 * np.sin(t * 25.0)
        phi = t * self.SPIN
        R = radius

        soft = np.zeros_like(frame)
        crisp = np.zeros_like(frame)

        # Order: fire glow behind, metallic wheel on top, sparks last.
        self._fire(soft, crisp, center, R, t)
        self._disc_glow(soft, center, R, flick)
        self._wheel(crisp, center, R, phi, flick)
        self._sparks(crisp, center, R, t)

        # Cheap golden bloom.
        h, w = frame.shape[:2]
        small = cv2.resize(soft, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.5, R * 0.06))
        bloom = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        frame = cv2.addWeighted(frame, 1.0, bloom, intensity, 0)
        frame = cv2.addWeighted(frame, 1.0, crisp, intensity, 0)
        return frame

    # -- fire ------------------------------------------------------------

    def _fire_col(self, f: float) -> Color:
        """Flame colour from hot base (f=0) to red tip (f=1)."""
        if f < 0.4:
            return _lerp(self.HOT, self.ORANGE, f / 0.4)
        return _lerp(self.ORANGE, self.RED, (f - 0.4) / 0.6)

    def _fire(self, soft, crisp, center, R, t):
        """A roaring ring of tall, flickering flame tongues around the rim."""
        cx, cy = center
        n = self.N_FLAMES
        for i in range(n):
            a = 2 * np.pi * i / n
            ca, sa = np.cos(a), np.sin(a)
            bx = cx + R * ca
            by = cy + R * self.TILT * sa
            # Outward normal of the ellipse, strongly biased upward (fire rises).
            nx, ny = ca, self.TILT * sa - 0.7
            ln = np.hypot(nx, ny) or 1.0
            nx, ny = nx / ln, ny / ln
            # Spiky, dancing height: layered sines + per-frame flicker.
            noise = abs(np.sin(a * 4 + t * 11 + i)) * 0.6 + \
                abs(np.sin(a * 9 - t * 7)) * 0.4
            fh = R * (0.25 + 0.95 * noise) * float(self._rng.uniform(0.8, 1.2))
            # slight sideways sway so tongues aren't perfectly radial
            sway = np.sin(t * 8 + i) * 0.18
            steps = 8
            for s in range(steps):
                f = s / steps
                px = int(bx + (nx + sway * -ny) * fh * f)
                py = int(by + (ny + sway * nx) * fh * f)
                rad = max(1, int((1 - f) ** 1.5 * R * 0.16))  # thin, tapering
                cv2.circle(soft, (px, py), rad, self._fire_col(f), -1,
                           cv2.LINE_AA)
                if s < 2:  # crisp hot cores near the base
                    cv2.circle(crisp, (px, py), max(1, rad // 2),
                               self._fire_col(f * 0.4), -1, cv2.LINE_AA)

    def _disc_glow(self, soft, center, R, flick):
        """Soft golden disc behind the wheel (feeds bloom)."""
        steps = 5
        for k in range(steps, 0, -1):
            rx = int(R * 0.95 * (k / steps))
            cv2.ellipse(soft, center, (rx, int(rx * self.TILT)), 0, 0, 360,
                        _scale(self.GOLD, ((1 - k / steps) * 0.5 + 0.2) * flick),
                        -1, cv2.LINE_AA)

    # -- metallic wheel --------------------------------------------------

    def _pt(self, center, r, ang) -> Point:
        """Project a disc point (radius r, angle ang) onto the tilted ellipse."""
        return (int(center[0] + r * np.cos(ang)),
                int(center[1] + r * self.TILT * np.sin(ang)))

    def _wheel(self, buf, center, R, phi, flick):
        cx, cy = center
        g = _scale(self.GOLD, flick)
        # rim rings
        cv2.ellipse(buf, center, (int(R), int(R * self.TILT)), 0, 0, 360, g,
                    2, cv2.LINE_AA)
        cv2.ellipse(buf, center, (int(R * 0.82), int(R * 0.82 * self.TILT)),
                    0, 0, 360, _scale(self.YELLOW, flick), 1, cv2.LINE_AA)
        cv2.ellipse(buf, center, (int(R * 0.34), int(R * 0.34 * self.TILT)),
                    0, 0, 360, g, 1, cv2.LINE_AA)
        # glyph studs around the rim (rotating)
        for i in range(self.N_STUDS):
            a = phi * 0.5 + 2 * np.pi * i / self.N_STUDS
            cv2.circle(buf, self._pt(center, R * 0.91, a), 2,
                       _scale(self.YELLOW, flick), -1, cv2.LINE_AA)
        # spokes
        for i in range(self.N_SPOKES):
            a = phi + 2 * np.pi * i / self.N_SPOKES
            cv2.line(buf, self._pt(center, R * 0.34, a),
                     self._pt(center, R * 0.80, a), g, 1, cv2.LINE_AA)
        # hub
        cv2.ellipse(buf, center, (int(R * 0.16), int(R * 0.16 * self.TILT)),
                    0, 0, 360, g, -1, cv2.LINE_AA)
        cv2.circle(buf, center, max(1, int(R * 0.06)), self.WHITE, -1,
                   cv2.LINE_AA)

    def _sparks(self, buf, center, R, t):
        cx, cy = center
        for _ in range(12):
            a = self._rng.uniform(0, 2 * np.pi)
            rr = R * self._rng.uniform(1.0, 1.6)
            p = (int(cx + rr * np.cos(a)),
                 int(cy + rr * self.TILT * np.sin(a) - R * 0.2))
            cv2.circle(buf, p, 1, self.YELLOW, -1, cv2.LINE_AA)
