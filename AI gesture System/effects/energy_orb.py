"""A cinematic fusion/nuclear reactor core rendered between the user's hands.

Styled after a contained fusion reactor: a dense fiery **iris core** (white-hot
centre with radial sunburst filaments), a roiling **dark smoke ring** wrapping
the fire, bright **swirling orbital trails** (atom-like rings with comet heads),
drifting embers, an outer glassy **containment sphere**, and energy beams to the
hands -- all wrapped in heavy bloom and faded by ``intensity``.

Pure NumPy + OpenCV. ``render(frame, center, radius, hand_anchors, intensity)``.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Color = Tuple[int, int, int]


def _rot(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    return rz @ ry @ rx


def _scale(c: Color, b: float) -> Color:
    return (min(255, int(c[0] * b)), min(255, int(c[1] * b)),
            min(255, int(c[2] * b)))


def _lerp(a: Color, b: Color, f: float) -> Color:
    f = max(0.0, min(1.0, f))
    return (int(a[0] + (b[0] - a[0]) * f), int(a[1] + (b[1] - a[1]) * f),
            int(a[2] + (b[2] - a[2]) * f))


class EnergyOrb:
    """Renders an animated contained fusion-reactor core onto BGR frames."""

    WHITE = (255, 255, 255)
    YELLOW = (130, 245, 255)
    ORANGE = (20, 140, 255)
    RED = (10, 45, 200)
    SMOKE = (50, 46, 54)
    CYAN = (255, 225, 110)
    AMBER = (40, 190, 255)

    N_FILAMENTS = 44
    N_ORBITALS = 6
    N_SMOKE = 18
    N_EMBERS = 22

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._rng = np.random.default_rng(7)
        theta = np.linspace(0, 2 * np.pi, 72, endpoint=True)
        self._circle = np.column_stack(
            [np.cos(theta), np.sin(theta), np.zeros_like(theta)]).astype(np.float32)
        # orbital ring orientations + per-ring spin speed + colour
        rng = self._rng
        self._orbits = []
        cols = [self.CYAN, self.AMBER, self.WHITE, self.CYAN, self.AMBER, self.WHITE]
        for i in range(self.N_ORBITALS):
            tilt = _rot(rng.uniform(0, np.pi), rng.uniform(0, np.pi),
                        rng.uniform(0, np.pi))
            self._orbits.append((tilt, float(rng.uniform(0.5, 1.4)) * (1 if i % 2 else -1),
                                 cols[i], float(rng.uniform(1.15, 1.5))))
        # ember orbit params
        self._e_ang = rng.uniform(0, 2 * np.pi, self.N_EMBERS).astype(np.float32)
        self._e_rad = rng.uniform(0.5, 1.3, self.N_EMBERS).astype(np.float32)
        self._e_spd = rng.uniform(0.3, 1.2, self.N_EMBERS).astype(np.float32)

    @staticmethod
    def _project(pts: np.ndarray, center: Point, radius: float):
        f = 3.2
        z = pts[:, 2]
        s = f / (f - z)
        xs = center[0] + pts[:, 0] * radius * s
        ys = center[1] - pts[:, 1] * radius * s
        return np.column_stack((xs, ys)).astype(np.int32), z

    # -- main render -----------------------------------------------------

    def render(
        self, frame: np.ndarray, center: Point, radius: float,
        hand_anchors: Optional[Sequence[Point]] = None, intensity: float = 1.0,
    ) -> np.ndarray:
        if intensity <= 0.02 or radius < 4:
            return frame
        intensity = float(min(1.0, intensity))
        t = time.perf_counter() - self._t0
        R = radius * (0.92 + 0.08 * np.sin(t * 4.0))

        # 1) Dark billowing smoke ring (darkens the frame; not additive).
        self._smoke(frame, center, R, t, intensity)

        # 2) Glowing layers -> bloom + crisp.
        soft = np.zeros_like(frame)
        crisp = np.zeros_like(frame)
        self._core(soft, crisp, center, R, t)
        self._sunburst(soft, crisp, center, R, t)
        self._embers(soft, crisp, center, R, t)
        self._orbitals(soft, crisp, center, R, t)
        self._containment(crisp, center, R, t)
        if hand_anchors:
            for a in hand_anchors:
                self._beam(soft, a, center, t, 5)
                self._beam(crisp, a, center, t, 2)

        h, w = frame.shape[:2]
        small = cv2.resize(soft, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(2.0, R * 0.05))
        bloom = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        frame = cv2.addWeighted(frame, 1.0, bloom, intensity, 0)
        frame = cv2.addWeighted(frame, 1.0, crisp, intensity, 0)
        return frame

    # -- smoke (darkening) ----------------------------------------------

    def _smoke(self, frame, center, R, t, intensity) -> None:
        # Work only inside a ROI around the orb (full-frame darken is too slow).
        h, w = frame.shape[:2]
        cx, cy = center
        ext = int(R * 1.45)
        x0, y0 = max(0, cx - ext), max(0, cy - ext)
        x1, y1 = min(w, cx + ext), min(h, cy + ext)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        sw, sh = max(2, rw // 2), max(2, rh // 2)   # build at half ROI res
        smoke = np.zeros((sh, sw, 3), np.uint8)
        mask = np.zeros((sh, sw), np.uint8)
        lcx, lcy = (cx - x0) // 2, (cy - y0) // 2
        rs = R / 2.0
        for i in range(self.N_SMOKE):
            a = 2 * np.pi * i / self.N_SMOKE + t * 0.12
            rr = rs * (0.95 + 0.18 * np.sin(t * 1.5 + i))
            p = (int(lcx + rr * np.cos(a)), int(lcy + rr * np.sin(a)))
            rad = int(rs * 0.42 * (0.7 + 0.4 * np.sin(t * 2 + i * 1.3)))
            shade = 0.7 + 0.3 * np.sin(t * 3 + i)
            cv2.circle(smoke, p, max(1, rad), _scale(self.SMOKE, shade), -1)
            cv2.circle(mask, p, max(1, rad), 255, -1)
        smoke = cv2.GaussianBlur(smoke, (0, 0), sigmaX=max(2.0, rs * 0.12))
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, rs * 0.12))
        smoke = cv2.resize(smoke, (rw, rh))
        a = np.clip(cv2.resize(mask, (rw, rh)).astype(np.float32) * 0.9 * intensity,
                    0, 255).astype(np.uint8)
        a3 = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        inv = cv2.bitwise_not(a3)
        roi[:] = cv2.add(cv2.multiply(roi, inv, scale=1 / 255.0),
                         cv2.multiply(smoke, a3, scale=1 / 255.0))

    # -- fiery core ------------------------------------------------------

    def _fire_grad(self, f: float) -> Color:
        if f < 0.22:
            return _lerp(self.WHITE, self.YELLOW, f / 0.22)
        if f < 0.55:
            return _lerp(self.YELLOW, self.ORANGE, (f - 0.22) / 0.33)
        return _lerp(self.ORANGE, self.RED, (f - 0.55) / 0.45)

    def _core(self, soft, crisp, center, R, t) -> None:
        flick = 0.85 + 0.15 * np.sin(t * 26) * np.cos(t * 11)
        steps = 16
        for k in range(steps, 0, -1):
            f = k / steps
            rr = int(R * 0.82 * f)
            b = ((1 - f) * 0.85 + 0.25) * flick
            cv2.circle(soft, center, max(1, rr), _scale(self._fire_grad(f), b),
                       -1, cv2.LINE_AA)
        # bright iris centre
        cv2.circle(crisp, center, max(2, int(R * 0.16)), self.WHITE, -1, cv2.LINE_AA)
        cv2.circle(crisp, center, max(3, int(R * 0.28)),
                   _scale(self.YELLOW, flick), 2, cv2.LINE_AA)

    def _sunburst(self, soft, crisp, center, R, t) -> None:
        """Radial filaments forming the fiery iris texture."""
        cx, cy = center
        n = self.N_FILAMENTS
        for i in range(n):
            a = 2 * np.pi * i / n + t * 0.12
            ln = 0.35 + 0.45 * abs(np.sin(i * 2.3 + t * 2.0))   # ragged lengths
            r0, r1 = R * 0.16, R * 0.82 * ln
            ca, sa = np.cos(a), np.sin(a)
            p0 = (int(cx + r0 * ca), int(cy + r0 * sa))
            p1 = (int(cx + r1 * ca), int(cy + r1 * sa))
            col = self._fire_grad(0.3 + 0.5 * ln)
            cv2.line(soft, p0, p1, col, 2, cv2.LINE_AA)
            cv2.line(crisp, p0, p1, _scale(col, 0.9), 1, cv2.LINE_AA)

    def _embers(self, soft, crisp, center, R, t) -> None:
        cx, cy = center
        for i in range(self.N_EMBERS):
            ang = self._e_ang[i] + t * self._e_spd[i]
            rr = R * self._e_rad[i] * (0.9 + 0.1 * np.sin(t * 3 + i))
            p = (int(cx + rr * np.cos(ang)), int(cy + rr * np.sin(ang)))
            sz = 1 + int(2 * abs(np.sin(t * 5 + i)))
            cv2.circle(soft, p, sz + 1, self.ORANGE, -1, cv2.LINE_AA)
            cv2.circle(crisp, p, sz, self.YELLOW, -1, cv2.LINE_AA)

    def _orbitals(self, soft, crisp, center, R, t) -> None:
        """Bright swirling atom-like rings with glowing comet heads."""
        for tilt, spd, col, scl in self._orbits:
            pts3d = self._circle @ (tilt @ _rot(0, t * spd, 0)).T
            xy, z = self._project(pts3d, center, R * scl)
            # One polyline per ring (fast) + a glowing comet head for motion.
            cv2.polylines(soft, [xy], True, _scale(col, 0.6), 3, cv2.LINE_AA)
            cv2.polylines(crisp, [xy], True, col, 1, cv2.LINE_AA)
            head = int((t * spd * 14) % len(xy))
            cv2.circle(crisp, tuple(xy[head]), 4, self.WHITE, -1, cv2.LINE_AA)
            cv2.circle(soft, tuple(xy[head]), 7, col, -1, cv2.LINE_AA)

    def _containment(self, crisp, center, R, t) -> None:
        c = (int(center[0]), int(center[1]))
        r = int(R * 1.5)
        cv2.circle(crisp, c, r, _scale(self.CYAN, 0.5), 1, cv2.LINE_AA)
        # specular highlight arcs (glassy sphere)
        cv2.ellipse(crisp, c, (r, r), 0, 200, 250, _scale(self.WHITE, 0.6), 2,
                    cv2.LINE_AA)
        cv2.ellipse(crisp, c, (int(r * 0.97), int(r * 0.97)), 0, 30, 70,
                    _scale(self.WHITE, 0.4), 1, cv2.LINE_AA)

    @staticmethod
    def _beam(buf, src, dst, t, width) -> None:
        src_v = np.array(src, np.float32)
        dst_v = np.array(dst, np.float32)
        normal = np.array([-(dst_v - src_v)[1], (dst_v - src_v)[0]], np.float32)
        nn = np.linalg.norm(normal) or 1.0
        normal /= nn
        pts = [src_v]
        segs = 7
        for s in range(1, segs):
            frac = s / segs
            base = src_v + (dst_v - src_v) * frac
            jit = np.sin(t * 13 + s * 2.1) * 13.0 * (1 - frac)
            pts.append(base + normal * jit)
        pts.append(dst_v)
        cv2.polylines(buf, [np.array(pts, np.int32)], False, (60, 200, 255),
                      width, cv2.LINE_AA)
