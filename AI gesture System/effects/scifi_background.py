"""Replace the real background with an animated sci-fi scene.

Uses MediaPipe's selfie ImageSegmenter to cut the person out of the webcam
frame and composites them over a procedurally animated backdrop: a deep-space
gradient with drifting nebula glow, a twinkling starfield, and a scrolling
perspective "Tron" grid floor. The person's silhouette gets a glowing cyan
rim-light for a holographic feel.

Toggle it from the main loop (default off). Pure NumPy + OpenCV + MediaPipe.
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision

_BaseOptions = mp.tasks.BaseOptions

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "selfie_segmenter.tflite"
)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


class SciFiBackground:
    """Segment the person and drop them into an animated sci-fi world."""

    def __init__(self, model_path: Optional[str] = None,
                 auto_download: bool = True) -> None:
        self._model_path = model_path or _DEFAULT_MODEL
        self._ensure_model(auto_download)
        options = vision.ImageSegmenterOptions(
            base_options=_BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._seg = vision.ImageSegmenter.create_from_options(options)

        self._t0 = time.perf_counter()
        self._base: Optional[np.ndarray] = None      # cached static backdrop
        self._size: Tuple[int, int] = (0, 0)
        self._stars = np.empty((0, 3), np.float32)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # -- model -----------------------------------------------------------

    def _ensure_model(self, auto_download: bool) -> None:
        if os.path.exists(self._model_path):
            return
        if not auto_download:
            raise FileNotFoundError(f"Segmenter model missing: {self._model_path}")
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        print(f"[INFO] Downloading selfie segmenter to {self._model_path} ...")
        urllib.request.urlretrieve(_MODEL_URL, self._model_path)
        print("[INFO] Segmenter model ready.")

    # -- background generation ------------------------------------------

    def _build_base(self, w: int, h: int) -> None:
        """Precompute the static parts of the backdrop (gradient, nebula, stars)."""
        # Vertical gradient: deep purple-blue at top -> near black at bottom.
        top = np.array([60, 20, 35], np.float32)     # BGR
        bot = np.array([15, 8, 12], np.float32)
        ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        col = (top[None, :] * (1 - ramp) + bot[None, :] * ramp)
        base = np.repeat(col[:, None, :], w, axis=1).astype(np.uint8)

        # Nebula: a few big soft colour blobs.
        glow = np.zeros_like(base)
        rng = np.random.default_rng(42)
        nebula_cols = [(255, 120, 40), (200, 60, 160), (255, 200, 90)]
        for c in nebula_cols:
            cx, cy = rng.integers(0, w), rng.integers(0, int(h * 0.7))
            cv2.circle(glow, (int(cx), int(cy)), int(w * 0.22), c, -1)
        glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=w * 0.08)
        base = cv2.addWeighted(base, 1.0, glow, 0.5, 0)

        # Starfield positions (x, y normalized, brightness).
        n = 220
        self._stars = np.column_stack([
            rng.random(n), rng.random(n), rng.uniform(0.3, 1.0, n)
        ]).astype(np.float32)
        for sx, sy, sb in self._stars:
            p = (int(sx * w), int(sy * h))
            cv2.circle(base, p, 1, (int(255 * sb),) * 3, -1)

        self._base = base
        self._size = (w, h)

    def _animated_bg(self, w: int, h: int) -> np.ndarray:
        """Base backdrop + per-frame animation (scrolling grid + twinkles)."""
        if self._base is None or self._size != (w, h):
            self._build_base(w, h)
        t = time.perf_counter() - self._t0
        img = self._base.copy()

        # Scrolling perspective grid floor (Tron-style).
        horizon = int(h * 0.60)
        cx = w // 2
        gcol = (200, 120, 30)
        for i in range(-12, 13):                       # converging verticals
            x_bottom = cx + i * (w // 12)
            cv2.line(img, (cx + i * 6, horizon), (x_bottom, h), gcol, 1, cv2.LINE_AA)
        n = 14
        for k in range(n):                             # receding horizontals
            f = ((k + (t * 0.25) % 1.0) / n)
            y = int(horizon + (h - horizon) * (f * f))
            cv2.line(img, (0, y), (w, y), gcol, 1, cv2.LINE_AA)

        # Twinkle a handful of stars.
        tw = (np.sin(t * 6 + self._stars[:, 0] * 30) > 0.6)
        for sx, sy, _ in self._stars[tw][:40]:
            cv2.circle(img, (int(sx * w), int(sy * h)), 1, (255, 255, 255), -1)
        return img

    # -- composite -------------------------------------------------------

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Cut the person from ``frame`` and place them over the sci-fi scene.

        Segmentation + mask work happen at a reduced resolution and the
        compositing uses C-optimised cv2 ops (not NumPy float math), which keeps
        this ~3x faster than the naive version.
        """
        h, w = frame.shape[:2]
        sw, sh = 640, 360

        # Segment at low res.
        small = cv2.resize(frame, (sw, sh))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        m = self._seg.segment(mp_img).confidence_masks[0].numpy_view()  # (sh,sw)

        # Silhouette edge (for rim light) computed at low res, then upscaled.
        m8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
        edge = cv2.morphologyEx(m8, cv2.MORPH_GRADIENT, self._kernel)
        edge = cv2.resize(edge, (w, h))

        # Upscale + feather the alpha mask.
        a = cv2.resize(m8, (w, h))
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=2.0)
        a3 = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        inv = cv2.bitwise_not(a3)

        bg = self._animated_bg(w, h)
        fg = cv2.multiply(frame, a3, scale=1 / 255.0)
        bgp = cv2.multiply(bg, inv, scale=1 / 255.0)
        out = cv2.add(fg, bgp)

        # Holographic cyan rim light on the silhouette.
        rim = np.zeros_like(out)
        rim[:, :, 0] = edge
        rim[:, :, 1] = edge
        out = cv2.addWeighted(out, 1.0, rim, 0.7, 0)
        return out

    def close(self) -> None:
        try:
            self._seg.close()
        except Exception:
            pass
