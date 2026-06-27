"""MediaPipe Hands wrapper (Tasks API).

MediaPipe >= 0.10.3x ships *only* the new "Tasks" API; the classic
``mp.solutions.hands`` module has been removed. This wrapper uses
``mediapipe.tasks.vision.HandLandmarker`` and hides it behind a small, typed
interface so the rest of the system only deals with plain Python objects
(:class:`HandData`) -- never MediaPipe internals.

It handles:

* loading (and auto-downloading) the ``hand_landmarker.task`` model,
* hand detection + 21 landmark extraction for one or both hands,
* handedness (Left / Right) labelling,
* both *normalized* (0..1) and *pixel* landmark coordinates,
* drawing the landmark / connection overlay onto a frame.

It deliberately knows nothing about gestures or actions.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Tasks API handles.
_BaseOptions = mp.tasks.BaseOptions
_HandLandmarker = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode

# Default model location + official download URL (used if the file is missing).
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task"
)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Canonical MediaPipe hand skeleton: 21 connections between the 21 landmarks.
# Defined explicitly so we don't depend on any MediaPipe drawing internals.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)


@dataclass
class HandData:
    """All per-hand information the rest of the system needs.

    Attributes:
        landmarks: (21, 3) array of normalized x, y, z. x/y are in [0, 1]
            relative to the frame; z is MediaPipe's relative depth. Position
            and scale are *not* normalized here -- the classifier does that
            when it needs position/size invariance.
        pixels: (21, 2) int array of x, y pixel coordinates in the frame.
        handedness: "Left" or "Right" as perceived in the (mirrored) display.
        score: Handedness classification confidence in [0, 1].
    """

    landmarks: np.ndarray
    pixels: np.ndarray
    handedness: str
    score: float


class HandTracker:
    """Typed wrapper around ``mediapipe.tasks.vision.HandLandmarker``."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
        min_presence_confidence: float = 0.4,
        model_path: Optional[str] = None,
        auto_download: bool = True,
    ) -> None:
        """Create the HandLandmarker in VIDEO running mode.

        Args:
            max_num_hands: Maximum hands to track simultaneously.
            min_detection_confidence: Palm-detector confidence threshold.
            min_tracking_confidence: Landmark-tracking confidence threshold.
            min_presence_confidence: Hand-presence confidence threshold.
            model_path: Path to ``hand_landmarker.task``. Defaults to
                ``<project>/models/hand_landmarker.task``.
            auto_download: If True and the model file is missing, download it.

        Raises:
            FileNotFoundError: If the model is missing and can't be downloaded.
        """
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._ensure_model(auto_download)

        # IMAGE mode runs full palm detection on EVERY frame. This is heavier
        # than VIDEO mode (which tracks and only re-detects occasionally) but is
        # far more reliable at picking up BOTH hands every frame -- VIDEO mode
        # tends to lock onto one hand and miss the second, especially when the
        # hands are close together. We have plenty of FPS headroom for it.
        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=self._model_path),
            running_mode=_RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)

    # -- model management -------------------------------------------------

    def _ensure_model(self, auto_download: bool) -> None:
        """Make sure the .task model exists locally, downloading if allowed."""
        if os.path.exists(self._model_path):
            return
        if not auto_download:
            raise FileNotFoundError(
                f"Hand model not found at {self._model_path}. "
                "Download hand_landmarker.task or pass auto_download=True."
            )
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        print(f"[INFO] Downloading hand model to {self._model_path} ...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, self._model_path)
        except Exception as exc:  # network/permission errors -> clear message
            raise FileNotFoundError(
                f"Could not download hand model from {_MODEL_URL}: {exc}"
            ) from exc
        print("[INFO] Model download complete.")

    # -- inference --------------------------------------------------------

    def process(self, frame_bgr: np.ndarray,
                infer_width: int = 480) -> List[HandData]:
        """Detect hands in a BGR frame and return structured results.

        Detection runs on a downscaled copy (``infer_width`` px wide) for speed;
        because MediaPipe landmarks are normalized (0..1) we map them back to the
        full frame's pixel size, so accuracy on screen is unaffected. This is a
        big FPS win vs. running the model on the full-resolution frame.

        Args:
            frame_bgr: An OpenCV BGR image (already mirrored if you want a
                selfie view; handedness labels follow whatever you pass in).
            infer_width: Width to downscale to before detection.

        Returns:
            A list of :class:`HandData`, one per detected hand (possibly empty).
        """
        h, w = frame_bgr.shape[:2]
        if w > infer_width:
            small = cv2.resize(frame_bgr, (infer_width, int(h * infer_width / w)))
        else:
            small = frame_bgr
        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect(mp_image)

        hands: List[HandData] = []
        if not result.hand_landmarks:
            return hands

        for idx, hand_lms in enumerate(result.hand_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32
            )
            pixels = np.column_stack(
                (coords[:, 0] * w, coords[:, 1] * h)
            ).astype(np.int32)

            label, score = "Unknown", 0.0
            if result.handedness and idx < len(result.handedness):
                cat = result.handedness[idx][0]
                label, score = cat.category_name, float(cat.score)

            hands.append(
                HandData(
                    landmarks=coords,
                    pixels=pixels,
                    handedness=label,
                    score=score,
                )
            )
        return hands

    # -- rendering --------------------------------------------------------

    @staticmethod
    def draw_hands(frame_bgr: np.ndarray, hands: List[HandData]) -> np.ndarray:
        """Draw landmarks and bone connections for each hand onto the frame.

        Uses our own renderer built from pixel coords, so it's independent of
        any MediaPipe drawing utilities.
        """
        for hand in hands:
            for start, end in HAND_CONNECTIONS:
                x1, y1 = hand.pixels[start]
                x2, y2 = hand.pixels[end]
                cv2.line(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 255, 255), 2, cv2.LINE_AA)
            for x, y in hand.pixels:
                cv2.circle(frame_bgr, (int(x), int(y)), 4, (0, 200, 255), -1)
        return frame_bgr

    def close(self) -> None:
        """Release the MediaPipe graph (call on shutdown)."""
        self._landmarker.close()
