"""Static (single-frame) hand-gesture classification.

Two layers live behind one interface:

1. **Rule-based** finger-state detection (this phase). We work out which
   fingers are extended using orientation- and scale-invariant geometry on the
   21 landmarks, then map the finger pattern (+ a few special cases) to a
   gesture name.
2. **Custom ML model** (wired up in Phase 6). When a trained model is present
   it is merged into the same :meth:`GestureClassifier.predict_static`
   interface, so callers never need to know which layer answered.

Public entry point: ``GestureClassifier.predict_static(landmarks) -> (name, conf)``.

Recognised built-ins: open_palm, fist, point, peace, thumbs_up, thumbs_down,
ok, pinch, three, rock. Returns ("none", 0.0) when nothing matches.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Landmark indices (MediaPipe hand model).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (tip, pip, mcp) triplets for the four non-thumb fingers.
_FINGERS = (
    (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    (RING_TIP, RING_PIP, RING_MCP),
    (PINKY_TIP, PINKY_PIP, PINKY_MCP),
)

# Finger-pattern -> gesture name. Pattern is (thumb, index, middle, ring, pinky).
# Direction-/touch-sensitive gestures (thumbs up/down, ok, pinch) are handled
# separately before this table is consulted.
_PATTERNS = {
    (1, 1, 1, 1, 1): "open_palm",
    (0, 0, 0, 0, 0): "fist",
    (0, 1, 0, 0, 0): "point",
    (1, 1, 0, 0, 0): "point",      # index up with thumb out -> still pointing
    (0, 1, 1, 0, 0): "peace",
    (0, 1, 1, 1, 0): "three",
    (1, 1, 1, 1, 0): "three",      # thumb+3 also read as "three"
    (0, 1, 0, 0, 1): "rock",
    (1, 1, 0, 0, 1): "rock",       # thumb out + horns
}


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two landmark points (uses x, y only)."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Make landmarks translation- and scale-invariant.

    Translates so the wrist is the origin and scales by the wrist->middle-MCP
    distance (a stable proxy for hand size). Used both here and by the custom
    trainer so live data matches training data.

    Args:
        landmarks: (21, 3) array of raw normalized landmarks from the tracker.

    Returns:
        (21, 3) float32 array, wrist-centred and unit-scaled.
    """
    pts = landmarks.astype(np.float32).copy()
    pts -= pts[WRIST]
    scale = _dist(pts[MIDDLE_MCP], pts[WRIST])
    if scale < 1e-6:
        scale = 1.0
    return pts / scale


def _finger_states(lm: np.ndarray) -> Tuple[List[int], float]:
    """Return ([thumb, index, middle, ring, pinky], confidence).

    A non-thumb finger is "up" when its tip is farther from the wrist than its
    PIP joint (true regardless of hand rotation). The thumb is "out" when its
    tip sits farther from the index MCP than its IP joint does.

    Confidence is the mean decisiveness of those comparisons: a finger that is
    clearly open or clearly closed contributes ~1.0; a borderline one ~0.5.
    """
    states: List[int] = []
    margins: List[float] = []

    # Thumb: lateral spread away from the index base.
    open_d = _dist(lm[THUMB_TIP], lm[INDEX_MCP])
    closed_d = _dist(lm[THUMB_IP], lm[INDEX_MCP])
    ratio = open_d / closed_d if closed_d > 1e-6 else 1.0
    states.append(1 if ratio > 1.15 else 0)
    margins.append(min(abs(ratio - 1.15) * 2.5, 1.0))

    # Four fingers: tip vs PIP distance from the wrist.
    for tip, pip, _mcp in _FINGERS:
        d_tip = _dist(lm[tip], lm[WRIST])
        d_pip = _dist(lm[pip], lm[WRIST])
        ratio = d_tip / d_pip if d_pip > 1e-6 else 1.0
        states.append(1 if ratio > 1.0 else 0)
        margins.append(min(abs(ratio - 1.0) * 3.0, 1.0))

    confidence = 0.5 + 0.5 * float(np.mean(margins))
    return states, confidence


class GestureClassifier:
    """Static gesture classifier (rule-based now; ML merged in Phase 6)."""

    def __init__(self, touch_threshold: float = 0.45) -> None:
        """
        Args:
            touch_threshold: thumb-index tip distance (in normalized hand-scale
                units) below which the two are considered "touching" -- used to
                distinguish the OK sign and pinch from other poses.
        """
        self._touch_threshold = touch_threshold
        self._ml_model = None          # set in Phase 6
        self._ml_labels: List[str] = []

    def is_open_hand(self, landmarks: np.ndarray) -> bool:
        """Robust "hand is open" test, independent of the thumb.

        Open-palm *classification* needs all five fingers (including the flaky
        thumb), which makes it unreliable as an effect trigger. This instead
        just checks the four reliable fingers (index, middle, ring, pinky):
        if at least three of them are extended, the hand counts as open.
        """
        lm = normalize_landmarks(landmarks)
        states, _ = _finger_states(lm)
        return sum(states[1:]) >= 3  # ignore thumb (states[0])

    def is_pointing(self, landmarks: np.ndarray) -> bool:
        """Robust 'index finger pointing' test, independent of the thumb.

        Index extended while middle/ring/pinky are folded. Used to drive the
        Sudarshan Chakra reliably (the strict 'point' gesture fails whenever the
        thumb happens to read as extended).
        """
        lm = normalize_landmarks(landmarks)
        states, _ = _finger_states(lm)
        return states[1] == 1 and states[2] == 0 and states[3] == 0 and states[4] == 0

    def predict_static(
        self, landmarks: np.ndarray, handedness: Optional[str] = None
    ) -> Tuple[str, float]:
        """Classify a single hand's pose.

        Args:
            landmarks: (21, 3) raw landmarks from :class:`HandData`.
            handedness: optional "Left"/"Right"; unused by the rule layer but
                forwarded to keep a stable signature for the ML layer.

        Returns:
            (gesture_name, confidence in [0, 1]). ("none", 0.0) if no match.
        """
        lm = normalize_landmarks(landmarks)
        states, conf = _finger_states(lm)
        thumb, index, middle, ring, pinky = states

        # Thumb-index touch governs OK vs pinch (checked before the table).
        # A balled fist also brings the thumb near the index tip, so we only
        # treat a touch as pinch when the index is still reaching outward
        # (tip farther from the wrist than its knuckle) -- in a fist the index
        # is curled deep into the palm, so that test fails and we fall through.
        touch = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
        if touch < self._touch_threshold:
            index_reaching = (
                _dist(lm[INDEX_TIP], lm[WRIST]) > _dist(lm[INDEX_MCP], lm[WRIST])
            )
            if middle and ring and pinky:
                return "ok", conf
            if index_reaching and not (middle or ring or pinky):
                return "pinch", conf

        # Thumb-only -> thumbs up/down by vertical direction of the thumb tip.
        if thumb and not any((index, middle, ring, pinky)):
            # Wrist is origin after normalization; image y grows downward.
            return ("thumbs_up" if lm[THUMB_TIP][1] < 0 else "thumbs_down"), conf

        name = _PATTERNS.get(tuple(states))
        if name is not None:
            return name, conf

        # Fall back to the custom ML model if one is loaded (Phase 6).
        if self._ml_model is not None:
            return self._predict_ml(lm)

        return "none", 0.0

    # -- custom ML layer (fully implemented in Phase 6) -------------------

    def load_custom_model(self, model_path: str, labels: List[str]) -> None:
        """Attach a trained custom-gesture model (implemented in Phase 6)."""
        raise NotImplementedError("Custom model loading arrives in Phase 6.")

    def _predict_ml(self, normalized_landmarks: np.ndarray) -> Tuple[str, float]:
        """Run the custom model on normalized landmarks (Phase 6)."""
        raise NotImplementedError("Custom model inference arrives in Phase 6.")
