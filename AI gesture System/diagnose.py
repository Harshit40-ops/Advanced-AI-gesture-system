"""Standalone diagnostic: runs the full perception pipeline for ~15 seconds,
shows the live window, and writes what it sees to `diagnose_log.txt`.

Use this when "kuch nahi ho raha" -- it records, per moment, how many hands
were detected, the recognised gesture, whether each hand counts as "open", and
the orb activation. Run it, wave/​show BOTH hands for the full 15s, then let the
developer read diagnose_log.txt.

Run:
    python diagnose.py
"""

from __future__ import annotations

import sys
import time

import cv2

from core.gesture_classifier import GestureClassifier
from core.hand_tracker import HandTracker
from main import open_camera, _put_text

DURATION_S = 15
LOG_PATH = "diagnose_log.txt"


def main() -> int:
    log: list[str] = []

    def record(msg: str) -> None:
        print(msg, flush=True)
        log.append(msg)

    record(f"[diag] python {sys.version.split()[0]} | platform {sys.platform}")

    try:
        cap = open_camera(0, 1280, 720)
    except RuntimeError as exc:
        record(f"[diag] CAMERA FAILED: {exc}")
        _flush(log)
        return 1
    record("[diag] camera opened OK")

    tracker = HandTracker(max_num_hands=2)
    classifier = GestureClassifier()
    record("[diag] tracker + classifier ready; starting 15s capture")

    window = "Diagnostic - show BOTH open hands"
    start = time.perf_counter()
    last_log = 0.0
    frames = 0
    max_hands_seen = 0
    both_open_frames = 0

    try:
        while time.perf_counter() - start < DURATION_S:
            ok, frame = cap.read()
            if not ok or frame is None:
                record("[diag] dropped frame")
                continue
            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)
            tracker.draw_hands(frame, hands)
            frames += 1
            max_hands_seen = max(max_hands_seen, len(hands))

            gestures = []
            open_count = 0
            for hand in hands:
                g, c = classifier.predict_static(hand.landmarks, hand.handedness)
                gestures.append(f"{hand.handedness[:1]}:{g}")
                if classifier.is_open_hand(hand.landmarks):
                    open_count += 1
            if open_count >= 2:
                both_open_frames += 1

            elapsed = time.perf_counter() - start
            _put_text(frame, f"hands={len(hands)} open={open_count}",
                      (12, 36), color=(0, 255, 0))
            _put_text(frame, f"t={elapsed:0.1f}s  show BOTH open hands",
                      (12, 72), scale=0.6, color=(0, 255, 255))
            cv2.imshow(window, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

            # Log a snapshot ~3x/second.
            if elapsed - last_log >= 0.3:
                last_log = elapsed
                record(f"[diag] t={elapsed:4.1f}s hands={len(hands)} "
                       f"open={open_count} gestures={gestures}")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    record("[diag] ===== SUMMARY =====")
    record(f"[diag] frames processed : {frames}")
    record(f"[diag] max hands seen   : {max_hands_seen}")
    record(f"[diag] frames both-open : {both_open_frames}")
    if max_hands_seen == 0:
        record("[diag] VERDICT: no hands detected at all (camera/lighting issue)")
    elif max_hands_seen == 1:
        record("[diag] VERDICT: only ONE hand ever detected -> orb needs TWO")
    elif both_open_frames == 0:
        record("[diag] VERDICT: two hands seen but never both 'open' together")
    else:
        record("[diag] VERDICT: two open hands WERE detected -> orb should show")
    _flush(log)
    return 0


def _flush(log: list[str]) -> None:
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print(f"[diag] wrote {LOG_PATH}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
