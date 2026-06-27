"""AI Gesture Control System - entry point and main loop.

Pipeline per frame: capture -> hand tracking -> static gesture classification
-> N-frame stabilization -> config-driven action dispatch (with cooldown,
hold-to-toggle, and smoothed continuous cursor control in mouse mode).

Remaining phases add motion gestures, custom-gesture training, and a GUI.

Run:
    python main.py

Controls:
    q / Esc  - quit
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

from actions import mouse_actions
from core.action_mapper import ActionMapper
from core.gesture_classifier import (
    INDEX_PIP, INDEX_TIP, MIDDLE_MCP, WRIST, GestureClassifier,
)
from core.hand_tracker import HandTracker
from core.smoothing import Cooldown, EMAFilter, GestureStabilizer
from effects.blast_fx import EnergyBlast, TrailFX
from effects.chakra_controller import ChakraController
from effects.elemental_powers import ElementalPowers
from effects.energy_orb import EnergyOrb
from effects.scifi_background import SciFiBackground
from effects.sound_fx import SoundFX
from effects.voice_fx import VoiceFX
from utils.helpers import FPSCounter

# --- tuning constants -------------------------------------------------------
_CONFIRM_FRAMES = 3      # frames a gesture must persist before it counts
_ACTION_COOLDOWN_S = 0.7  # min seconds between repeats of a discrete action
_TOGGLE_HOLD_S = 3.0     # hold open_palm this long to toggle the system
_CURSOR_MARGIN = 0.15    # frame edge cropped from the cursor active region
_EMA_ALPHA = 0.4         # cursor smoothing factor (higher = snappier)

# --- energy-orb effect (both palms open) ------------------------------------
_ORB_EASE = 0.18         # fade-in/out speed of the orb (0..1 per frame)
_ORB_RADIUS_FACTOR = 0.24  # orb radius as a fraction of the inter-hand gap
_ORB_RADIUS_MIN = 55
_ORB_RADIUS_MAX = 240

# --- energy blast (fast single-open-palm thrust) ----------------------------
_BLAST_SPEED = 1100      # px/sec palm speed that fires a force-push blast

# --- Sudarshan Chakra (right-hand index point) ------------------------------
# The displayed frame is mirrored (selfie view), so MediaPipe labels the user's
# physical RIGHT hand as "Left". Flip this to "Right" if it triggers on the
# wrong hand.
_CHAKRA_HAND_LABEL = "Left"
_CHAKRA_GRACE_FRAMES = 6      # keep chakra alive this many frames after a drop
_CHAKRA_RADIUS_FACTOR = 1.0   # radius as a multiple of palm size
_CHAKRA_OFFSET = 0.55     # how far above the fingertip the chakra floats


def open_camera(
    index: int, width: int, height: int, fps: int = 30
) -> cv2.VideoCapture:
    """Open a webcam, requesting a resolution, with a clear error if it fails.

    We request the MJPG pixel format and a target FPS because many Windows
    webcams cap raw (YUY2) capture at 15 FPS; MJPG often unlocks 30. If the
    camera doesn't support it these calls are simply ignored (no harm).

    Args:
        index: Camera device index (0 is usually the built-in webcam).
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
        fps: Requested capture frame rate.

    Returns:
        An opened :class:`cv2.VideoCapture`.

    Raises:
        RuntimeError: If the camera can't be opened (missing or busy).
    """
    # CAP_DSHOW avoids the slow MSMF backend startup on Windows.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. "
            "Is a webcam connected and not in use by another app?"
        )
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Gesture Control System")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=960, help="capture width")
    parser.add_argument("--height", type=int, default=540, help="capture height")
    parser.add_argument(
        "--max-hands", type=int, default=2, help="max hands to track"
    )
    args = parser.parse_args()

    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    tracker = HandTracker(max_num_hands=args.max_hands)
    classifier = GestureClassifier()
    mapper = ActionMapper()
    fps = FPSCounter()
    window = "AI Gesture Control - Phase 4"

    # Temporal smoothing.
    stabilizer = GestureStabilizer(_CONFIRM_FRAMES)
    cooldown = Cooldown(_ACTION_COOLDOWN_S)
    cursor_ema = EMAFilter(_EMA_ALPHA)

    # Visual FX.
    orb = EnergyOrb()
    orb_intensity = 0.0          # eased 0..1 activation
    last_orb = None              # (center, radius, anchors) for graceful fade-out
    chakra_ctl = ChakraController()  # charge + throw state machine
    chakra_grace = 0                 # frames of pointing-persistence remaining
    chakra_last_placement = None     # last good fingertip placement
    sfx = SoundFX()                  # procedural audio (silent if no device)
    chakra_charged = False           # latches the "fully charged" ding
    scifi = SciFiBackground()        # person -> animated sci-fi backdrop
    bg_enabled = False               # toggled with the 'b' key
    powers = ElementalPowers()       # per-gesture fire / ice / lightning
    powers_enabled = True            # toggled with the 'p' key
    trails = TrailFX()               # glowing fingertip motion trails
    trails_on = True                 # toggled with the 't' key
    blast = EnergyBlast()            # force-push energy shockwaves
    blast_cd = Cooldown(0.45)        # debounce blasts
    prev_palm = None                 # previous single-open-palm centre (for speed)
    voice = VoiceFX()                # spoken activation call-outs
    orb_announced = False            # latches the "Reactor activated" voice
    chakra_announced = False         # latches the "Sudarshan Chakra ..." voice

    # App state driven by control gestures.
    current_mode = mapper.start_mode
    system_enabled = True
    toggle_consumed = False  # latches the open-palm hold so it fires once
    last_action_msg = ""
    prev_time = time.perf_counter()  # for per-frame dt (effect physics)

    print("[INFO] Running. Press 'q' or Esc to quit.")
    print(f"[INFO] Modes: {mapper.modes} | start: {current_mode}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Dropped frame from camera.", file=sys.stderr)
                continue

            # Selfie view: mirror so the user's movements feel natural.
            frame = cv2.flip(frame, 1)

            hands = tracker.process(frame)
            # Replace the real background BEFORE drawing skeleton/effects, so
            # those overlay the composited scene (segmentation uses the person).
            if bg_enabled:
                frame = scifi.composite(frame)
            tracker.draw_hands(frame, hands)
            if trails_on:
                frame = trails.render(frame, hands)

            # Classify each detected hand and label it near its wrist.
            primary_gesture = "none"
            primary_conf = 0.0
            open_centers: list[tuple[int, int]] = []  # palm centers of open hands
            chakra_active = False
            chakra_placement = None  # (center, radius, direction)
            hand_gestures = []        # (hand, gesture) for elemental powers
            for i, hand in enumerate(hands):
                gesture, conf = classifier.predict_static(
                    hand.landmarks, hand.handedness
                )
                _label_hand(frame, hand, gesture, conf)
                hand_gestures.append((hand, gesture))
                if i == 0:
                    primary_gesture, primary_conf = gesture, conf
                # Robust open-hand test (thumb-independent) drives the orb, so
                # it triggers reliably whenever both hands are open.
                if classifier.is_open_hand(hand.landmarks):
                    cx, cy = hand.pixels.mean(axis=0).astype(int)
                    open_centers.append((int(cx), int(cy)))
                # Sudarshan Chakra: ANY pointing hand -> chakra on the fingertip.
                # Robust pointing test (thumb-independent) so it triggers even
                # when the thumb is out.
                if classifier.is_pointing(hand.landmarks):
                    chakra_placement = _chakra_placement(hand)
                    chakra_last_placement = chakra_placement
                    chakra_grace = _CHAKRA_GRACE_FRAMES

            # --- Sudarshan Chakra (charge + throw) --------------------------
            # Persistence grace: keep the chakra alive for a few frames after a
            # momentary detection drop, so it doesn't flicker / throw by accident.
            if chakra_grace > 0:
                chakra_active = True
                chakra_grace -= 1
                if chakra_placement is None:
                    chakra_placement = chakra_last_placement

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            chakra_ctl.update(chakra_active, chakra_placement, dt)
            frame = chakra_ctl.render(frame)

            # Chakra audio + voice: whir while charging, ding when full,
            # whoosh on throw, and a spoken call-out when it first appears.
            charging = chakra_ctl.state == ChakraController.CHARGING
            if charging and not chakra_announced:
                voice.say("Sudarshan Chakra activated")
                chakra_announced = True
            elif chakra_ctl.state == ChakraController.IDLE:
                chakra_announced = False
            sfx.set_chakra(chakra_ctl.charge if charging else 0.0)
            if chakra_ctl.just_launched:
                sfx.whoosh()
            if charging and chakra_ctl.charge >= 1.0 and not chakra_charged:
                sfx.ding()
                chakra_charged = True
            if not charging:
                chakra_charged = False

            # --- Energy orb when BOTH hands are open ------------------------
            both_open = len(open_centers) >= 2
            if both_open:
                c1, c2 = open_centers[0], open_centers[1]
                gap = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                radius = int(min(_ORB_RADIUS_MAX,
                                 max(_ORB_RADIUS_MIN, gap * _ORB_RADIUS_FACTOR)))
                center = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                last_orb = (center, radius, [c1, c2])
            target = 1.0 if both_open else 0.0
            orb_intensity += (target - orb_intensity) * _ORB_EASE
            sfx.set_orb(orb_intensity)  # looping energy hum tracks the orb
            if orb_intensity > 0.6 and not orb_announced:
                sfx.reactor_powerup()          # cinematic spin-up
                voice.say("Reactor activated")
                orb_announced = True
            elif orb_intensity < 0.1:
                orb_announced = False
            if orb_intensity > 0.02 and last_orb is not None:
                oc, orad, oanchors = last_orb
                # render() returns a NEW composited frame -> must reassign.
                frame = orb.render(frame, oc, orad, oanchors, orb_intensity)
            elif len(open_centers) == 1:
                # Guide the user: one open hand seen, need the other.
                _put_text(frame, "Dono haath kholo -> Energy Orb",
                          (open_centers[0][0] - 120, open_centers[0][1] - 40),
                          scale=0.6, color=(0, 255, 255))

            # --- Elemental powers (fist=fire, peace=ice, rock=lightning) ----
            if powers_enabled:
                frame = powers.render(frame, hand_gestures)

            # --- Energy blast: fast thrust of a single open palm ------------
            single_palm = open_centers[0] if len(open_centers) == 1 else None
            if single_palm is not None and prev_palm is not None and dt > 0:
                v = (np.array(single_palm, np.float32) - prev_palm) / dt
                speed = float(np.linalg.norm(v))
                if speed > _BLAST_SPEED and blast_cd.ready("blast"):
                    blast.fire(single_palm, v / speed, frame.shape[0] * 0.12)
                    blast_cd.mark("blast")
                    sfx.whoosh()
            prev_palm = (np.array(single_palm, np.float32)
                         if single_palm is not None else None)
            frame = blast.render(frame, dt)

            # N-frame confirmation: flicker-free stable gesture.
            raw = primary_gesture if hands else "none"
            stable = stabilizer.update(raw)

            # Reset the toggle latch whenever we stop holding an open palm.
            if stable != "open_palm":
                toggle_consumed = False

            hud_hint = ""

            # --- Continuous cursor control (mouse mode + point) -------------
            # Runs every frame, bypassing the cooldown for smooth motion.
            moved_cursor = False
            if system_enabled and current_mode == "mouse" and stable == "point" and hands:
                tip = hands[0].landmarks[INDEX_TIP]
                sx, sy = mouse_actions.normalized_to_screen(
                    float(tip[0]), float(tip[1]), _CURSOR_MARGIN
                )
                fx, fy = cursor_ema.filter(sx, sy)
                mouse_actions.move_cursor(fx, fy)
                moved_cursor = True
                last_action_msg = "cursor move"
            else:
                cursor_ema.reset()  # avoid a glide-in when re-entering move mode

            # --- Discrete actions ------------------------------------------
            if not moved_cursor and stable != "none":
                entry = mapper.resolve(current_mode, stable)
                if entry is not None:
                    action = entry.get("action", "")
                    allowed = system_enabled or action == "toggle_system"

                    if allowed and action == "toggle_system" and not both_open:
                        # Hold-to-toggle: fire once after a sustained hold.
                        # Suppressed while both palms are open (orb mode).
                        held = stabilizer.held_seconds()
                        if not toggle_consumed:
                            hud_hint = (
                                f"hold to toggle: "
                                f"{min(held, _TOGGLE_HOLD_S):.1f}/{_TOGGLE_HOLD_S:.0f}s"
                            )
                        if held >= _TOGGLE_HOLD_S and not toggle_consumed:
                            system_enabled = not system_enabled
                            toggle_consumed = True
                            last_action_msg = (
                                f"system {'ON' if system_enabled else 'OFF'}"
                            )
                            print(f"[ACTION] {last_action_msg}")
                    elif allowed and action == "mouse_move":
                        pass  # handled continuously above
                    elif allowed and cooldown.ready(stable):
                        result = mapper.dispatch(current_mode, stable)
                        if result.fired:
                            cooldown.mark(stable)
                            if result.control == "switch_mode":
                                idx = mapper.modes.index(current_mode)
                                current_mode = mapper.modes[
                                    (idx + 1) % len(mapper.modes)
                                ]
                                last_action_msg = f"mode -> {current_mode}"
                            else:
                                last_action_msg = result.message
                            print(f"[ACTION] {last_action_msg}")

            fps.tick()
            _draw_hud(
                frame, fps.fps, len(hands), primary_gesture, primary_conf,
                current_mode, system_enabled, last_action_msg, hud_hint,
            )
            _put_text(frame, f"[b] BG: {'ON' if bg_enabled else 'OFF'}",
                      (frame.shape[1] - 190, 30), scale=0.6,
                      color=(255, 200, 0) if bg_enabled else (160, 160, 160))

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                break
            if key == ord("b"):        # toggle sci-fi background
                bg_enabled = not bg_enabled
                print(f"[INFO] Sci-fi background: {'ON' if bg_enabled else 'OFF'}")
            if key == ord("p"):        # toggle elemental powers
                powers_enabled = not powers_enabled
                print(f"[INFO] Elemental powers: {'ON' if powers_enabled else 'OFF'}")
            if key == ord("t"):        # toggle fingertip trails
                trails_on = not trails_on
                print(f"[INFO] Fingertip trails: {'ON' if trails_on else 'OFF'}")
            # If the user clicks the window's X button, the window vanishes.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        tracker.close()
        sfx.close()
        voice.close()
        scifi.close()
        cv2.destroyAllWindows()

    print("[INFO] Shut down cleanly.")
    return 0


def _put_text(frame, text, org, scale=0.8, color=(0, 255, 0)) -> None:
    """Draw text with a black outline so it stays readable on any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def _draw_hud(
    frame, fps_value: float, num_hands: int,
    gesture: str = "none", confidence: float = 0.0,
    mode: str = "", system_enabled: bool = True, last_action: str = "",
    hint: str = "",
) -> None:
    """Draw FPS, hand count, gesture, mode, system state, and last action."""
    status = "ON" if system_enabled else "OFF"
    status_color = (0, 255, 0) if system_enabled else (0, 0, 255)
    lines = [
        (f"FPS: {fps_value:5.1f}", (0, 255, 0)),
        (f"Hands: {num_hands}", (0, 255, 0)),
        (f"Gesture: {gesture} ({confidence:.0%})", (0, 255, 0)),
        (f"Mode: {mode}", (255, 200, 0)),
        (f"System: {status}", status_color),
    ]
    y = 30
    for text, color in lines:
        _put_text(frame, text, (12, y), color=color)
        y += 34
    if hint:
        _put_text(frame, hint, (12, y), scale=0.7, color=(0, 255, 255))
        y += 30
    if last_action:
        _put_text(frame, f"> {last_action}", (12, y), scale=0.6,
                  color=(200, 200, 255))


def _chakra_placement(hand):
    """Compute (center, radius, direction) for the chakra above the index finger.

    Sizes the chakra to the hand and offsets it along the finger direction so it
    appears to spin just above the fingertip. ``direction`` is the unit vector
    the finger points in (used to launch the chakra on a throw).
    """
    tip = hand.pixels[INDEX_TIP].astype(np.float32)
    pip = hand.pixels[INDEX_PIP].astype(np.float32)
    wrist = hand.pixels[WRIST].astype(np.float32)
    mmcp = hand.pixels[MIDDLE_MCP].astype(np.float32)

    scale = float(np.linalg.norm(wrist - mmcp))  # palm size in pixels
    radius = max(40.0, scale * _CHAKRA_RADIUS_FACTOR)

    direction = tip - pip
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-6 else np.array([0.0, -1.0], np.float32)
    center = tip + direction * (radius * _CHAKRA_OFFSET)
    return center, radius, direction


def _label_hand(frame, hand, gesture: str, confidence: float) -> None:
    """Annotate an individual hand with its handedness and recognized gesture."""
    x, y = int(hand.pixels[WRIST][0]), int(hand.pixels[WRIST][1])
    label = f"{hand.handedness}: {gesture} {confidence:.0%}"
    _put_text(frame, label, (x - 20, y + 28), scale=0.6, color=(0, 255, 255))


if __name__ == "__main__":
    raise SystemExit(main())
