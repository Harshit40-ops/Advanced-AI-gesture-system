# AI Gesture Control System

Real-time hand-gesture control for your PC using a standard webcam
(OpenCV + MediaPipe). Built in phases — see `Build status` below.

> Full gesture guide, training instructions, and troubleshooting are added in
> Phase 8. This README currently covers setup and the phases completed so far.

## Requirements

- Python 3.10+ (developed/tested on **Python 3.14**, Windows 11)
- A webcam

## Setup

```powershell
pip install -r requirements.txt
```

This installs MediaPipe (which also pulls in OpenCV) and NumPy for the current
phase, plus the libraries used by later phases.

The MediaPipe hand model (`models/hand_landmarker.task`, ~7.8 MB) is downloaded
automatically on first run if it isn't already present.

## Run (Phase 4 — stable control + mouse)

```powershell
python main.py
```

Optional flags:

```powershell
python main.py --camera 0 --width 1280 --height 720 --max-hands 2
```

You should see your webcam feed (mirrored), a white hand skeleton with orange
joints, an FPS / hand-count HUD top-left, and the **recognized gesture name +
confidence** both in the HUD (primary hand) and next to each hand's wrist.

Recognized static gestures: `open_palm`, `fist`, `point`, `peace`, `three`,
`rock`, `thumbs_up`, `thumbs_down`, `ok`, `pinch`.

### Active actions (Phase 3)

Gestures are mapped to actions in [config/gestures.json](config/gestures.json),
resolved as `active mode → global → any`. Currently wired and working:

| Mode    | Gesture     | Action                        |
|---------|-------------|-------------------------------|
| global  | open_palm   | toggle system ON/OFF          |
| global  | rock        | switch mode (cycles)          |
| any     | three       | screenshot (→ `screenshots/`) |
| media   | thumbs_up   | volume up (fine-grained)      |
| media   | thumbs_down | volume down                   |
| media   | fist        | play / pause                  |
| media   | ok          | mute toggle                   |

### Mouse mode (Phase 4)

Switch to `mouse` mode (show `rock` once to cycle modes), then:

| Gesture       | Action                          |
|---------------|---------------------------------|
| point (index) | move cursor (EMA-smoothed)      |
| pinch         | left click                      |
| peace (V)     | right click                     |

Cursor mapping uses a central *active region* (default: 15% cropped off each
edge) stretched to the full screen, so small comfortable movements span the
display. Tune `_CURSOR_MARGIN` / `_EMA_ALPHA` in `main.py`.

### ✨ Energy Orb (futuristic FX)

Open **both hands** (open palm) and face them at the camera: a glowing,
rotating **3-D energy sphere** materialises between them — wireframe globe, hot
core, orbiting particles, and lightning energy beams running from each hand to
the orb. Move your hands apart/together and the orb grows/shrinks; drop a hand
and it fades out smoothly.

- Implemented in [effects/energy_orb.py](effects/energy_orb.py) (pure
  NumPy + OpenCV — no GPU/OpenGL needed).
- Tuning in `main.py`: `_ORB_RADIUS_FACTOR`, `_ORB_RADIUS_MIN/MAX`, `_ORB_EASE`.
- Renders only while both palms are visible, so it doesn't cost FPS otherwise.

### ⚡🔥❄️ Elemental powers (press `p` to toggle, default on)

Each gesture summons an element around the hand:

| Gesture     | Element                                  |
|-------------|------------------------------------------|
| fist ✊     | 🔥 fire erupting from palm + fingertips   |
| peace ✌️    | ❄️ ice shards + frost sparkles            |
| rock 🤘     | ⚡ lightning arcing across the fingers     |

Implemented in [effects/elemental_powers.py](effects/elemental_powers.py)
(~11 ms/frame, all hands composited in one bloom pass).

### 🗣️ Voice call-outs

A synthesized voice announces activations (Windows SAPI, non-blocking):
- Both palms open → **"Reactor activated"**
- Right-hand point → **"Sudarshan Chakra activated"**

See [effects/voice_fx.py](effects/voice_fx.py).

### 🌌 Sci-fi background (press `b`)

Press **`b`** to replace your real room with an animated synthwave scene —
nebula sky, twinkling starfield, and a scrolling Tron-style perspective grid —
with you cut out and given a glowing holographic rim-light. Press `b` again to
turn it off.

- Implemented in [effects/scifi_background.py](effects/scifi_background.py)
  using MediaPipe's selfie ImageSegmenter (`models/selfie_segmenter.tflite`,
  auto-downloaded).
- Runs segmentation at reduced resolution + cv2-accelerated compositing
  (~22 ms/frame) so it stays usable alongside hand tracking.

### 🔱 Sudarshan Chakra (right-hand index point)

Raise **only your right-hand index finger** (the `point` gesture): a divine
**spinning flaming chakra** materialises and floats above your fingertip —
tilted ornate wheel, golden bloom, and a roaring fire ring.

- Implemented in [effects/sudarshan_chakra.py](effects/sudarshan_chakra.py).
- Triggered on the hand MediaPipe labels per `_CHAKRA_HAND_LABEL` in `main.py`
  (default `"Left"`, which is your physical right hand because the view is
  mirrored). Flip to `"Right"` if it appears on the wrong hand.

### Stability & the system toggle (Phase 4)

- **N-frame confirmation:** a gesture must persist a few frames before it
  counts (no single-frame flicker).
- **Cooldown:** discrete actions won't repeat faster than ~0.7s.
- **Hold-to-toggle:** `open_palm` now toggles the system only after being held
  for **3 seconds** — the HUD shows a `hold to toggle: x.x/3s` countdown.
- Swipe / scroll / presentation actions arrive in Phase 5 (mapper warns and
  no-ops until then).

Controls:
- `q` or `Esc` — quit

## Build status

- [x] **Phase 1** — Webcam capture, MediaPipe hand tracking, landmark overlay, live FPS
- [x] **Phase 2** — Static gesture recognition (10 rule-based gestures)
- [x] **Phase 3** — Action mapping (config-driven; volume / media / screenshot / modes)
- [x] **Phase 4** — Smoothing/debounce, hold-to-toggle, EMA mouse control + pinch-click
- [x] **Bonus** — ✨ Energy Orb 3-D visual FX (both palms open)
- [ ] Phase 5 — Dynamic / motion gestures
- [ ] Phase 6 — Custom gesture training
- [ ] Phase 7 — GUI dashboard
- [ ] Phase 8 — Config, modes & polish

## Notes / troubleshooting

- **FPS feels low (~15):** Hand-tracking inference itself runs ~50 FPS on CPU;
  if you see ~15 FPS it's almost always the *camera* capping its output (common
  on budget webcams or in dim light, where auto-exposure lowers the frame
  rate). Try better lighting or a different `--camera` index. `main.py` already
  requests MJPG @ 30 FPS, which unlocks higher rates on cameras that support it.
- **"Could not open camera":** Another app may be using the webcam, or the
  index is wrong — try `--camera 1`.
- The first frame is slow (model warm-up); this is normal.
