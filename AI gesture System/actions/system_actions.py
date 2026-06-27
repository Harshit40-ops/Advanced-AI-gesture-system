"""System-level actions: volume, brightness, screenshot, lock, app launch.

Windows is the primary target. Volume uses **pycaw** for fine-grained,
percentage-based control and falls back to media keys if the COM interface is
unavailable. All OS-specific calls are guarded so importing this module never
fails on another platform; the action simply reports it's unsupported.

Every function returns a short status string for logging / on-screen feedback.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

import pyautogui

_IS_WINDOWS = sys.platform == "win32"

# pyautogui's failsafe (slam mouse to corner to abort) fights with our own
# cursor control in later phases, so disable it globally here.
pyautogui.FAILSAFE = False


# --------------------------------------------------------------------------- #
# Volume (pycaw on Windows, media keys elsewhere / on failure)
# --------------------------------------------------------------------------- #

_volume_iface = None  # cached pycaw endpoint


def _get_volume_interface():
    """Lazily create and cache the pycaw master-volume interface.

    Returns:
        The IAudioEndpointVolume pointer, or None if unavailable.
    """
    global _volume_iface
    if _volume_iface is not None:
        return _volume_iface
    if not _IS_WINDOWS:
        return None
    try:
        # Modern pycaw wraps the default speaker as an AudioDevice and exposes
        # the IAudioEndpointVolume interface directly via .EndpointVolume.
        from pycaw.pycaw import AudioUtilities

        _volume_iface = AudioUtilities.GetSpeakers().EndpointVolume
    except Exception:  # no audio device, COM error, API change, etc.
        _volume_iface = None
    return _volume_iface


def _nudge_volume(delta: float) -> str:
    """Change master volume by ``delta`` (fraction of full scale)."""
    vol = _get_volume_interface()
    if vol is not None:
        try:
            current = vol.GetMasterVolumeLevelScalar()
            new = min(1.0, max(0.0, current + delta))
            vol.SetMasterVolumeLevelScalar(new, None)
            return f"volume {'up' if delta > 0 else 'down'} -> {new:.0%}"
        except Exception:
            pass  # fall through to media key
    key = "volumeup" if delta > 0 else "volumedown"
    pyautogui.press(key)
    return f"volume {'up' if delta > 0 else 'down'} (media key)"


def volume_up(step: float = 0.06, **_) -> str:
    """Raise master volume by ``step`` (default 6%)."""
    return _nudge_volume(abs(step))


def volume_down(step: float = 0.06, **_) -> str:
    """Lower master volume by ``step`` (default 6%)."""
    return _nudge_volume(-abs(step))


def mute(**_) -> str:
    """Toggle master mute."""
    vol = _get_volume_interface()
    if vol is not None:
        try:
            vol.SetMute(not vol.GetMute(), None)
            return "mute toggled"
        except Exception:
            pass
    pyautogui.press("volumemute")
    return "mute toggled (media key)"


# --------------------------------------------------------------------------- #
# Brightness
# --------------------------------------------------------------------------- #

def _nudge_brightness(delta: int) -> str:
    """Change display brightness by ``delta`` percentage points."""
    try:
        import screen_brightness_control as sbc

        current = sbc.get_brightness()
        base = current[0] if isinstance(current, list) else current
        new = min(100, max(0, int(base) + delta))
        sbc.set_brightness(new)
        return f"brightness -> {new}%"
    except Exception as exc:
        return f"brightness unavailable ({exc})"


def brightness_up(step: int = 10, **_) -> str:
    """Raise display brightness by ``step`` percentage points."""
    return _nudge_brightness(abs(step))


def brightness_down(step: int = 10, **_) -> str:
    """Lower display brightness by ``step`` percentage points."""
    return _nudge_brightness(-abs(step))


# --------------------------------------------------------------------------- #
# Screenshot / lock / launch
# --------------------------------------------------------------------------- #

def screenshot(directory: str = "screenshots", **_) -> str:
    """Capture the full screen to a timestamped PNG under ``directory``."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png")
    try:
        img = pyautogui.screenshot()
        img.save(path)
        return f"screenshot saved: {path}"
    except Exception as exc:
        return f"screenshot failed ({exc})"


def lock_screen(**_) -> str:
    """Lock the workstation (Windows)."""
    if _IS_WINDOWS:
        try:
            import ctypes

            ctypes.windll.user32.LockWorkStation()
            return "screen locked"
        except Exception as exc:
            return f"lock failed ({exc})"
    return "lock not supported on this OS"


def launch_app(app: Optional[str] = None, **_) -> str:
    """Launch an application by name or path (e.g. ``"notepad"``)."""
    if not app:
        return "launch_app: no app configured"
    try:
        if _IS_WINDOWS:
            # startfile resolves PATH apps and file associations.
            os.startfile(app)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([app])
        return f"launched: {app}"
    except Exception as exc:
        return f"launch failed ({exc})"
