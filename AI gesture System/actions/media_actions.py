"""Media transport actions via OS media keys.

These send the standard multimedia keyboard keys (play/pause, next, previous),
which virtually every media player and browser honours. Implemented with
pyautogui so they work without a player-specific integration.

Each function returns a short status string for logging / on-screen feedback.
"""

from __future__ import annotations

import pyautogui


def play_pause(**_) -> str:
    """Toggle play / pause of the active media."""
    pyautogui.press("playpause")
    return "play/pause"


def next_track(**_) -> str:
    """Skip to the next track."""
    pyautogui.press("nexttrack")
    return "next track"


def previous_track(**_) -> str:
    """Go to the previous track."""
    pyautogui.press("prevtrack")
    return "previous track"
