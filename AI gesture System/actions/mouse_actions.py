"""Mouse-control actions: cursor move, clicks, drag, scroll.

Cursor positioning maps a normalized hand coordinate (0..1 within the camera
frame) to an absolute screen pixel. A central *active region* is mapped to the
full screen so the user doesn't have to reach the frame edges (which are hard
to hit and where tracking is least reliable).

Each function returns a short status string for logging / on-screen feedback.
"""

from __future__ import annotations

from typing import Tuple

import pyautogui

# Continuous cursor control needs zero per-call delay and no corner-failsafe.
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

_SCREEN_W, _SCREEN_H = pyautogui.size()


def screen_size() -> Tuple[int, int]:
    """Return (width, height) of the primary screen in pixels."""
    return _SCREEN_W, _SCREEN_H


def normalized_to_screen(
    nx: float, ny: float, margin: float = 0.15
) -> Tuple[int, int]:
    """Map a normalized frame point to absolute screen pixels.

    The square region ``[margin, 1 - margin]`` in the frame is stretched to
    cover the whole screen, so small comfortable hand movements span the
    display and the unreliable frame edges are excluded.

    Args:
        nx: Normalized x in [0, 1] (already mirrored to match the display).
        ny: Normalized y in [0, 1].
        margin: Fraction of the frame to crop off each side.

    Returns:
        (x, y) screen pixel coordinates, clamped to the screen bounds.
    """
    span = max(1e-6, 1.0 - 2.0 * margin)
    x = min(max((nx - margin) / span, 0.0), 1.0)
    y = min(max((ny - margin) / span, 0.0), 1.0)
    return int(x * _SCREEN_W), int(y * _SCREEN_H)


def move_cursor(x: int, y: int, **_) -> str:
    """Move the cursor to absolute screen pixel (x, y)."""
    pyautogui.moveTo(int(x), int(y))
    return "cursor move"


def left_click(**_) -> str:
    """Single left click at the current cursor position."""
    pyautogui.click()
    return "left click"


def right_click(**_) -> str:
    """Single right click at the current cursor position."""
    pyautogui.click(button="right")
    return "right click"


def double_click(**_) -> str:
    """Double left click at the current cursor position."""
    pyautogui.doubleClick()
    return "double click"


def drag_start(**_) -> str:
    """Press and hold the left button (begin a drag)."""
    pyautogui.mouseDown()
    return "drag start"


def drag_end(**_) -> str:
    """Release the left button (end a drag)."""
    pyautogui.mouseUp()
    return "drag end"


def scroll_up(amount: int = 300, **_) -> str:
    """Scroll the view up."""
    pyautogui.scroll(int(amount))
    return "scroll up"


def scroll_down(amount: int = 300, **_) -> str:
    """Scroll the view down."""
    pyautogui.scroll(-int(amount))
    return "scroll down"
