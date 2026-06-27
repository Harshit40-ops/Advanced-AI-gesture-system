"""Config-driven gesture -> action dispatch.

Loads ``config/gestures.json`` and turns a recognized ``(mode, gesture)`` pair
into a concrete action call. Resolution order for a gesture is:

1. the active mode's mappings (e.g. ``media``),
2. the ``global`` mappings (mode-independent, e.g. toggle / switch mode),
3. the ``any`` mappings (work in every mode, e.g. screenshot).

Two action names are *control* actions handled by the app rather than the
``actions/`` package: ``toggle_system`` and ``switch_mode``. The mapper reports
these back via :class:`ActionResult` so the main loop can update app state.

Actions referenced in config but not yet implemented (e.g. mouse actions before
Phase 4) are tolerated: the mapper logs a one-time warning and no-ops, so the
full default config can ship before every action exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from actions import media_actions, mouse_actions, system_actions

# Control actions interpreted by the app, not dispatched to actions/.
CONTROL_ACTIONS: Set[str] = {"toggle_system", "switch_mode"}


@dataclass
class ActionResult:
    """Outcome of a dispatch call.

    Attributes:
        fired: True if a mapping matched and something happened.
        action: The resolved action name ("" if no mapping matched).
        message: Human-readable result (for logging / HUD).
        control: A control signal for the app ("toggle_system" /
            "switch_mode"), or None for ordinary actions.
    """

    fired: bool = False
    action: str = ""
    message: str = ""
    control: Optional[str] = None


def _build_registry() -> Dict[str, Callable[..., str]]:
    """Map action names -> callables. Only implemented actions are listed.

    Mouse / presentation actions are added in their respective phases; until
    then they're simply absent and the mapper warns + no-ops if referenced.
    """
    return {
        # system
        "volume_up": system_actions.volume_up,
        "volume_down": system_actions.volume_down,
        "mute": system_actions.mute,
        "brightness_up": system_actions.brightness_up,
        "brightness_down": system_actions.brightness_down,
        "screenshot": system_actions.screenshot,
        "lock_screen": system_actions.lock_screen,
        "launch_app": system_actions.launch_app,
        # media
        "play_pause": media_actions.play_pause,
        "next_track": media_actions.next_track,
        "previous_track": media_actions.previous_track,
        # mouse (mouse_move is handled continuously by the main loop, not here)
        "left_click": mouse_actions.left_click,
        "right_click": mouse_actions.right_click,
        "double_click": mouse_actions.double_click,
        "scroll_up": mouse_actions.scroll_up,
        "scroll_down": mouse_actions.scroll_down,
    }


class ActionMapper:
    """Loads the gesture config and dispatches actions."""

    def __init__(self, config_path: str = "config/gestures.json") -> None:
        """
        Args:
            config_path: Path to the gesture->action JSON config.
        """
        self._config_path = config_path
        self._registry = _build_registry()
        self._warned_missing: Set[str] = set()
        self.reload()

    # -- config -----------------------------------------------------------

    def reload(self) -> None:
        """(Re)load the config from disk. Safe to call after editing the JSON."""
        with open(self._config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        settings = cfg.get("settings", {})
        self.modes: List[str] = settings.get("modes", ["media"])
        self.start_mode: str = settings.get("start_mode", self.modes[0])
        self._mappings: Dict[str, Dict[str, dict]] = cfg.get("mappings", {})

    # -- resolution / dispatch -------------------------------------------

    def resolve(self, mode: str, gesture: str) -> Optional[dict]:
        """Return the mapping entry for (mode, gesture), or None.

        Checks the active mode, then ``global``, then ``any``.
        """
        for scope in (mode, "global", "any"):
            entry = self._mappings.get(scope, {}).get(gesture)
            if entry is not None:
                return entry
        return None

    def dispatch(self, mode: str, gesture: str) -> ActionResult:
        """Resolve and execute the action bound to (mode, gesture).

        Args:
            mode: The currently active mode.
            gesture: The recognized gesture name.

        Returns:
            An :class:`ActionResult` describing what happened.
        """
        if gesture in ("none", ""):
            return ActionResult(fired=False)

        entry = self.resolve(mode, gesture)
        if entry is None:
            return ActionResult(fired=False)

        action = entry.get("action", "")
        params = entry.get("params", {})

        # Control actions are handled by the app (state changes).
        if action in CONTROL_ACTIONS:
            return ActionResult(
                fired=True, action=action, control=action,
                message=action.replace("_", " "),
            )

        func = self._registry.get(action)
        if func is None:
            if action not in self._warned_missing:
                print(f"[WARN] Action '{action}' not implemented yet; ignoring.")
                self._warned_missing.add(action)
            return ActionResult(fired=False, action=action)

        message = func(**params)
        return ActionResult(fired=True, action=action, message=message)

    def register(self, name: str, func: Callable[..., str]) -> None:
        """Register an additional action callable at runtime.

        Used by later phases (mouse, presentation) to plug in their actions
        without editing this module.
        """
        self._registry[name] = func
