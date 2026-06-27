"""Spoken voice announcements via Windows SAPI (text-to-speech).

A background worker thread owns the COM SAPI voice and speaks queued phrases, so
calling :meth:`VoiceFX.say` never blocks the render loop. Degrades to silence if
SAPI / pywin32 isn't available.

Used for dramatic call-outs like "Sudarshan Chakra activated" and
"Reactor activated".
"""

from __future__ import annotations

import queue
import threading
from typing import Optional


class VoiceFX:
    """Non-blocking text-to-speech announcer."""

    def __init__(self, rate: int = 1, volume: int = 100) -> None:
        """
        Args:
            rate: SAPI speaking rate (-10..10; slightly fast = 1).
            volume: SAPI volume (0..100).
        """
        self.ok = False
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        try:
            import pythoncom  # noqa: F401  (ensure pywin32 present)
            import win32com.client  # noqa: F401
        except Exception:
            return
        self._rate = rate
        self._volume = volume
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.ok = True

    def _run(self) -> None:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            try:
                voice.Rate = self._rate
                voice.Volume = self._volume
            except Exception:
                pass
            while True:
                text = self._q.get()
                if text is None:
                    break
                try:
                    voice.Speak(text, 0)  # synchronous within this thread
                except Exception:
                    pass
        finally:
            pythoncom.CoUninitialize()

    def say(self, text: str) -> None:
        """Queue a phrase to be spoken (returns immediately)."""
        if self.ok:
            self._q.put(text)

    def close(self) -> None:
        if self.ok:
            self._q.put(None)
            self.ok = False
