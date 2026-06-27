"""Procedural sound effects for the visual FX (no audio files needed).

All sounds are synthesised with NumPy and mixed live through a single
``sounddevice`` output stream, so multiple sounds can overlap (a looping orb
hum + a chakra whir + a one-shot whoosh). Everything degrades gracefully: if no
audio device is available the class just goes silent instead of crashing.

API:
    sfx.set_orb(level)       # 0..1 looping energy hum
    sfx.set_chakra(level)    # 0..1 looping spinning whir
    sfx.whoosh()             # one-shot throw sound
    sfx.ding()               # one-shot "fully charged" chime
    sfx.close()
"""

from __future__ import annotations

import threading
from typing import Dict, List

import numpy as np

try:
    import sounddevice as sd
    _HAVE_SD = True
except Exception:  # pragma: no cover - environment without PortAudio
    _HAVE_SD = False

SR = 44100  # sample rate


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


class SoundFX:
    """Realtime synth mixer: looping ambiences + overlapping one-shots."""

    def __init__(self) -> None:
        self.ok = False
        self._loops: Dict[str, np.ndarray] = {}
        self._pos: Dict[str, int] = {}
        self._gain: Dict[str, float] = {}
        self._target: Dict[str, float] = {}
        self._oneshots: List[List] = []   # [buffer, position, gain]
        self._lock = threading.Lock()
        if not _HAVE_SD:
            return

        self._loops = {"orb": self._hum(), "chakra": self._whir()}
        self._pos = {k: 0 for k in self._loops}
        self._gain = {k: 0.0 for k in self._loops}
        self._target = {k: 0.0 for k in self._loops}
        try:
            self._stream = sd.OutputStream(
                samplerate=SR, channels=1, blocksize=1024, dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self.ok = True
        except Exception:
            self.ok = False

    # -- public controls -------------------------------------------------

    def set_orb(self, level: float) -> None:
        """Set the looping orb-hum volume (0..1)."""
        if self.ok:
            self._target["orb"] = _clip01(level) * 0.5

    def set_chakra(self, level: float) -> None:
        """Set the looping chakra-whir volume (0..1)."""
        if self.ok:
            self._target["chakra"] = _clip01(level) * 0.6

    def whoosh(self) -> None:
        """Play a one-shot throw whoosh."""
        self._add(self._whoosh(), 0.9)

    def reactor_powerup(self) -> None:
        """Play the one-shot reactor spin-up (on orb activation)."""
        self._add(self._powerup(), 0.95)

    def ding(self) -> None:
        """Play a one-shot 'fully charged' chime."""
        self._add(self._ding(), 0.7)

    def close(self) -> None:
        if self.ok:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self.ok = False

    # -- mixer callback --------------------------------------------------

    def _add(self, buf: np.ndarray, gain: float) -> None:
        if not self.ok:
            return
        with self._lock:
            self._oneshots.append([buf, 0, gain])

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            for name, buf in self._loops.items():
                pos = self._pos[name]
                g0 = self._gain[name]
                g1 = g0 + (self._target[name] - g0) * 0.25  # ease per block
                gains = np.linspace(g0, g1, frames, dtype=np.float32)
                idx = (pos + np.arange(frames)) % len(buf)
                out += buf[idx] * gains
                self._gain[name] = g1
                self._pos[name] = (pos + frames) % len(buf)

            keep = []
            for item in self._oneshots:
                buf, pos, g = item
                n = min(frames, len(buf) - pos)
                out[:n] += buf[pos:pos + n] * g
                item[1] = pos + n
                if item[1] < len(buf):
                    keep.append(item)
            self._oneshots = keep

        np.clip(out, -1.0, 1.0, out=out)
        outdata[:, 0] = out

    # -- synthesis (all return float32 in [-1, 1]) -----------------------

    @staticmethod
    def _norm(a: np.ndarray, peak: float = 0.9) -> np.ndarray:
        """Scale a signal so its loudest sample sits at ``peak``."""
        m = float(np.abs(a).max())
        if m > 1e-6:
            a = a * (peak / m)
        return a.astype(np.float32)

    @staticmethod
    def _hum() -> np.ndarray:
        """Cinematic energy-reactor drone (4 s, seamless loop).

        Layers a deep bass core, harmonic body, a shimmering detuned energy
        layer (slow beating), a slow spectral sweep, and a reactor "throb".
        All frequencies are integers so the loop is seamless over 4 s.
        """
        dur = 4.0
        tau = 2 * np.pi
        t = np.arange(int(SR * dur)) / SR

        # Sub-bass rumble + deep core (fundamental + octave) + harmonic body.
        subbass = 0.6 * np.sin(tau * 30 * t)
        bass = np.sin(tau * 55 * t) + 0.5 * np.sin(tau * 110 * t)
        body = 0.35 * np.sin(tau * 165 * t) + 0.22 * np.sin(tau * 220 * t)

        # Shimmering energy: detuned pairs beat slowly against each other.
        shimmer = (np.sin(tau * 440 * t) + np.sin(tau * 442 * t)
                   + 0.6 * (np.sin(tau * 660 * t) + np.sin(tau * 663 * t)))
        sweep = 0.5 + 0.5 * np.sin(tau * 0.25 * t)       # slow spectral sweep
        shimmer *= 0.08 * (0.4 + 0.6 * sweep)

        # High "energy whine" that pulses in and out -- adds a sci-fi edge.
        whine = 0.035 * np.sin(tau * 1320 * t) * (0.5 + 0.5 * np.sin(tau * 0.5 * t))

        throb = 0.7 + 0.3 * np.sin(tau * 0.5 * t)        # reactor heartbeat
        swell = 0.85 + 0.15 * np.sin(tau * 0.25 * t)     # gentle breathing
        sig = (subbass + bass * 0.6 + body + shimmer + whine) * throb * swell
        return SoundFX._norm(sig, 0.9)

    @staticmethod
    def _powerup() -> np.ndarray:
        """One-shot reactor 'spin-up': rising chirp + energy swell + sub impact."""
        dur = 1.6
        tau = 2 * np.pi
        t = np.arange(int(SR * dur)) / SR
        rise = np.clip(t / (dur * 0.72), 0, 1)

        # Linear pitch sweep 70 -> 660 Hz (the spin-up).
        f0, f1 = 70.0, 660.0
        phase = tau * (f0 * t + (f1 - f0) * t * t / (2 * dur))
        chirp = np.sin(phase) * rise
        # Detuned high layer rising too (energy building).
        chirp += 0.5 * np.sin(phase * 2.01) * (rise ** 2)

        # Filtered noise swell.
        noise = np.random.default_rng(1).standard_normal(len(t)) * (rise ** 2) * 0.4

        # Deep sub-bass impact when it "locks in" near the end.
        it = t - dur * 0.70
        impact = np.where(it >= 0, np.sin(tau * 46 * it) * np.exp(-it * 7.0), 0.0)

        sig = chirp * 0.5 + noise * 0.3 + impact * 0.9
        return SoundFX._norm(sig, 0.95)

    @staticmethod
    def _whir() -> np.ndarray:
        """Metallic spinning whir for the chakra (1 s, seamless)."""
        dur = 1.0
        t = np.arange(int(SR * dur)) / SR
        partials = [311, 437, 661, 884]
        sig = sum(np.sin(2 * np.pi * f * t) / (k + 1)
                  for k, f in enumerate(partials))
        spin = 0.6 + 0.4 * np.sin(2 * np.pi * 18 * t)    # 18 Hz whirr AM
        noise = 0.05 * np.random.default_rng(3).standard_normal(len(t))
        return ((sig * 0.25 + noise) * spin).astype(np.float32)

    @staticmethod
    def _whoosh() -> np.ndarray:
        """One-shot airy whoosh with a downward sweep (0.5 s)."""
        dur = 0.5
        t = np.arange(int(SR * dur)) / SR
        env = np.sin(np.pi * t / dur) ** 2               # smooth swell
        noise = np.random.default_rng(9).standard_normal(len(t))
        sweep = np.sin(2 * np.pi * (700 - 500 * t / dur) * t)  # 700->200 Hz
        return SoundFX._norm((0.7 * noise + 0.3 * sweep) * env, 0.9)

    @staticmethod
    def _ding() -> np.ndarray:
        """Bright bell for 'fully charged' (0.4 s)."""
        dur = 0.4
        t = np.arange(int(SR * dur)) / SR
        env = np.exp(-t * 9)
        sig = (np.sin(2 * np.pi * 880 * t)
               + 0.5 * np.sin(2 * np.pi * 1320 * t))
        return SoundFX._norm(sig * env, 0.85)
