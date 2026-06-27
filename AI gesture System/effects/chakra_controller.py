"""State machine that gives the Sudarshan Chakra charge-and-throw behaviour.

Flow:
    IDLE      -> nothing on screen.
    CHARGING  -> right-hand index pointing: the chakra sits on the fingertip and
                 powers up (grows, spins faster, a charge ring fills).
    FLYING    -> on a fast flick, or on release once charged enough, the chakra
                 detaches and hurtles in the finger's direction, then fades.

It owns timing/physics only; the actual look is drawn by
:class:`effects.sudarshan_chakra.SudarshanChakra`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from effects.sudarshan_chakra import SudarshanChakra

Placement = Tuple[np.ndarray, float, np.ndarray]  # center, radius, direction


class ChakraController:
    IDLE, CHARGING, FLYING = 0, 1, 2

    def __init__(
        self,
        chakra: Optional[SudarshanChakra] = None,
        charge_time: float = 1.1,
        min_charge: float = 0.33,
        launch_speed: float = 2300.0,   # px/sec along the finger direction
        flick_speed: float = 100000.0,  # effectively off: natural movement must
        #                                 not throw the chakra; throw on release
        flight_time: float = 1.0,       # seconds the chakra stays in flight
    ) -> None:
        self.chakra = chakra or SudarshanChakra()
        self.charge_time = charge_time
        self.min_charge = min_charge
        self.launch_speed = launch_speed
        self.flick_speed = flick_speed
        self.flight_time = flight_time

        self.state = self.IDLE
        self.charge = 0.0
        self.vis = 0.0                     # eased display intensity
        self.center = np.zeros(2, np.float32)
        self.radius = 60.0
        self.dir = np.array([0.0, -1.0], np.float32)
        self._prev_center: Optional[np.ndarray] = None
        # flight state
        self.pos = np.zeros(2, np.float32)
        self.vel = np.zeros(2, np.float32)
        self.launch_radius = 60.0
        self.flight_t = 0.0
        self.cooldown = 0.0
        self.just_launched = False   # True only on the frame a throw begins

    # -- update ----------------------------------------------------------

    def update(self, active: bool, placement: Optional[Placement], dt: float) -> None:
        """Advance the state machine by ``dt`` seconds.

        Args:
            active: True if the right-hand "point" is currently held.
            placement: (center, radius, direction) for the fingertip, or None.
            dt: Seconds since the previous frame.
        """
        dt = float(max(1e-3, min(dt, 0.1)))
        self.just_launched = False

        if self.state == self.FLYING:
            self._update_flight(dt)
            return

        if self.cooldown > 0:
            self.cooldown = max(0.0, self.cooldown - dt)

        if active and placement is not None and self.cooldown <= 0:
            center, radius, direction = placement
            center = center.astype(np.float32)
            vel = np.zeros(2, np.float32)
            if self._prev_center is not None:
                vel = (center - self._prev_center) / dt
            self._prev_center = center.copy()

            self.center, self.radius, self.dir = center, float(radius), direction
            if self.state == self.IDLE:
                self.state = self.CHARGING
                self.charge = 0.0
            self.charge = min(1.0, self.charge + dt / self.charge_time)
            self.vis = min(1.0, self.vis + dt * 7)

            if self.charge > 0.25 and float(np.linalg.norm(vel)) > self.flick_speed:
                self._launch(vel)
        else:
            # released / no longer pointing
            if self.state == self.CHARGING and self.charge >= self.min_charge:
                self._launch(np.zeros(2, np.float32))
            else:
                self.state = self.IDLE
                self.charge = 0.0
                self._prev_center = None
                self.vis = max(0.0, self.vis - dt * 5)

    def _launch(self, extra_vel: np.ndarray) -> None:
        self.state = self.FLYING
        self.just_launched = True
        self.vel = self.dir * self.launch_speed + extra_vel * 0.4
        self.pos = self.center.copy()
        self.launch_radius = self.radius * (1.0 + 0.45 * self.charge)
        self.flight_t = 0.0
        self.vis = 1.0
        self._prev_center = None

    def _update_flight(self, dt: float) -> None:
        self.flight_t += dt
        self.pos = self.pos + self.vel * dt
        # fade out over the last 40% of the flight
        tail = self.flight_time * 0.6
        if self.flight_t > tail:
            self.vis = max(0.0, 1.0 - (self.flight_t - tail) / (self.flight_time - tail))
        if self.flight_t >= self.flight_time:
            self.state = self.IDLE
            self.charge = 0.0
            self.vis = 0.0
            self.cooldown = 0.35

    # -- render ----------------------------------------------------------

    def render(self, frame: np.ndarray) -> np.ndarray:
        if self.state == self.FLYING:
            prog = self.flight_t / self.flight_time
            r = self.launch_radius * (1.0 - 0.25 * min(1.0, prog))
            return self.chakra.render(
                frame, (int(self.pos[0]), int(self.pos[1])), r, self.vis
            )
        if self.state == self.CHARGING:
            r = self.radius * (1.0 + 0.45 * self.charge)
            c = (int(self.center[0]), int(self.center[1]))
            frame = self.chakra.render(frame, c, r, self.vis)
            self._charge_arc(frame, c, r, self.charge)
            return frame
        if self.vis > 0.02:
            c = (int(self.center[0]), int(self.center[1]))
            return self.chakra.render(frame, c, self.radius, self.vis)
        return frame

    @staticmethod
    def _charge_arc(frame, center, radius, charge) -> None:
        """A ring that fills up as the chakra charges (white -> red when full)."""
        r = int(radius * 1.32)
        col = (int(255 * (1 - charge)), int(255 * (1 - charge)), 255)  # ->red
        cv2.ellipse(frame, center, (r, r), -90, 0, int(360 * charge),
                    col, 3, cv2.LINE_AA)
        if charge >= 1.0:  # fully charged pip
            cv2.circle(frame, (center[0], center[1] - r), 4, (255, 255, 255),
                       -1, cv2.LINE_AA)
