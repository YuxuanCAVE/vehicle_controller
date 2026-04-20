from __future__ import annotations

from vehicle_controller.types import ControllerMemory


class LongitudinalPID:

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        a_min: float,
        a_max: float,
        int_error_min: float | None = None,
        int_error_max: float | None = None,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.int_error_min = None if int_error_min is None else float(int_error_min)
        self.int_error_max = None if int_error_max is None else float(int_error_max)

    def step(self, v_ref: float, vx: float, memory: ControllerMemory, dt: float) -> float:
        ev = float(v_ref) - float(vx)
        dt_safe = max(float(dt), 1e-6)

        memory.int_speed_error += ev * dt_safe
        if self.int_error_min is not None:
            memory.int_speed_error = max(memory.int_speed_error, self.int_error_min)
        if self.int_error_max is not None:
            memory.int_speed_error = min(memory.int_speed_error, self.int_error_max)

        d_ev = (ev - float(memory.last_speed_error)) / dt_safe
        memory.last_speed_error = ev
        memory.last_update_stamp_sec += dt_safe

        a_des = (
            self.kp * ev
            + self.ki * float(memory.int_speed_error)
            + self.kd * d_ev
        )
        return self._clamp(a_des, self.a_min, self.a_max)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(min(float(value), float(upper)), float(lower))
