from typing import Sequence

import numpy as np

from vehicle_controller.types import ControllerMemory


class LongitudinalLQR:
    def __init__(
        self,
        initial_dt: float = 0.10,
        q: Sequence[float] = (20.0, 10.0),
        r: float = 0.4,
        a_min: float = -7.357,
        a_max: float = 3.372,
        int_error_min: float = -10.0,
        int_error_max: float = 10.0,
    ):
        self.initial_dt = float(initial_dt)
        q_array = np.asarray(q, dtype=float).reshape(-1)
        if q_array.size != 2:
            raise ValueError("Longitudinal LQR requires exactly 2 Q weights: [ev, int_ev].")
        self.q = q_array
        self.r = float(r)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.int_error_min = float(int_error_min)
        self.int_error_max = float(int_error_max)

    def step(self, v_ref: float, vx: float, memory: ControllerMemory, dt: float) -> float:
        dt = max(float(dt), 1e-3)
        ev = float(v_ref) - float(vx)

        x = np.array([[ev], [memory.int_speed_error]], dtype=float)
        a_mat = np.array([[1.0, 0.0], [dt, 1.0]], dtype=float)
        b_mat = np.array([[-dt], [0.0]], dtype=float)
        q_mat = np.diag(self.q).astype(float)
        r_mat = np.array([[self.r]], dtype=float)
        k_gain = self._solve_dlqr_gain(a_mat, b_mat, q_mat, r_mat)

        a_des = float((-k_gain @ x).item())
        a_des = min(max(a_des, self.a_min), self.a_max)
        memory.int_speed_error = min(
            max(memory.int_speed_error, self.int_error_min),
            self.int_error_max,
        )
        memory.last_accel_cmd = a_des
        return a_des

    @staticmethod
    def _solve_dlqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        P = Q.copy()

        for _ in range(200):
            bt_p_b = B.T @ P @ B
            g = R + bt_p_b
            p_next = A.T @ P @ A - (A.T @ P @ B) @ np.linalg.solve(g, B.T @ P @ A) + Q

            if np.linalg.norm(p_next - P, ord="fro") < 1e-9:
                P = p_next
                break

            P = p_next

        return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
