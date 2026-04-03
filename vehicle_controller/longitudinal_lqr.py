import numpy as np

from vehicle_controller.types import ControllerMemory


class LongitudinalLQR:
    def __init__(
        self,
        dt: float,
        q_speed_error: float = 8.0,
        q_int_error: float = 1.5,
        r_accel: float = 0.8,
        a_min: float = -3.0,
        a_max: float = 2.0,
    ):
        self.dt = float(dt)
        self.q_speed_error = float(q_speed_error)
        self.q_int_error = float(q_int_error)
        self.r_accel = float(r_accel)
        self.a_min = float(a_min)
        self.a_max = float(a_max)

        self.A = np.array(
            [
                [1.0, 0.0],
                [self.dt, 1.0],
            ],
            dtype=float,
        )
        self.B = np.array(
            [
                [-self.dt],
                [0.0],
            ],
            dtype=float,
        )
        self.Q = np.diag([self.q_speed_error, self.q_int_error]).astype(float)
        self.R = np.array([[self.r_accel]], dtype=float)
        self.K = self._solve_dlqr_gain(self.A, self.B, self.Q, self.R)

    def step(self, v_ref: float, vx: float, memory: ControllerMemory) -> float:
        ev = float(v_ref) - float(vx)
        memory.int_speed_error += ev * self.dt

        x = np.array(
            [
                [ev],
                [memory.int_speed_error],
            ],
            dtype=float,
        )

        a_des = float((-self.K @ x).item())
        a_des = min(max(a_des, self.a_min), self.a_max)
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
