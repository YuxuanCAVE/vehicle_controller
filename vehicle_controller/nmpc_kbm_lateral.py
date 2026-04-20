from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from vehicle_controller.reference_manager import ReferencePath
from vehicle_controller.types import ControllerMemory, MeasuredState, ReferencePoint


class NMPCKBMLateral:
    
    def __init__(
        self,
        initial_dt: float,
        horizon: int,
        q_x: float,
        q_y: float,
        q_psi: float,
        r_delta: float,
        r_du: float,
        max_steer: float,
        wheelbase: float,
        lr: float,
        delta_rate_max: float,
        max_iterations: int = 100,
        max_fun_evals: int = 4000,
    ) -> None:
        self.initial_dt = float(initial_dt)
        self.horizon = int(horizon)
        self.q_x = float(q_x)
        self.q_y = float(q_y)
        self.q_psi = float(q_psi)
        self.r_delta = float(r_delta)
        self.r_du = float(r_du)
        self.max_steer = float(max_steer)
        self.wheelbase = max(float(wheelbase), 1e-6)
        self.lr = max(float(lr), 1e-6)
        self.delta_rate_max = max(float(delta_rate_max), 1e-6)
        self.max_iterations = int(max_iterations)
        self.max_fun_evals = int(max_fun_evals)
        self._last_delta_seq = np.zeros(self.horizon, dtype=float)

    def step(
        self,
        meas: MeasuredState,
        ref_point: ReferencePoint,
        ref_path: ReferencePath,
        memory: ControllerMemory,
        dt: float,
    ) -> float:
        ts = max(float(dt), 1e-3)
        preview = self._build_reference_preview(meas, ref_point, ref_path, ts)
        delta_prev = float(meas.delta_prev)
        u_init = self._build_initial_guess(delta_prev, ts)
        bounds = [(-self.max_steer, self.max_steer)] * self.horizon

        rate_limit = self.delta_rate_max * ts
        constraints = []
        for k in range(self.horizon):
            if k == 0:
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda u, delta_prev=delta_prev, rate_limit=rate_limit: (
                            rate_limit - abs(float(u[0]) - delta_prev)
                        ),
                    }
                )
            else:
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda u, k=k, rate_limit=rate_limit: (
                            rate_limit - abs(float(u[k]) - float(u[k - 1]))
                        ),
                    }
                )

        result = minimize(
            fun=lambda u: self._stage_cost(u, meas, preview, delta_prev, ts),
            x0=u_init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "disp": False,
                "maxiter": self.max_iterations,
                "maxfun": self.max_fun_evals,
            },
        )

        if result.success and result.x.size == self.horizon:
            u_opt = np.asarray(result.x, dtype=float)
        else:
            u_opt = u_init

        self._last_delta_seq = np.asarray(u_opt, dtype=float)
        return self._clamp(float(u_opt[0]), -self.max_steer, self.max_steer)

    def _build_reference_preview(
        self,
        meas: MeasuredState,
        ref_point: ReferencePoint,
        ref_path: ReferencePath,
        ts: float,
    ) -> dict[str, np.ndarray]:
        n_ref = len(ref_path.x)
        idx = min(max(int(ref_point.idx), 0), n_ref - 1)

        i_lo = max(0, idx - 1)
        i_hi = min(n_ref - 1, idx + 1)
        ds_local = math.hypot(
            float(ref_path.x[i_hi] - ref_path.x[i_lo]),
            float(ref_path.y[i_hi] - ref_path.y[i_lo]),
        ) / max(i_hi - i_lo, 1)
        ds_local = max(ds_local, 0.01)

        speed_measured = max(float(meas.vx), 0.0)
        idx_cursor = idx

        xr = np.zeros(self.horizon, dtype=float)
        yr = np.zeros(self.horizon, dtype=float)
        psir = np.zeros(self.horizon, dtype=float)
        v_known = np.full(self.horizon, speed_measured, dtype=float)

        for k in range(self.horizon):
            idx_advance = max(1, int(round(speed_measured * ts / ds_local)))
            idx_cursor = min(idx_cursor + idx_advance, n_ref - 1)
            xr[k] = float(ref_path.x[idx_cursor])
            yr[k] = float(ref_path.y[idx_cursor])
            psir[k] = self._get_ref_heading(ref_path, idx_cursor)

        return {"xr": xr, "yr": yr, "psir": psir, "v": v_known}

    def _build_initial_guess(self, delta_prev: float, ts: float) -> np.ndarray:
        rate_limit = self.delta_rate_max * ts
        if self._last_delta_seq.shape != (self.horizon,):
            guess = np.full(self.horizon, delta_prev, dtype=float)
        else:
            guess = np.concatenate((self._last_delta_seq[1:], self._last_delta_seq[-1:]))

        guess[0] = self._clamp(guess[0], delta_prev - rate_limit, delta_prev + rate_limit)
        for k in range(1, self.horizon):
            guess[k] = self._clamp(guess[k], guess[k - 1] - rate_limit, guess[k - 1] + rate_limit)
        return np.clip(guess, -self.max_steer, self.max_steer)

    def _stage_cost(
        self,
        u: np.ndarray,
        meas: MeasuredState,
        preview: dict[str, np.ndarray],
        delta_prev: float,
        ts: float,
    ) -> float:
        x_pred, ay_pred = self._predict_trajectory(meas, u, preview["v"], ts)

        cost = 0.0
        delta_last = float(delta_prev)
        for k in range(self.horizon):
            e_x = x_pred[0, k + 1] - preview["xr"][k]
            e_y = x_pred[1, k + 1] - preview["yr"][k]
            e_psi = self._angle_wrap(x_pred[2, k + 1] - preview["psir"][k])
            delta_k = float(u[k])
            ddelta_k = delta_k - delta_last

            cost += (
                self.q_x * e_x * e_x
                + self.q_y * e_y * e_y
                + self.q_psi * e_psi * e_psi
                + self.r_delta * delta_k * delta_k
                + self.r_du * ddelta_k * ddelta_k
                + 1e-6 * ay_pred[k] * ay_pred[k]
            )
            delta_last = delta_k
        return float(cost)

    def _predict_trajectory(
        self, meas: MeasuredState, u: np.ndarray, v_known: np.ndarray, ts: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(u)
        x_pred = np.zeros((3, n + 1), dtype=float)
        ay_pred = np.zeros(n, dtype=float)
        x_pred[:, 0] = [float(meas.x), float(meas.y), float(meas.yaw)]

        for k in range(n):
            x_k = x_pred[0, k]
            y_k = x_pred[1, k]
            psi_k = x_pred[2, k]
            v_k = float(v_known[k])
            delta_k = float(u[k])
            beta_k = math.atan((self.lr / self.wheelbase) * math.tan(delta_k))

            x_pred[0, k + 1] = x_k + ts * v_k * math.cos(psi_k + beta_k)
            x_pred[1, k + 1] = y_k + ts * v_k * math.sin(psi_k + beta_k)
            x_pred[2, k + 1] = self._angle_wrap(
                psi_k + ts * (v_k / self.lr) * math.sin(beta_k)
            )
            ay_pred[k] = v_k * v_k / self.wheelbase * math.tan(delta_k)

        return x_pred, ay_pred

    @staticmethod
    def _get_ref_heading(ref_path: ReferencePath, idx: int) -> float:
        if idx <= 0:
            return math.atan2(
                float(ref_path.y[1] - ref_path.y[0]),
                float(ref_path.x[1] - ref_path.x[0]),
            )
        if idx >= len(ref_path.x) - 1:
            return math.atan2(
                float(ref_path.y[-1] - ref_path.y[-2]),
                float(ref_path.x[-1] - ref_path.x[-2]),
            )
        return math.atan2(
            float(ref_path.y[idx + 1] - ref_path.y[idx - 1]),
            float(ref_path.x[idx + 1] - ref_path.x[idx - 1]),
        )

    @staticmethod
    def _angle_wrap(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(min(float(value), float(upper)), float(lower))
