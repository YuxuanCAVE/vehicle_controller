from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from vehicle_controller.types import ControllerMemory, MeasuredState, ReferencePoint


class KinematicLateralMPC:
    def __init__(
        self,
        initial_dt: float = 0.10,
        horizon: int = 25,
        q: Sequence[float] = (15.0, 12.0),
        r: float = 5.0,
        rd: float = 15.0,
        kappa_ff_gain: float = 1.0,
        max_steer: float = 0.6108652382,
        wheelbase: float = 2.720,
    ):
        self.initial_dt = float(initial_dt)
        self.horizon = int(horizon)
        self.q = np.diag(np.asarray(q, dtype=float))
        self.r = float(r)
        self.rd = float(rd)
        self.kappa_ff_gain = float(kappa_ff_gain)
        self.max_steer = float(max_steer)
        self.wheelbase = float(wheelbase)

    def step(
        self,
        meas: MeasuredState,
        ref_point: ReferencePoint,
        ref_path,
        memory: ControllerMemory,
        dt: float,
    ) -> float:
        del ref_path
        del memory

        vx = max(float(meas.vx), 0.5)
        e_y = float(ref_point.e_y)
        e_psi = float(ref_point.e_psi)
        kappa0 = float(ref_point.kappa_ref)
        delta_prev = float(meas.delta_prev)
        dt = max(float(dt), 1e-3)

        delta_ff = self.kappa_ff_gain * np.arctan(self.wheelbase * kappa0)
        cos_ff = np.cos(delta_ff)
        cos_ff_sq = max(float(cos_ff * cos_ff), 0.25)

        a_d = np.array(
            [
                [1.0, vx * dt],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        b_d = np.array(
            [
                0.0,
                vx * dt / (self.wheelbase * cos_ff_sq),
            ],
            dtype=float,
        )

        nx = 2
        n = self.horizon
        x0 = np.array([e_y, e_psi], dtype=float)

        q_bar = np.kron(np.eye(n), self.q)
        r_bar = np.eye(n, dtype=float) * self.r

        sx = np.zeros((nx * n, nx), dtype=float)
        su = np.zeros((nx * n, n), dtype=float)

        a_power = np.eye(nx, dtype=float)
        for i in range(n):
            a_power = a_power @ a_d
            rows = slice(i * nx, (i + 1) * nx)
            sx[rows, :] = a_power
            for j in range(i + 1):
                su[rows, j] = np.linalg.matrix_power(a_d, i - j) @ b_d

        d_mat = np.eye(n, dtype=float)
        if n > 1:
            d_mat = d_mat - np.vstack((np.zeros((1, n)), np.eye(n - 1, n)))
        d0 = np.zeros(n, dtype=float)
        d0[0] = delta_prev - delta_ff

        x_free = sx @ x0
        h = su.T @ q_bar @ su + r_bar + self.rd * (d_mat.T @ d_mat)
        h = 0.5 * (h + h.T)
        f = su.T @ q_bar @ x_free - self.rd * (d_mat.T @ d0)

        lb = -self.max_steer * np.ones(n, dtype=float) - delta_ff
        ub = self.max_steer * np.ones(n, dtype=float) - delta_ff

        try:
            u0 = np.clip(np.full(n, d0[0], dtype=float), lb, ub)
            result = minimize(
                fun=lambda u: float(u @ h @ u + 2.0 * f @ u),
                x0=u0,
                jac=lambda u: 2.0 * (h @ u + f),
                bounds=list(zip(lb, ub)),
                method="L-BFGS-B",
            )
            delta_corr = 0.0 if not result.success else float(result.x[0])
        except Exception:
            delta_corr = -0.8 * e_y - 1.5 * e_psi

        return float(delta_corr + delta_ff)
