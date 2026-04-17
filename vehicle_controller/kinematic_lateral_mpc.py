from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from vehicle_controller.types import ControllerMemory, MeasuredState, ReferencePoint


class KinematicLateralMPC:
    def __init__(
        self,
        initial_dt: float = 0.10,
        horizon: int = 16,
        q: Sequence[float] = (2.2, 0.8),
        r: float = 25.0,
        rd: float = 15.0,
        kappa_ff_gain: float = 0.1,
        max_steer: float = 0.3490658504,
        wheelbase: float = 2.720,
        fallback_k_e_y: float = 0.9,
        fallback_k_e_psi: float = 1.4,
    ):
        self.initial_dt = float(initial_dt)
        self.horizon = int(horizon)
        self.q = np.diag(np.asarray(q, dtype=float))
        self.r = float(r)
        self.rd = float(rd)
        self.kappa_ff_gain = float(kappa_ff_gain)
        self.max_steer = float(max_steer)
        self.wheelbase = float(wheelbase)
        self.fallback_k_e_y = float(fallback_k_e_y)
        self.fallback_k_e_psi = float(fallback_k_e_psi)

    def step(
        self,
        meas: MeasuredState,
        ref_point: ReferencePoint,
        ref_path,
        memory: ControllerMemory,
        dt: float,
    ) -> float:
        vx = max(float(meas.vx), 0.5)
        e_y = float(ref_point.e_y)
        e_psi = float(ref_point.e_psi)
        idx = int(ref_point.idx)
        kappa0 = float(ref_point.kappa_ref)
        dt = max(float(dt), 1e-3)

        nx = 2
        n = self.horizon
        x0 = np.array([e_y, e_psi], dtype=float)

        a_c = np.array([[0.0, vx], [0.0, 0.0]], dtype=float)
        b_c = np.array([[0.0], [vx / self.wheelbase]], dtype=float)
        g_c_unit = np.array([[0.0], [-vx]], dtype=float)

        m_aug = np.zeros((nx + 2, nx + 2), dtype=float)
        m_aug[:nx, :nx] = a_c
        m_aug[:nx, nx : nx + 1] = b_c
        m_aug[:nx, nx + 1 : nx + 2] = g_c_unit
        e_mat = expm(m_aug * dt)
        a_d = e_mat[:nx, :nx]
        b_d = e_mat[:nx, nx : nx + 1].reshape(nx)
        g_d_unit = e_mat[:nx, nx + 1 : nx + 2].reshape(nx)

        ds_local = self._estimate_local_spacing(ref_path.x, ref_path.y, idx)
        idx_per_step = max(1, int(round(vx * dt / ds_local)))

        n_ref = len(ref_path.kappa)
        kappa_pred = np.zeros(n, dtype=float)
        delta_ff_pred = np.zeros(n, dtype=float)
        for k in range(n):
            future_idx = min(idx + (k + 1) * idx_per_step, n_ref - 1)
            kappa_pred[k] = float(ref_path.kappa[future_idx])
            delta_ff_pred[k] = self.kappa_ff_gain * np.arctan(
                self.wheelbase * kappa_pred[k]
            )

        delta_ff0 = self.kappa_ff_gain * np.arctan(self.wheelbase * kappa0)
        delta_prev_corr = float(meas.delta_prev) - delta_ff0

        q_bar = np.kron(np.eye(n), self.q)
        r_bar = np.eye(n, dtype=float) * self.r

        sx = np.zeros((nx * n, nx), dtype=float)
        su = np.zeros((nx * n, n), dtype=float)
        sd = np.zeros(nx * n, dtype=float)
        a_power = np.eye(nx, dtype=float)

        for i in range(n):
            a_power = a_power @ a_d
            rows = slice(i * nx, (i + 1) * nx)
            sx[rows, :] = a_power
            d_sum = np.zeros(nx, dtype=float)
            for j in range(i + 1):
                a_ij = np.linalg.matrix_power(a_d, i - j)
                su[rows, j] = a_ij @ b_d
                d_j = b_d * delta_ff_pred[j] + g_d_unit * kappa_pred[j]
                d_sum = d_sum + a_ij @ d_j
            sd[rows] = d_sum

        x_free = sx @ x0 + sd

        d_mat = np.eye(n, dtype=float)
        if n > 1:
            d_mat = d_mat - np.vstack((np.zeros((1, n)), np.eye(n - 1, n)))
        d0 = np.zeros(n, dtype=float)
        d0[0] = delta_prev_corr

        h = su.T @ q_bar @ su + r_bar + self.rd * (d_mat.T @ d_mat)
        h = 0.5 * (h + h.T)
        f = su.T @ q_bar @ x_free - self.rd * (d_mat.T @ d0)

        lb = -self.max_steer * np.ones(n, dtype=float) - delta_ff_pred
        ub = self.max_steer * np.ones(n, dtype=float) - delta_ff_pred

        try:
            u0 = np.clip(np.full(n, delta_prev_corr, dtype=float), lb, ub)
            result = minimize(
                fun=lambda u: float(u @ h @ u + 2.0 * f @ u),
                x0=u0,
                jac=lambda u: 2.0 * (h @ u + f),
                bounds=list(zip(lb, ub)),
                method="L-BFGS-B",
            )
            delta_corr = 0.0 if not result.success else float(result.x[0])
        except Exception:
            delta_corr = -self.fallback_k_e_y * e_y - self.fallback_k_e_psi * e_psi

        return float(delta_corr + delta_ff0)

    @staticmethod
    def _estimate_local_spacing(x_ref: np.ndarray, y_ref: np.ndarray, idx: int) -> float:
        n_ref = len(x_ref)
        i_lo = max(0, idx - 1)
        i_hi = min(n_ref - 1, idx + 1)
        denom = max(i_hi - i_lo, 1)
        ds_local = np.hypot(x_ref[i_hi] - x_ref[i_lo], y_ref[i_hi] - y_ref[i_lo]) / denom
        return max(float(ds_local), 0.01)
