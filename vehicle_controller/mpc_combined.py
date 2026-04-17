from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from vehicle_controller.types import ControllerMemory, MeasuredState, ReferencePoint


class MPCCombined:
    def __init__(
        self,
        initial_dt: float = 0.10,
        horizon: int = 25,
        q: Sequence[float] = (15.0, 12.0, 8.0, 10.0, 8.0, 20.0),
        r: Sequence[float] = (5.0, 0.5),
        rd: Sequence[float] = (15.0, 2.0),
        kappa_ff_gain: float = 0.5,
        max_steer: float = 0.3490658504,
        a_min: float = -7.357,
        a_max: float = 3.372,
        wheelbase: float = 2.720,
        mass: float = 1948.0,
        iz: float = 2500.0,
        lf: float = 1.214,
        lr: float = 1.506,
        cf: float = 98800.0,
        cr: float = 116025.0,
    ):
        self.initial_dt = float(initial_dt)
        self.horizon = int(horizon)
        self.q = np.diag(np.asarray(q, dtype=float))
        self.r = np.diag(np.asarray(r, dtype=float))
        self.rd = np.diag(np.asarray(rd, dtype=float))
        self.kappa_ff_gain = float(kappa_ff_gain)
        self.max_steer = float(max_steer)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.wheelbase = float(wheelbase)
        self.mass = float(mass)
        self.iz = float(iz)
        self.lf = float(lf)
        self.lr = float(lr)
        self.cf = float(cf)
        self.cr = float(cr)

    def step(
        self,
        meas: MeasuredState,
        ref_point: ReferencePoint,
        ref_path,
        memory: ControllerMemory,
        dt: float,
    ) -> tuple[float, float]:
        x_pos = float(meas.x)
        y_pos = float(meas.y)
        yaw = float(meas.yaw)
        vx = max(float(meas.vx), 0.5)
        vy = float(meas.vy)
        r_yaw = float(meas.yaw_rate)
        delta_prev = float(meas.delta_prev)
        z_lon = float(memory.int_speed_error)

        idx = int(ref_point.idx)
        x_ref = float(ref_point.xr)
        y_ref = float(ref_point.yr)
        psi_ref = float(ref_point.psi_ref)
        kappa0 = float(ref_point.kappa_ref)

        dx = x_pos - x_ref
        dy = y_pos - y_ref
        e_y = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
        e_psi = _angle_wrap(yaw - psi_ref)
        v_ref_now = float(ref_point.v_ref)
        e_v = v_ref_now - vx

        nx = 6
        nu = 2
        n = self.horizon
        dt = max(float(dt), 1e-3)

        a_c = np.zeros((nx, nx), dtype=float)
        a_c[0, 1] = vx
        a_c[0, 2] = 1.0
        a_c[1, 3] = 1.0
        a_c[2, 2] = -(self.cf + self.cr) / (self.mass * vx)
        a_c[2, 3] = (-self.lf * self.cf + self.lr * self.cr) / (self.mass * vx) - vx
        a_c[3, 2] = (-self.lf * self.cf + self.lr * self.cr) / (self.iz * vx)
        a_c[3, 3] = -(self.lf**2 * self.cf + self.lr**2 * self.cr) / (self.iz * vx)
        a_c[5, 4] = 1.0

        b_c = np.zeros((nx, nu), dtype=float)
        b_c[2, 0] = self.cf / self.mass
        b_c[3, 0] = self.lf * self.cf / self.iz
        b_c[4, 1] = -1.0

        g_c_curve = np.zeros(nx, dtype=float)
        g_c_curve[1] = -vx
        g_c_dvref = np.zeros(nx, dtype=float)
        g_c_dvref[4] = 1.0

        a_d, b_d, g_curve = self._discretize(a_c, b_c, g_c_curve, dt)
        _, _, g_dvref = self._discretize(a_c, b_c, g_c_dvref, dt)

        ds_local = self._estimate_local_spacing(ref_path.x, ref_path.y, idx)
        idx_per_step = max(1, int(round(vx * dt / ds_local)))

        n_ref = len(ref_path.kappa)
        kappa_pred = np.zeros(n, dtype=float)
        v_ref_pred = np.zeros(n, dtype=float)
        dv_ref_pred = np.zeros(n, dtype=float)
        for k in range(n):
            future_idx = min(idx + (k + 1) * idx_per_step, n_ref - 1)
            kappa_pred[k] = float(ref_path.kappa[future_idx])
            v_ref_pred[k] = float(ref_path.v_ref[future_idx])
            if k == 0:
                dv_ref_pred[k] = (v_ref_pred[k] - v_ref_now) / dt
            else:
                dv_ref_pred[k] = (v_ref_pred[k] - v_ref_pred[k - 1]) / dt

        x0 = np.array([e_y, e_psi, vy, r_yaw, e_v, z_lon], dtype=float)

        a_des_prev = float(memory.last_accel_cmd)

        q_bar = np.kron(np.eye(n), self.q)
        r_bar = np.kron(np.eye(n), self.r)
        rd_bar = np.kron(np.eye(n), self.rd)

        sx = np.zeros((nx * n, nx), dtype=float)
        su = np.zeros((nx * n, nu * n), dtype=float)
        sg = np.zeros(nx * n, dtype=float)
        a_pow = np.eye(nx, dtype=float)

        for i in range(n):
            a_pow = a_pow @ a_d
            rows = slice(i * nx, (i + 1) * nx)
            sx[rows, :] = a_pow
            g_sum = np.zeros(nx, dtype=float)
            for j in range(i + 1):
                a_ij = np.linalg.matrix_power(a_d, i - j)
                cols = slice(j * nu, (j + 1) * nu)
                su[rows, cols] = a_ij @ b_d
                g_sum = g_sum + a_ij @ (
                    g_curve * kappa_pred[j] + g_dvref * dv_ref_pred[j]
                )
            sg[rows] = g_sum

        d_single = np.eye(n, dtype=float)
        if n > 1:
            d_single = d_single - np.vstack((np.zeros((1, n)), np.eye(n - 1, n)))
        d_mat = np.kron(d_single, np.eye(nu))

        d0 = np.zeros(nu * n, dtype=float)
        d0[:nu] = np.array([delta_prev, a_des_prev], dtype=float)

        x_free = sx @ x0 + sg
        h = su.T @ q_bar @ su + r_bar + d_mat.T @ rd_bar @ d_mat
        h = 0.5 * (h + h.T)
        f = su.T @ q_bar @ x_free - d_mat.T @ rd_bar @ d0

        delta_ff = self.kappa_ff_gain * np.arctan(self.wheelbase * kappa0)
        lb = np.zeros(nu * n, dtype=float)
        ub = np.zeros(nu * n, dtype=float)
        for k in range(n):
            steer_idx = k * nu
            accel_idx = steer_idx + 1
            lb[steer_idx] = -self.max_steer - delta_ff
            ub[steer_idx] = self.max_steer - delta_ff
            lb[accel_idx] = self.a_min
            ub[accel_idx] = self.a_max

        try:
            u0 = np.clip(np.zeros(nu * n, dtype=float), lb, ub)
            result = minimize(
                fun=lambda u: float(u @ h @ u + 2.0 * f @ u),
                x0=u0,
                jac=lambda u: 2.0 * (h @ u + f),
                bounds=list(zip(lb, ub)),
                method="L-BFGS-B",
            )
            if not result.success:
                delta_corr = 0.0
                a_des = 0.0
            else:
                delta_corr = float(result.x[0])
                a_des = float(result.x[1])
        except Exception:
            delta_corr = -0.8 * e_y - 1.2 * e_psi - 0.05 * vy - 0.08 * r_yaw
            a_des = 2.0 * e_v

        delta = float(delta_corr + delta_ff)
        a_des = float(np.clip(a_des, self.a_min, self.a_max))
        return delta, a_des

    @staticmethod
    def _discretize(
        a_c: np.ndarray,
        b_c: np.ndarray,
        g_c: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx = a_c.shape[0]
        nu = b_c.shape[1]
        aug = np.zeros((nx + nu + 1, nx + nu + 1), dtype=float)
        aug[:nx, :nx] = a_c
        aug[:nx, nx : nx + nu] = b_c
        aug[:nx, nx + nu] = g_c
        e_mat = expm(aug * dt)
        a_d = e_mat[:nx, :nx]
        b_d = e_mat[:nx, nx : nx + nu]
        g_d = e_mat[:nx, nx + nu]
        return a_d, b_d, g_d

    @staticmethod
    def _estimate_local_spacing(x_ref: np.ndarray, y_ref: np.ndarray, idx: int) -> float:
        n_ref = len(x_ref)
        i_lo = max(0, idx - 1)
        i_hi = min(n_ref - 1, idx + 1)
        denom = max(i_hi - i_lo, 1)
        ds_local = np.hypot(x_ref[i_hi] - x_ref[i_lo], y_ref[i_hi] - y_ref[i_lo]) / denom
        return max(float(ds_local), 0.01)


def _angle_wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
