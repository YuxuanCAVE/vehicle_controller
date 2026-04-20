from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat

from vehicle_controller.types import MeasuredState, ReferencePoint


@dataclass
class ReferencePath:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    kappa: np.ndarray
    v_ref: np.ndarray


class ReferenceManager:
    def __init__(
        self,
        path_file: str,
        speed_profile: str = "",
        speed_mode: str = "constant",
        constant_speed: float = 5.0,
        search_window: int = 40,
    ):
        self.path_file = Path(path_file)
        self.speed_profile = str(speed_profile).strip()
        self.speed_profile_file = self._resolve_speed_profile_file()
        self.speed_mode = speed_mode
        self.constant_speed = float(constant_speed)
        self.search_window = int(search_window)
        self.ref = self._load_reference()

    def query(self, meas: MeasuredState, idx_hint: Optional[int] = None) -> ReferencePoint:
        idx, xr, yr, psi_ref, e_y, seg_idx, seg_t = self._nearest_path_ref_point(
            x=meas.x,
            y=meas.y,
            x_ref=self.ref.x,
            y_ref=self.ref.y,
            idx_hint=idx_hint,
            window=self.search_window,
        )

        kappa_ref = self._interpolate_projected_curvature(self.ref.kappa, idx, seg_idx, seg_t)
        e_psi = self._angle_wrap(meas.yaw - psi_ref)
        v_ref = float(self.ref.v_ref[min(max(idx, 0), len(self.ref.v_ref) - 1)])

        return ReferencePoint(
            idx=int(idx),
            xr=float(xr),
            yr=float(yr),
            psi_ref=float(psi_ref),
            kappa_ref=float(kappa_ref),
            v_ref=v_ref,
            e_y=float(e_y),
            e_psi=float(e_psi),
        )

    def _load_reference(self) -> ReferencePath:
        mat = loadmat(self.path_file)
        if "x_opt" not in mat or "y_opt" not in mat:
            raise KeyError(f"Reference file must contain x_opt and y_opt: {self.path_file}")

        x = np.asarray(mat["x_opt"]).reshape(-1).astype(float)
        y = np.asarray(mat["y_opt"]).reshape(-1).astype(float)

        if len(x) < 2:
            raise ValueError("Reference path must contain at least 2 points.")

        if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 0.50:
            x = x[:-1]
            y = y[:-1]

        yaw = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        den = np.power(dx * dx + dy * dy, 1.5)
        den[den < 1e-9] = 1e-9
        kappa = (dx * ddy - dy * ddx) / den
        kappa[~np.isfinite(kappa)] = 0.0

        v_ref = self._load_speed_profile(len(x))
        return ReferencePath(x=x, y=y, yaw=yaw, kappa=kappa, v_ref=v_ref)

    def _load_speed_profile(self, n_points: int) -> np.ndarray:
        if self.speed_mode == "constant":
            return np.full(n_points, self.constant_speed, dtype=float)

        if self.speed_profile_file is None:
            raise ValueError(
                "speed.mode is set to 'profile' but no speed profile was selected. "
                "Set speed.profile."
            )

        mat = loadmat(self.speed_profile_file)
        if "pathv_ref" in mat:
            v_profile = np.asarray(mat["pathv_ref"]).reshape(-1).astype(float)
        else:
            raise KeyError(
                f"Speed profile file must contain pathv_ref: {self.speed_profile_file}"
            )

        # Keep MATLAB samples 26..384 (1-based), which maps to Python [25:384).
        v_profile = v_profile[25:384]

        if len(v_profile) != n_points:
            raise ValueError(
                f"Speed profile length mismatch: expected {n_points}, got {len(v_profile)}"
            )

        return v_profile

    def _resolve_speed_profile_file(self) -> Optional[Path]:
        if not self.speed_profile:
            return None

        package_root = self.path_file.parent
        profile_dir = package_root / "reference_velocity"
        if not profile_dir.exists():
            raise FileNotFoundError(f"Speed profile directory not found: {profile_dir}")

        profile_stem = self.speed_profile
        if profile_stem.isdigit():
            profile_stem = f"referencePath_Velocity_peak_velocity_{profile_stem}"

        profile_file = profile_dir / profile_stem
        if profile_file.suffix != ".mat":
            profile_file = profile_file.with_suffix(".mat")

        if not profile_file.exists():
            raise FileNotFoundError(f"Speed profile file not found: {profile_file}")

        return profile_file

    def _nearest_path_ref_point(
        self,
        x: float,
        y: float,
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        idx_hint: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Tuple[int, float, float, float, float, int, float]:
        n = len(x_ref)
        if n < 2:
            raise ValueError("Path must contain at least 2 waypoints.")

        if idx_hint is None:
            d2 = (x_ref - x) ** 2 + (y_ref - y) ** 2
            idx = int(np.argmin(d2))
        else:
            local_window = self.search_window if window is None else int(window)
            i0 = max(0, int(idx_hint) - local_window)
            i1 = min(n - 1, int(idx_hint) + local_window)
            d2 = (x_ref[i0 : i1 + 1] - x) ** 2 + (y_ref[i0 : i1 + 1] - y) ** 2
            idx = i0 + int(np.argmin(d2))

        xr = float(x_ref[idx])
        yr = float(y_ref[idx])

        if idx == 0:
            psi_r = np.arctan2(y_ref[1] - y_ref[0], x_ref[1] - x_ref[0])
        elif idx == n - 1:
            psi_r = np.arctan2(y_ref[n - 1] - y_ref[n - 2], x_ref[n - 1] - x_ref[n - 2])
        else:
            psi_r = np.arctan2(y_ref[idx + 1] - y_ref[idx - 1], x_ref[idx + 1] - x_ref[idx - 1])

        dx = float(x) - xr
        dy = float(y) - yr
        ey = -np.sin(psi_r) * dx + np.cos(psi_r) * dy

        seg_idx = idx
        seg_t = 0.0
        return idx, xr, yr, float(psi_r), float(ey), int(seg_idx), float(seg_t)

    @staticmethod
    def _interpolate_projected_curvature(
        kappa_ref: np.ndarray, idx: int, seg_idx: Optional[int], seg_t: Optional[float]
    ) -> float:
        n = len(kappa_ref)
        if seg_idx is not None and 0 <= seg_idx < n - 1 and seg_t is not None:
            return float((1.0 - seg_t) * kappa_ref[seg_idx] + seg_t * kappa_ref[seg_idx + 1])
        return float(kappa_ref[idx])

    @staticmethod
    def _angle_wrap(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
