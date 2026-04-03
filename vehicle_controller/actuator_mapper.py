from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from vehicle_controller.types import ActuatorDebug, ControlOutput


class ActuatorMapper:
    def __init__(
        self,
        accel_map_file: str,
        brake_map_file: str,
        max_steer: float,
        mass: float,
        aero_a: float,
        aero_b: float,
        aero_c: float,
        max_pedal_publish: float = 0.60,
    ):
        self.accel_map_file = Path(accel_map_file)
        self.brake_map_file = Path(brake_map_file)
        self.max_steer = float(max_steer)
        self.mass = float(mass)
        self.aero_a = float(aero_a)
        self.aero_b = float(aero_b)
        self.aero_c = float(aero_c)
        self.max_pedal_publish = float(max_pedal_publish)

        self._validate_map_path(self.accel_map_file, "accel_map_file")
        self._validate_map_path(self.brake_map_file, "brake_map_file")
        self.acc_cmd_full, self.acc_force_full = self._load_accel_map(self.accel_map_file)
        self.brake_cmd_full, self.brake_force_full = self._load_brake_map(self.brake_map_file)
        self._validate_accel_map()
        self._validate_brake_map()

    def map_command(
        self, steering_cmd_rad: float, accel_cmd: float, vx: float
    ) -> tuple[ControlOutput, ActuatorDebug]:
        throttle, brake, debug = self._accel_to_normalized_pedals(accel_cmd, vx)
        steering = self._normalize_steering(steering_cmd_rad)
        command = ControlOutput(
            brake=brake,
            throttle=throttle,
            steering=steering,
            accel_cmd=float(accel_cmd),
        )
        return command, debug

    def _accel_to_normalized_pedals(
        self, accel_cmd: float, vx: float
    ) -> tuple[float, float, ActuatorDebug]:
        vx = float(vx)
        accel_cmd = float(accel_cmd)

        f_resist = self.aero_a + self.aero_b * vx + self.aero_c * vx * vx
        f_required = self.mass * accel_cmd + f_resist

        throttle_publish = 0.0
        brake_publish = 0.0
        acc_req = 0.0
        brk_req = 0.0

        if f_required >= 0.0:
            acc_req = self._interp_extrap(self.acc_force_full, self.acc_cmd_full, f_required)
            throttle_publish = (acc_req / max(float(np.max(self.acc_cmd_full)), 1e-6)) * self.max_pedal_publish
            throttle_publish = self._clamp(throttle_publish, 0.0, self.max_pedal_publish)
        else:
            f_brake_required = max(-f_required, 0.0)
            brk_req = self._interp_extrap(self.brake_force_full, self.brake_cmd_full, f_brake_required)
            brake_publish = (brk_req / max(float(np.max(self.brake_cmd_full)), 1e-6)) * self.max_pedal_publish
            brake_publish = self._clamp(brake_publish, 0.0, self.max_pedal_publish)

        # Keep MATLAB-style pedal publish saturation internally, then expose
        # normalized outputs in [0, 1] for the ROS command interface.
        throttle_norm = throttle_publish / max(self.max_pedal_publish, 1e-6)
        brake_norm = brake_publish / max(self.max_pedal_publish, 1e-6)
        debug = ActuatorDebug(
            f_resist=float(f_resist),
            f_required=float(f_required),
            acc_req=float(acc_req),
            brk_req=float(brk_req),
            throttle_publish=float(throttle_publish),
            brake_publish=float(brake_publish),
            throttle_norm=float(throttle_norm),
            brake_norm=float(brake_norm),
        )
        return float(throttle_norm), float(brake_norm), debug

    def _normalize_steering(self, steering_cmd_rad: float) -> float:
        steering_norm = float(steering_cmd_rad) / max(self.max_steer, 1e-6)
        return self._clamp(steering_norm, -1.0, 1.0)

    @staticmethod
    def _validate_map_path(path: Path, name: str) -> None:
        if not str(path):
            raise ValueError(f"{name} is empty.")
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    @staticmethod
    def _load_accel_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
        mat = loadmat(path)
        acc_cmd = _make_col(_get_first_existing_field(mat, ("Acc_Full", "Acc_full")))
        acc_force = _make_col(_get_first_existing_field(mat, ("Force_full", "Force_Full")))

        _, unique_idx = np.unique(acc_cmd, return_index=True)
        unique_idx = np.sort(unique_idx)
        acc_cmd = acc_cmd[unique_idx]
        acc_force = acc_force[unique_idx]

        sort_idx = np.argsort(acc_force)
        return acc_cmd[sort_idx], acc_force[sort_idx]

    @staticmethod
    def _load_brake_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
        mat = loadmat(path)
        brake_cmd_raw = _make_col(
            _get_first_existing_field(mat, ("Break_Full", "Brake_Full", "Brake_full"))
        )
        brake_force_raw = _make_col(_get_first_existing_field(mat, ("Force_full", "Force_Full")))

        brake_cmd_mag = np.abs(brake_cmd_raw)
        brake_force_mag = np.abs(brake_force_raw)

        _, unique_idx = np.unique(brake_cmd_mag, return_index=True)
        unique_idx = np.sort(unique_idx)
        brake_cmd_mag = brake_cmd_mag[unique_idx]
        brake_force_mag = brake_force_mag[unique_idx]

        # Interpolation later uses force as the independent variable, so the
        # lookup table must be monotonic in force, not in brake command.
        sort_idx = np.argsort(brake_force_mag)
        brake_force_sorted = brake_force_mag[sort_idx]
        brake_cmd_sorted = brake_cmd_mag[sort_idx]
        return brake_cmd_sorted, brake_force_sorted

    def _validate_accel_map(self) -> None:
        self._validate_numeric_array(self.acc_cmd_full, "acc_cmd_full")
        self._validate_numeric_array(self.acc_force_full, "acc_force_full")
        if len(self.acc_cmd_full) < 2 or len(self.acc_force_full) < 2:
            raise ValueError("Acceleration map must contain at least 2 valid samples.")
        if not np.all(np.diff(self.acc_force_full) > 0.0):
            raise ValueError("Acceleration force map must be strictly increasing for interpolation.")

    def _validate_brake_map(self) -> None:
        self._validate_numeric_array(self.brake_cmd_full, "brake_cmd_full")
        self._validate_numeric_array(self.brake_force_full, "brake_force_full")
        if len(self.brake_cmd_full) < 2 or len(self.brake_force_full) < 2:
            raise ValueError("Brake map must contain at least 2 valid samples.")
        if not np.all(np.diff(self.brake_force_full) > 0.0):
            raise ValueError("Brake force map must be strictly increasing for interpolation.")

    @staticmethod
    def _validate_numeric_array(values: np.ndarray, name: str) -> None:
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1-D after preprocessing.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or Inf values.")

    @staticmethod
    def _interp_extrap(xp: np.ndarray, fp: np.ndarray, x: float) -> float:
        x = float(x)
        if x <= xp[0]:
            return float(fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-9))
        if x >= xp[-1]:
            return float(fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-9))
        return float(np.interp(x, xp, fp))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(min(float(value), float(upper)), float(lower))


def _get_first_existing_field(mat_data: dict, field_names: Iterable[str]):
    for field_name in field_names:
        if field_name in mat_data:
            return mat_data[field_name]
    raise KeyError(f"Required field not found. Tried fields: {tuple(field_names)}")


def _make_col(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)
