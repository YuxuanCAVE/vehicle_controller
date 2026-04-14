from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

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
        (
            self.acc_cmd_full,
            self.acc_force_full,
            self.acc_vel_full,
        ) = self._load_accel_map(self.accel_map_file)
        (
            self.brake_cmd_full,
            self.brake_force_full,
            self.brake_vel_full,
        ) = self._load_brake_map(self.brake_map_file)
        self._validate_accel_map()
        self._validate_brake_map()
        self._build_interpolators()

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
            acc_req = self._lookup_accel_command(vx=vx, f_required=f_required)
            throttle_publish = (acc_req / max(float(np.max(self.acc_cmd_full)), 1e-6)) * self.max_pedal_publish
            throttle_publish = self._clamp(throttle_publish, 0.0, self.max_pedal_publish)
        else:
            brk_req = self._lookup_brake_command(vx=vx, f_required=f_required)
            brake_publish = (abs(brk_req) / max(float(np.max(np.abs(self.brake_cmd_full))), 1e-6)) * self.max_pedal_publish
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
    def _load_accel_map(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = loadmat(path)
        acc_cmd = _make_col(_get_first_existing_field(mat, ("Acc_Full", "Acc_full")))
        acc_force = _make_col(_get_first_existing_field(mat, ("Force_full", "Force_Full")))
        acc_vel = _make_col(_get_first_existing_field(mat, ("Vel_Full", "Vel_full")))

        points = np.column_stack((acc_vel, acc_force))
        _, unique_idx = np.unique(points, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return acc_cmd[unique_idx], acc_force[unique_idx], acc_vel[unique_idx]

    @staticmethod
    def _load_brake_map(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = loadmat(path)
        brake_cmd = _make_col(
            _get_first_existing_field(mat, ("Break_Full", "Brake_Full", "Brake_full"))
        )
        brake_force = _make_col(_get_first_existing_field(mat, ("Force_full", "Force_Full")))
        brake_vel = _make_col(_get_first_existing_field(mat, ("Vel_Full", "Vel_full")))

        points = np.column_stack((brake_vel, brake_force))
        _, unique_idx = np.unique(points, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return brake_cmd[unique_idx], brake_force[unique_idx], brake_vel[unique_idx]

    def _validate_accel_map(self) -> None:
        self._validate_numeric_array(self.acc_cmd_full, "acc_cmd_full")
        self._validate_numeric_array(self.acc_force_full, "acc_force_full")
        self._validate_numeric_array(self.acc_vel_full, "acc_vel_full")
        if len(self.acc_cmd_full) < 3 or len(self.acc_force_full) < 3 or len(self.acc_vel_full) < 3:
            raise ValueError("Acceleration map must contain at least 3 valid samples.")

    def _validate_brake_map(self) -> None:
        self._validate_numeric_array(self.brake_cmd_full, "brake_cmd_full")
        self._validate_numeric_array(self.brake_force_full, "brake_force_full")
        self._validate_numeric_array(self.brake_vel_full, "brake_vel_full")
        if (
            len(self.brake_cmd_full) < 3
            or len(self.brake_force_full) < 3
            or len(self.brake_vel_full) < 3
        ):
            raise ValueError("Brake map must contain at least 3 valid samples.")

    @staticmethod
    def _validate_numeric_array(values: np.ndarray, name: str) -> None:
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1-D after preprocessing.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or Inf values.")

    def _build_interpolators(self) -> None:
        acc_points = np.column_stack((self.acc_vel_full, self.acc_force_full))
        self._acc_linear = LinearNDInterpolator(acc_points, self.acc_cmd_full)
        self._acc_nearest = NearestNDInterpolator(acc_points, self.acc_cmd_full)

        brake_points = np.column_stack((self.brake_vel_full, self.brake_force_full))
        self._brake_linear = LinearNDInterpolator(brake_points, self.brake_cmd_full)
        self._brake_nearest = NearestNDInterpolator(brake_points, self.brake_cmd_full)

    def _lookup_accel_command(self, vx: float, f_required: float) -> float:
        return self._lookup_scattered(
            linear_interp=self._acc_linear,
            nearest_interp=self._acc_nearest,
            vx=vx,
            force=f_required,
        )

    def _lookup_brake_command(self, vx: float, f_required: float) -> float:
        return self._lookup_scattered(
            linear_interp=self._brake_linear,
            nearest_interp=self._brake_nearest,
            vx=vx,
            force=f_required,
        )

    @staticmethod
    def _lookup_scattered(
        linear_interp,
        nearest_interp,
        vx: float,
        force: float,
    ) -> float:
        value = linear_interp(float(vx), float(force))
        if value is None:
            value = np.nan
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size == 0 or not np.isfinite(value[0]):
            value = nearest_interp(float(vx), float(force))
            value = np.asarray(value, dtype=float).reshape(-1)
        if value.size == 0 or not np.isfinite(value[0]):
            raise ValueError(
                f"Could not interpolate actuator map at vx={float(vx):.3f}, force={float(force):.3f}"
            )
        return float(value[0])

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
