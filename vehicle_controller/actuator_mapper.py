from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from vehicle_controller.types import ActuatorDebug, ControlOutput


@dataclass
class _LookupMap1D:
    cmd_full: np.ndarray
    force_full: np.ndarray
    cmd_min: float
    cmd_max: float
    cmd_min_effective: float
    force_min_effective: float
    force_exit_coast: float


class ActuatorMapper:
    def __init__(
        self,
        accel_map_file: str,
        brake_map_file: str,
        max_steer: float,
        steering_sign: float,
        mass: float,
        aero_a: float,
        aero_b: float,
        aero_c: float,
        max_pedal_publish: float = 0.60,
    ):
        self.accel_map_file = Path(accel_map_file)
        self.brake_map_file = Path(brake_map_file)
        self.max_steer = float(max_steer)
        self.steering_sign = 1.0 if float(steering_sign) >= 0.0 else -1.0
        self.mass = float(mass)
        self.aero_a = float(aero_a)
        self.aero_b = float(aero_b)
        self.aero_c = float(aero_c)
        self.max_pedal_publish = float(max_pedal_publish)
        self.actuator_mode = "coast"

        self._validate_map_path(self.accel_map_file, "accel_map_file")
        self._validate_map_path(self.brake_map_file, "brake_map_file")
        self.acc_map = self._load_accel_map(self.accel_map_file)
        self.brake_map = self._load_brake_map(self.brake_map_file)

    def map_command(
        self, steering_cmd_rad: float, accel_cmd: float, vx: float
    ) -> tuple[ControlOutput, ActuatorDebug]:
        throttle, brake, debug = self._accel_to_normalized_pedals(accel_cmd, vx)
        steering = self._publish_steering_rad(steering_cmd_rad)
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
        self._update_actuator_mode(f_required)

        throttle_publish = 0.0
        brake_publish = 0.0
        acc_req = 0.0
        brk_req = 0.0
        f_drive_actual = 0.0
        branch_mode = 0.0

        if self.actuator_mode == "drive":
            f_tractive_required = max(f_required, self.acc_map.force_min_effective)
            acc_req = self._invert_force_map_1d(f_tractive_required, self.acc_map)
            throttle_publish = (
                acc_req / max(self.acc_map.cmd_max, 1e-6) * self.max_pedal_publish
            )
            throttle_publish = self._clamp(throttle_publish, 0.0, self.max_pedal_publish)

            acc_internal = (
                throttle_publish / max(self.max_pedal_publish, 1e-6) * self.acc_map.cmd_max
            )
            f_drive_actual = self._eval_force_map_1d(acc_internal, self.acc_map)
            branch_mode = 1.0
        elif self.actuator_mode == "brake":
            f_brake_required = max(-f_required, self.brake_map.force_min_effective)
            brk_req = self._invert_force_map_1d(f_brake_required, self.brake_map)
            brake_publish = (
                brk_req / max(self.brake_map.cmd_max, 1e-6) * self.max_pedal_publish
            )
            brake_publish = self._clamp(brake_publish, 0.0, self.max_pedal_publish)

            brk_internal = (
                brake_publish / max(self.max_pedal_publish, 1e-6) * self.brake_map.cmd_max
            )
            f_drive_actual = -self._eval_force_map_1d(brk_internal, self.brake_map)
            branch_mode = -1.0

        # Publish the lookup-map result directly.
        # The DBW command uses the actual pedal request, capped by max_pedal_publish,
        # rather than a second normalization back to [0, 1].
        throttle_cmd = throttle_publish
        brake_cmd = brake_publish
        debug = ActuatorDebug(
            f_resist=float(f_resist),
            f_required=float(f_required),
            acc_req=float(acc_req),
            brk_req=float(brk_req),
            throttle_publish=float(throttle_publish),
            brake_publish=float(brake_publish),
            throttle_norm=float(throttle_cmd),
            brake_norm=float(brake_cmd),
            f_drive_actual=float(f_drive_actual),
            branch_mode=float(branch_mode),
        )
        return float(throttle_cmd), float(brake_cmd), debug

    def _publish_steering_rad(self, steering_cmd_rad: float) -> float:
        steering_rad = self.steering_sign * float(steering_cmd_rad)
        return self._clamp(steering_rad, -self.max_steer, self.max_steer)

    def _update_actuator_mode(self, f_required: float) -> None:
        drive_enter = self.acc_map.force_min_effective
        drive_exit = self.acc_map.force_exit_coast
        brake_enter = self.brake_map.force_min_effective
        brake_exit = self.brake_map.force_exit_coast

        if self.actuator_mode == "drive":
            if f_required <= -brake_enter:
                self.actuator_mode = "brake"
            elif f_required <= drive_exit:
                self.actuator_mode = "coast"
            return

        if self.actuator_mode == "brake":
            if f_required >= drive_enter:
                self.actuator_mode = "drive"
            elif f_required >= -brake_exit:
                self.actuator_mode = "coast"
            return

        if f_required >= drive_enter:
            self.actuator_mode = "drive"
        elif f_required <= -brake_enter:
            self.actuator_mode = "brake"
        else:
            self.actuator_mode = "coast"

    def _eval_force_map_1d(self, cmd: float, lookup_map: _LookupMap1D) -> float:
        if float(cmd) < lookup_map.cmd_min_effective:
            return 0.0

        cmd_q = self._clamp(cmd, lookup_map.cmd_min_effective, lookup_map.cmd_max)
        return self._interp_extrap(lookup_map.cmd_full, lookup_map.force_full, cmd_q)

    def _invert_force_map_1d(self, force_target: float, lookup_map: _LookupMap1D) -> float:
        if float(force_target) < lookup_map.force_min_effective:
            return 0.0

        cmd = self._interp_extrap(lookup_map.force_full, lookup_map.cmd_full, force_target)
        return self._clamp(cmd, lookup_map.cmd_min_effective, lookup_map.cmd_max)

    @staticmethod
    def _validate_map_path(path: Path, name: str) -> None:
        if not str(path):
            raise ValueError(f"{name} is empty.")
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    def _load_accel_map(self, path: Path) -> _LookupMap1D:
        mat = loadmat(path)
        acc_cmd_raw = _make_col(_get_first_existing_field(mat, ("Acc_Full", "Acc_full")))
        acc_force_raw = _make_col(_get_first_existing_field(mat, ("Force_full", "Force_Full")))

        self._validate_numeric_array(acc_cmd_raw, "acc_cmd_raw")
        self._validate_numeric_array(acc_force_raw, "acc_force_raw")

        _, unique_idx = np.unique(acc_cmd_raw, return_index=True)
        unique_idx = np.sort(unique_idx)
        acc_cmd = acc_cmd_raw[unique_idx]
        acc_force = acc_force_raw[unique_idx]
        order = np.argsort(acc_force)
        acc_cmd = acc_cmd[order]
        acc_force = acc_force[order]
        acc_force, force_unique_idx = np.unique(acc_force, return_index=True)
        acc_cmd = acc_cmd[force_unique_idx]
        acc_cmd, acc_force = self._add_zero_anchor(acc_cmd, acc_force)

        cmd_min = 0.0
        cmd_max = float(np.max(acc_cmd))
        cmd_min_effective = 0.0
        force_min_effective = 0.0

        return _LookupMap1D(
            cmd_full=acc_cmd,
            force_full=acc_force,
            cmd_min=cmd_min,
            cmd_max=cmd_max,
            cmd_min_effective=cmd_min_effective,
            force_min_effective=float(force_min_effective),
            force_exit_coast=0.0,
        )

    def _load_brake_map(self, path: Path) -> _LookupMap1D:
        mat = loadmat(path)
        brk_cmd_raw = np.abs(
            _make_col(
                _get_first_existing_field(mat, ("Break_Full", "Brake_Full", "Brake_full"))
            )
        )
        brk_force_raw = np.abs(
            _make_col(_get_first_existing_field(mat, ("Force_full", "Force_Full")))
        )

        self._validate_numeric_array(brk_cmd_raw, "brk_cmd_raw")
        self._validate_numeric_array(brk_force_raw, "brk_force_raw")

        _, unique_idx = np.unique(brk_cmd_raw, return_index=True)
        unique_idx = np.sort(unique_idx)
        brk_cmd = brk_cmd_raw[unique_idx]
        brk_force = brk_force_raw[unique_idx]
        order = np.argsort(brk_cmd)
        brk_cmd = brk_cmd[order]
        brk_force = brk_force[order]
        brk_cmd, cmd_unique_idx = np.unique(brk_cmd, return_index=True)
        brk_force = brk_force[cmd_unique_idx]
        brk_cmd, brk_force = self._add_zero_anchor(brk_cmd, brk_force)

        cmd_min = 0.0
        cmd_max = float(np.max(brk_cmd))
        cmd_min_effective = 0.0
        force_min_effective = 0.0

        return _LookupMap1D(
            cmd_full=brk_cmd,
            force_full=brk_force,
            cmd_min=cmd_min,
            cmd_max=cmd_max,
            cmd_min_effective=cmd_min_effective,
            force_min_effective=float(force_min_effective),
            force_exit_coast=0.0,
        )

    @staticmethod
    def _validate_numeric_array(values: np.ndarray, name: str) -> None:
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1-D after preprocessing.")
        if values.size < 2:
            raise ValueError(f"{name} must contain at least 2 samples.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or Inf values.")

    @staticmethod
    def _interp_extrap(xp: np.ndarray, fp: np.ndarray, x: float) -> float:
        xp = np.asarray(xp, dtype=float).reshape(-1)
        fp = np.asarray(fp, dtype=float).reshape(-1)
        if xp.size != fp.size:
            raise ValueError("Interpolation arrays must have the same length.")
        if xp.size < 2:
            raise ValueError("Interpolation requires at least 2 samples.")

        order = np.argsort(xp)
        xp_sorted = xp[order]
        fp_sorted = fp[order]
        xp_sorted, unique_idx = np.unique(xp_sorted, return_index=True)
        fp_sorted = fp_sorted[unique_idx]

        x = float(x)
        if x <= xp_sorted[0]:
            x0, x1 = xp_sorted[0], xp_sorted[1]
            y0, y1 = fp_sorted[0], fp_sorted[1]
        elif x >= xp_sorted[-1]:
            x0, x1 = xp_sorted[-2], xp_sorted[-1]
            y0, y1 = fp_sorted[-2], fp_sorted[-1]
        else:
            return float(np.interp(x, xp_sorted, fp_sorted))

        dx = x1 - x0
        if abs(dx) < 1e-9:
            return float(y0)
        return float(y0 + (x - x0) * (y1 - y0) / dx)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(min(float(value), float(upper)), float(lower))

    @staticmethod
    def _add_zero_anchor(cmd_axis: np.ndarray, force_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cmd_axis = np.asarray(cmd_axis, dtype=float).reshape(-1)
        force_axis = np.asarray(force_axis, dtype=float).reshape(-1)
        if cmd_axis.size != force_axis.size:
            raise ValueError("Command and force arrays must have the same length.")

        order = np.argsort(cmd_axis)
        cmd_axis = cmd_axis[order]
        force_axis = force_axis[order]
        cmd_axis, unique_idx = np.unique(cmd_axis, return_index=True)
        force_axis = force_axis[unique_idx]

        if cmd_axis.size == 0:
            return np.array([0.0], dtype=float), np.array([0.0], dtype=float)

        if cmd_axis[0] > 0.0:
            cmd_axis = np.concatenate((np.array([0.0]), cmd_axis))
            force_axis = np.concatenate((np.array([0.0]), force_axis))
        else:
            cmd_axis = cmd_axis.copy()
            force_axis = force_axis.copy()
            cmd_axis[0] = 0.0
            force_axis[0] = 0.0

        return cmd_axis, force_axis


def _get_first_existing_field(mat_data: dict, field_names: Iterable[str]):
    for field_name in field_names:
        if field_name in mat_data:
            return mat_data[field_name]
    raise KeyError(f"Required field not found. Tried fields: {tuple(field_names)}")


def _make_col(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)
