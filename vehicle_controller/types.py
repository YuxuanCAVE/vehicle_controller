from dataclasses import dataclass, field
from typing import List


@dataclass
class MeasuredState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    ax: float = 0.0
    delta_prev: float = 0.0
    stamp_sec: float = 0.0


@dataclass
class ReferencePoint:
    idx: int = 0
    xr: float = 0.0
    yr: float = 0.0
    psi_ref: float = 0.0
    kappa_ref: float = 0.0
    v_ref: float = 0.0
    e_y: float = 0.0
    e_psi: float = 0.0


@dataclass
class ControllerMemory:
    idx_progress: int = 0
    int_speed_error: float = 0.0
    last_steering_rad: float = 0.0
    last_steering_norm: float = 0.0
    last_accel_cmd: float = 0.0
    steer_delay_buffer: List[float] = field(default_factory=list)
    lon_delay_buffer: List[float] = field(default_factory=list)


@dataclass
class ControlOutput:
    brake: float = 0.0
    throttle: float = 0.0
    steering: float = 0.0
    accel_cmd: float = 0.0

    def as_command_array(self) -> List[float]:
        return [
            float(self.brake),
            float(self.throttle),
            float(self.steering),
            0.0,
            0.0,
            0.0,
            0.0,
        ]


@dataclass
class ActuatorDebug:
    f_resist: float = 0.0
    f_required: float = 0.0
    acc_req: float = 0.0
    brk_req: float = 0.0
    throttle_publish: float = 0.0
    brake_publish: float = 0.0
    throttle_norm: float = 0.0
    brake_norm: float = 0.0
