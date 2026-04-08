from dataclasses import dataclass
from enum import IntEnum


class SafetyMode(IntEnum):
    NORMAL = 0
    DEGRADED = 1
    PROTECTIVE_BRAKE = 2
    CONTROLLED_STOP = 3


MODE_NAME_BY_VALUE = {
    SafetyMode.NORMAL: "NORMAL",
    SafetyMode.DEGRADED: "DEGRADED",
    SafetyMode.PROTECTIVE_BRAKE: "PROTECTIVE_BRAKE",
    SafetyMode.CONTROLLED_STOP: "CONTROLLED_STOP",
}


@dataclass
class SafetyThresholds:
    lateral_error_enter: float
    heading_error_enter: float
    speed_enter: float
    enter_persistence_sec: float
    lateral_error_exit: float
    heading_error_exit: float
    speed_exit: float
    exit_persistence_sec: float


@dataclass
class SafetyActionLimits:
    target_speed_mps: float
    max_accel_mps2: float
    max_throttle_norm: float
    max_steer_rate_radps: float
    max_steer_abs_rad: float
    overspeed_brake_accel_mps2: float = 0.0
    brake_accel_mps2: float = 0.0
    hold_brake_accel_mps2: float = 0.0
    hold_speed_mps: float = 0.0


@dataclass
class TrackingSafetyConfig:
    enabled: bool
    degraded_thresholds: SafetyThresholds
    protective_brake_thresholds: SafetyThresholds
    controlled_stop_thresholds: SafetyThresholds
    degraded_limits: SafetyActionLimits
    protective_brake_limits: SafetyActionLimits
    controlled_stop_limits: SafetyActionLimits


@dataclass
class SafetySupervisorDebug:
    mode: str
    mode_code: int
    lateral_error: float
    heading_error: float
    speed_mps: float
    target_speed_mps: float
    degraded_timer_sec: float
    protective_brake_timer_sec: float
    controlled_stop_timer_sec: float
    mode_exit_timer_sec: float
    override_active: bool

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mode_code": int(self.mode_code),
            "lateral_error": float(self.lateral_error),
            "heading_error": float(self.heading_error),
            "speed_mps": float(self.speed_mps),
            "target_speed_mps": float(self.target_speed_mps),
            "degraded_timer_sec": float(self.degraded_timer_sec),
            "protective_brake_timer_sec": float(self.protective_brake_timer_sec),
            "controlled_stop_timer_sec": float(self.controlled_stop_timer_sec),
            "mode_exit_timer_sec": float(self.mode_exit_timer_sec),
            "override_active": bool(self.override_active),
        }


@dataclass
class SafetySupervisorOutput:
    steering_cmd_rad: float
    accel_cmd: float
    max_throttle_norm: float
    override_active: bool
    debug: SafetySupervisorDebug


class TrackingSafetySupervisor:
    def __init__(self, config: TrackingSafetyConfig):
        self.config = config
        self.current_mode = SafetyMode.NORMAL
        self._entry_timers = {
            SafetyMode.DEGRADED: 0.0,
            SafetyMode.PROTECTIVE_BRAKE: 0.0,
            SafetyMode.CONTROLLED_STOP: 0.0,
        }
        self._exit_timers = {
            SafetyMode.DEGRADED: 0.0,
            SafetyMode.PROTECTIVE_BRAKE: 0.0,
            SafetyMode.CONTROLLED_STOP: 0.0,
        }
        self._last_steering_cmd_rad = 0.0
        self._steering_initialized = False

    def apply(
        self,
        steering_cmd_rad: float,
        accel_cmd: float,
        vx: float,
        lateral_error: float,
        heading_error: float,
        v_ref: float,
        dt: float,
    ) -> SafetySupervisorOutput:
        dt = max(float(dt), 1e-3)
        lateral_error = abs(float(lateral_error))
        heading_error = abs(float(heading_error))
        vx = abs(float(vx))
        raw_steering_cmd_rad = float(steering_cmd_rad)
        raw_accel_cmd = float(accel_cmd)
        v_ref = max(float(v_ref), 0.0)

        if not self.config.enabled:
            self._last_steering_cmd_rad = raw_steering_cmd_rad
            self._steering_initialized = True
            debug = self._build_debug(
                mode=SafetyMode.NORMAL,
                lateral_error=lateral_error,
                heading_error=heading_error,
                vx=vx,
                target_speed_mps=v_ref,
                override_active=False,
            )
            return SafetySupervisorOutput(
                steering_cmd_rad=raw_steering_cmd_rad,
                accel_cmd=raw_accel_cmd,
                max_throttle_norm=1.0,
                override_active=False,
                debug=debug,
            )

        self._update_entry_timers(lateral_error=lateral_error, heading_error=heading_error, vx=vx, dt=dt)
        self._update_mode(lateral_error=lateral_error, heading_error=heading_error, vx=vx, dt=dt)

        output_steering_cmd_rad = raw_steering_cmd_rad
        output_accel_cmd = raw_accel_cmd
        max_throttle_norm = 1.0
        target_speed_mps = v_ref
        override_active = False

        if self.current_mode == SafetyMode.DEGRADED:
            limits = self.config.degraded_limits
            target_speed_mps = min(v_ref, max(limits.target_speed_mps, 0.0))
            output_accel_cmd = min(raw_accel_cmd, limits.max_accel_mps2)
            if vx > target_speed_mps:
                output_accel_cmd = min(output_accel_cmd, -abs(limits.overspeed_brake_accel_mps2))
            output_steering_cmd_rad = self._smooth_steering(
                raw_steering_cmd_rad,
                limits.max_steer_rate_radps,
                limits.max_steer_abs_rad,
                dt,
            )
            max_throttle_norm = self._clamp(limits.max_throttle_norm, 0.0, 1.0)
            override_active = True
        elif self.current_mode == SafetyMode.PROTECTIVE_BRAKE:
            limits = self.config.protective_brake_limits
            target_speed_mps = min(v_ref, max(limits.target_speed_mps, 0.0))
            output_accel_cmd = min(raw_accel_cmd, -abs(limits.brake_accel_mps2))
            output_steering_cmd_rad = self._smooth_steering(
                raw_steering_cmd_rad,
                limits.max_steer_rate_radps,
                limits.max_steer_abs_rad,
                dt,
            )
            max_throttle_norm = 0.0
            override_active = True
        elif self.current_mode == SafetyMode.CONTROLLED_STOP:
            limits = self.config.controlled_stop_limits
            target_speed_mps = 0.0
            brake_accel = -abs(limits.brake_accel_mps2)
            if vx <= limits.hold_speed_mps:
                brake_accel = -abs(limits.hold_brake_accel_mps2)
            output_accel_cmd = min(raw_accel_cmd, brake_accel)
            output_steering_cmd_rad = self._smooth_steering(
                raw_steering_cmd_rad,
                limits.max_steer_rate_radps,
                limits.max_steer_abs_rad,
                dt,
            )
            max_throttle_norm = 0.0
            override_active = True
        else:
            self._last_steering_cmd_rad = raw_steering_cmd_rad
            self._steering_initialized = True

        debug = self._build_debug(
            mode=self.current_mode,
            lateral_error=lateral_error,
            heading_error=heading_error,
            vx=vx,
            target_speed_mps=target_speed_mps,
            override_active=override_active,
        )
        return SafetySupervisorOutput(
            steering_cmd_rad=output_steering_cmd_rad,
            accel_cmd=output_accel_cmd,
            max_throttle_norm=max_throttle_norm,
            override_active=override_active,
            debug=debug,
        )

    def _update_entry_timers(
        self,
        lateral_error: float,
        heading_error: float,
        vx: float,
        dt: float,
    ) -> None:
        threshold_map = {
            SafetyMode.DEGRADED: self.config.degraded_thresholds,
            SafetyMode.PROTECTIVE_BRAKE: self.config.protective_brake_thresholds,
            SafetyMode.CONTROLLED_STOP: self.config.controlled_stop_thresholds,
        }

        for mode, thresholds in threshold_map.items():
            if self._breach_detected(lateral_error, heading_error, vx, thresholds, use_exit=False):
                self._entry_timers[mode] += dt
            else:
                self._entry_timers[mode] = 0.0

    def _update_mode(
        self,
        lateral_error: float,
        heading_error: float,
        vx: float,
        dt: float,
    ) -> None:
        requested_mode = self._highest_requested_mode()
        if requested_mode > self.current_mode:
            self.current_mode = requested_mode
            self._reset_exit_timers()
            return

        if self.current_mode == SafetyMode.NORMAL:
            return

        thresholds = self._thresholds_for_mode(self.current_mode)
        if self._breach_detected(lateral_error, heading_error, vx, thresholds, use_exit=True):
            self._exit_timers[self.current_mode] = 0.0
            return

        self._exit_timers[self.current_mode] += dt
        if self._exit_timers[self.current_mode] >= thresholds.exit_persistence_sec:
            self.current_mode = requested_mode
            self._reset_exit_timers()

    def _highest_requested_mode(self) -> SafetyMode:
        requested_mode = SafetyMode.NORMAL
        for mode in (SafetyMode.DEGRADED, SafetyMode.PROTECTIVE_BRAKE, SafetyMode.CONTROLLED_STOP):
            thresholds = self._thresholds_for_mode(mode)
            if self._entry_timers[mode] >= thresholds.enter_persistence_sec:
                requested_mode = mode
        return requested_mode

    def _thresholds_for_mode(self, mode: SafetyMode) -> SafetyThresholds:
        if mode == SafetyMode.DEGRADED:
            return self.config.degraded_thresholds
        if mode == SafetyMode.PROTECTIVE_BRAKE:
            return self.config.protective_brake_thresholds
        if mode == SafetyMode.CONTROLLED_STOP:
            return self.config.controlled_stop_thresholds
        raise ValueError(f"Unsupported safety mode: {mode}")

    @staticmethod
    def _breach_detected(
        lateral_error: float,
        heading_error: float,
        vx: float,
        thresholds: SafetyThresholds,
        use_exit: bool,
    ) -> bool:
        if use_exit:
            lat_limit = thresholds.lateral_error_exit
            heading_limit = thresholds.heading_error_exit
            speed_limit = thresholds.speed_exit
        else:
            lat_limit = thresholds.lateral_error_enter
            heading_limit = thresholds.heading_error_enter
            speed_limit = thresholds.speed_enter

        return (
            lateral_error >= lat_limit
            or heading_error >= heading_limit
            or vx >= speed_limit
        )

    def _build_debug(
        self,
        mode: SafetyMode,
        lateral_error: float,
        heading_error: float,
        vx: float,
        target_speed_mps: float,
        override_active: bool,
    ) -> SafetySupervisorDebug:
        return SafetySupervisorDebug(
            mode=MODE_NAME_BY_VALUE[mode],
            mode_code=int(mode),
            lateral_error=lateral_error,
            heading_error=heading_error,
            speed_mps=vx,
            target_speed_mps=target_speed_mps,
            degraded_timer_sec=self._entry_timers[SafetyMode.DEGRADED],
            protective_brake_timer_sec=self._entry_timers[SafetyMode.PROTECTIVE_BRAKE],
            controlled_stop_timer_sec=self._entry_timers[SafetyMode.CONTROLLED_STOP],
            mode_exit_timer_sec=self._exit_timers.get(mode, 0.0),
            override_active=override_active,
        )

    def _smooth_steering(
        self,
        steering_cmd_rad: float,
        max_steer_rate_radps: float,
        max_steer_abs_rad: float,
        dt: float,
    ) -> float:
        steering_cmd_rad = self._clamp(steering_cmd_rad, -abs(max_steer_abs_rad), abs(max_steer_abs_rad))
        if not self._steering_initialized:
            self._last_steering_cmd_rad = steering_cmd_rad
            self._steering_initialized = True
            return steering_cmd_rad

        delta_max = abs(float(max_steer_rate_radps)) * max(float(dt), 1e-3)
        du = steering_cmd_rad - self._last_steering_cmd_rad
        du = self._clamp(du, -delta_max, delta_max)
        steering_output = self._last_steering_cmd_rad + du
        self._last_steering_cmd_rad = steering_output
        return steering_output

    def _reset_exit_timers(self) -> None:
        for mode in self._exit_timers:
            self._exit_timers[mode] = 0.0

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(min(float(value), float(upper)), float(lower))
