import json
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

from vehicle_controller.actuator_mapper import ActuatorMapper
from vehicle_controller.lateral_mpc import LateralMPC
from vehicle_controller.longitudinal_lqr import LongitudinalLQR
from vehicle_controller.reference_manager import ReferenceManager
from vehicle_controller.state_adapter import StateAdapter
from vehicle_controller.tracking_safety_supervisor import (
    SafetyActionLimits,
    SafetyThresholds,
    TrackingSafetyConfig,
    TrackingSafetySupervisor,
)
from vehicle_controller.types import ControlOutput, ControllerMemory


class VehicleControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("vehicle_controller")

        package_root = Path(__file__).resolve().parents[1]
        default_path_file = str(package_root / "data" / "path_ref.mat")

        self.declare_parameter("odom_topic", "/ins/odometry")
        self.declare_parameter("imu_topic", "/ins/imu")
        self.declare_parameter("command_topic", "/command")
        self.declare_parameter("enable_topic", "/enable")
        self.declare_parameter("control_dt", 0.05)
        self.declare_parameter("reference.path_file", default_path_file)
        self.declare_parameter("reference.speed_mode", "constant")
        self.declare_parameter("reference.constant_speed", 5.0)
        self.declare_parameter("reference.speed_profile_file", "")
        self.declare_parameter("reference.search_window", 40)
        self.declare_parameter("lateral_mpc.horizon", 25)
        self.declare_parameter("lateral_mpc.q", [15.0, 12.0, 8.0, 10.0])
        self.declare_parameter("lateral_mpc.r", 5.0)
        self.declare_parameter("lateral_mpc.rd", 15.0)
        self.declare_parameter("lateral_mpc.kappa_ff_gain", 0.5)
        self.declare_parameter("vehicle.mass", 1948.0)
        self.declare_parameter("vehicle.wheelbase", 2.720)
        self.declare_parameter("vehicle.lf", 1.214)
        self.declare_parameter("vehicle.lr", 1.506)
        self.declare_parameter("vehicle.iz", 3712.0)
        self.declare_parameter("vehicle.aero_a", 45.0)
        self.declare_parameter("vehicle.aero_b", 10.0)
        self.declare_parameter("vehicle.aero_c", 0.518)
        self.declare_parameter("vehicle.max_steer", 0.6108652382)
        self.declare_parameter("vehicle.max_steer_rate", 0.6981317008)
        self.declare_parameter("vehicle.max_pedal_publish", 0.60)
        self.declare_parameter("vehicle.delay.steer_s", 0.10)
        self.declare_parameter("vehicle.delay.longitudinal_s", 0.10)
        self.declare_parameter("vehicle.tire.front.calpha", 11115.0)
        self.declare_parameter("vehicle.tire.rear.calpha", 12967.5)
        self.declare_parameter("vehicle.accel_map_file", "")
        self.declare_parameter("vehicle.brake_map_file", "")
        self.declare_parameter("longitudinal_lqr.q_speed_error", 8.0)
        self.declare_parameter("longitudinal_lqr.q_int_error", 1.5)
        self.declare_parameter("longitudinal_lqr.r_accel", 0.8)
        self.declare_parameter("longitudinal_lqr.a_min", -3.0)
        self.declare_parameter("longitudinal_lqr.a_max", 2.0)
        self.declare_parameter("longitudinal_lqr.int_error_min", -10.0)
        self.declare_parameter("longitudinal_lqr.int_error_max", 10.0)
        self.declare_parameter("timing.dt_min", 0.01)
        self.declare_parameter("timing.dt_max", 0.10)
        self.declare_parameter("safety.enabled", True)
        self.declare_parameter("safety.debug_topic", "/safety/debug")
        self.declare_parameter("safety.degraded.lateral_error_enter", 0.60)
        self.declare_parameter("safety.degraded.lateral_error_exit", 0.45)
        self.declare_parameter("safety.degraded.heading_error_enter", 0.35)
        self.declare_parameter("safety.degraded.heading_error_exit", 0.25)
        self.declare_parameter("safety.degraded.speed_enter", 12.0)
        self.declare_parameter("safety.degraded.speed_exit", 10.0)
        self.declare_parameter("safety.degraded.enter_persistence_sec", 0.50)
        self.declare_parameter("safety.degraded.exit_persistence_sec", 1.00)
        self.declare_parameter("safety.degraded.target_speed_mps", 4.0)
        self.declare_parameter("safety.degraded.max_accel_mps2", 0.8)
        self.declare_parameter("safety.degraded.max_throttle_norm", 0.35)
        self.declare_parameter("safety.degraded.max_steer_rate_radps", 0.35)
        self.declare_parameter("safety.degraded.max_steer_abs_rad", 0.45)
        self.declare_parameter("safety.degraded.overspeed_brake_accel_mps2", 0.6)
        self.declare_parameter("safety.protective_brake.lateral_error_enter", 1.10)
        self.declare_parameter("safety.protective_brake.lateral_error_exit", 0.90)
        self.declare_parameter("safety.protective_brake.heading_error_enter", 0.70)
        self.declare_parameter("safety.protective_brake.heading_error_exit", 0.50)
        self.declare_parameter("safety.protective_brake.speed_enter", 16.0)
        self.declare_parameter("safety.protective_brake.speed_exit", 13.0)
        self.declare_parameter("safety.protective_brake.enter_persistence_sec", 0.30)
        self.declare_parameter("safety.protective_brake.exit_persistence_sec", 1.00)
        self.declare_parameter("safety.protective_brake.target_speed_mps", 2.0)
        self.declare_parameter("safety.protective_brake.max_accel_mps2", 0.0)
        self.declare_parameter("safety.protective_brake.max_throttle_norm", 0.0)
        self.declare_parameter("safety.protective_brake.max_steer_rate_radps", 0.20)
        self.declare_parameter("safety.protective_brake.max_steer_abs_rad", 0.35)
        self.declare_parameter("safety.protective_brake.brake_accel_mps2", 1.2)
        self.declare_parameter("safety.controlled_stop.lateral_error_enter", 1.80)
        self.declare_parameter("safety.controlled_stop.lateral_error_exit", 1.20)
        self.declare_parameter("safety.controlled_stop.heading_error_enter", 1.00)
        self.declare_parameter("safety.controlled_stop.heading_error_exit", 0.70)
        self.declare_parameter("safety.controlled_stop.speed_enter", 20.0)
        self.declare_parameter("safety.controlled_stop.speed_exit", 15.0)
        self.declare_parameter("safety.controlled_stop.enter_persistence_sec", 0.20)
        self.declare_parameter("safety.controlled_stop.exit_persistence_sec", 1.50)
        self.declare_parameter("safety.controlled_stop.target_speed_mps", 0.0)
        self.declare_parameter("safety.controlled_stop.max_accel_mps2", 0.0)
        self.declare_parameter("safety.controlled_stop.max_throttle_norm", 0.0)
        self.declare_parameter("safety.controlled_stop.max_steer_rate_radps", 0.12)
        self.declare_parameter("safety.controlled_stop.max_steer_abs_rad", 0.20)
        self.declare_parameter("safety.controlled_stop.brake_accel_mps2", 0.8)
        self.declare_parameter("safety.controlled_stop.hold_brake_accel_mps2", 1.5)
        self.declare_parameter("safety.controlled_stop.hold_speed_mps", 0.5)

        odom_topic = self.get_parameter("odom_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        command_topic = self.get_parameter("command_topic").value
        enable_topic = self.get_parameter("enable_topic").value
        safety_debug_topic = self.get_parameter("safety.debug_topic").value
        control_dt = float(self.get_parameter("control_dt").value)

        path_file = self.get_parameter("reference.path_file").value
        speed_mode = self.get_parameter("reference.speed_mode").value
        constant_speed = float(self.get_parameter("reference.constant_speed").value)
        speed_profile_file = self.get_parameter("reference.speed_profile_file").value
        search_window = int(self.get_parameter("reference.search_window").value)
        vehicle_mass = float(self.get_parameter("vehicle.mass").value)
        vehicle_wheelbase = float(self.get_parameter("vehicle.wheelbase").value)
        vehicle_lf = float(self.get_parameter("vehicle.lf").value)
        vehicle_lr = float(self.get_parameter("vehicle.lr").value)
        vehicle_iz = float(self.get_parameter("vehicle.iz").value)
        vehicle_aero_a = float(self.get_parameter("vehicle.aero_a").value)
        vehicle_aero_b = float(self.get_parameter("vehicle.aero_b").value)
        vehicle_aero_c = float(self.get_parameter("vehicle.aero_c").value)
        self.max_steer = float(self.get_parameter("vehicle.max_steer").value)
        self.max_steer_rate = float(self.get_parameter("vehicle.max_steer_rate").value)
        self.max_pedal_publish = float(self.get_parameter("vehicle.max_pedal_publish").value)
        self.steer_delay_s = float(self.get_parameter("vehicle.delay.steer_s").value)
        self.longitudinal_delay_s = float(self.get_parameter("vehicle.delay.longitudinal_s").value)
        tire_front_calpha = float(self.get_parameter("vehicle.tire.front.calpha").value)
        tire_rear_calpha = float(self.get_parameter("vehicle.tire.rear.calpha").value)
        accel_map_file = self.get_parameter("vehicle.accel_map_file").value
        brake_map_file = self.get_parameter("vehicle.brake_map_file").value

        lateral_horizon = int(self.get_parameter("lateral_mpc.horizon").value)
        lateral_q = self.get_parameter("lateral_mpc.q").value
        lateral_r = float(self.get_parameter("lateral_mpc.r").value)
        lateral_rd = float(self.get_parameter("lateral_mpc.rd").value)
        lateral_kappa_ff_gain = float(self.get_parameter("lateral_mpc.kappa_ff_gain").value)

        lqr_q_speed_error = float(self.get_parameter("longitudinal_lqr.q_speed_error").value)
        lqr_q_int_error = float(self.get_parameter("longitudinal_lqr.q_int_error").value)
        lqr_r_accel = float(self.get_parameter("longitudinal_lqr.r_accel").value)
        self.lqr_a_min = float(self.get_parameter("longitudinal_lqr.a_min").value)
        self.lqr_a_max = float(self.get_parameter("longitudinal_lqr.a_max").value)
        lqr_int_error_min = float(self.get_parameter("longitudinal_lqr.int_error_min").value)
        lqr_int_error_max = float(self.get_parameter("longitudinal_lqr.int_error_max").value)
        self.dt_update_min = float(self.get_parameter("timing.dt_min").value)
        self.dt_update_max = float(self.get_parameter("timing.dt_max").value)
        safety_enabled = bool(self.get_parameter("safety.enabled").value)

        self.state_adapter = StateAdapter(max_state_age_sec=max(4.0 * control_dt, 0.2))
        self.reference_manager = ReferenceManager(
            path_file=path_file,
            speed_profile_file=speed_profile_file if speed_profile_file else None,
            speed_mode=speed_mode,
            constant_speed=constant_speed,
            search_window=search_window,
        )
        self.memory = ControllerMemory()
        steer_delay_steps = max(round(self.steer_delay_s / control_dt), 0)
        lon_delay_steps = max(round(self.longitudinal_delay_s / control_dt), 0)
        self.memory.steer_delay_buffer = [0.0] * (steer_delay_steps + 1)
        self.memory.lon_delay_buffer = [0.0] * (lon_delay_steps + 1)

        self.lateral_controller = LateralMPC(
            dt=control_dt,
            horizon=lateral_horizon,
            q=lateral_q,
            r=lateral_r,
            rd=lateral_rd,
            kappa_ff_gain=lateral_kappa_ff_gain,
            max_steer=self.max_steer,
            wheelbase=vehicle_wheelbase,
            mass=vehicle_mass,
            iz=vehicle_iz,
            lf=vehicle_lf,
            lr=vehicle_lr,
            cf=tire_front_calpha,
            cr=tire_rear_calpha,
            steer_delay_steps=steer_delay_steps,
        )
        self.longitudinal_controller = LongitudinalLQR(
            dt=control_dt,
            q_speed_error=lqr_q_speed_error,
            q_int_error=lqr_q_int_error,
            r_accel=lqr_r_accel,
            a_min=self.lqr_a_min,
            a_max=self.lqr_a_max,
            int_error_min=lqr_int_error_min,
            int_error_max=lqr_int_error_max,
        )
        self.actuator_mapper = ActuatorMapper(
            accel_map_file=accel_map_file,
            brake_map_file=brake_map_file,
            max_steer=self.max_steer,
            mass=vehicle_mass,
            aero_a=vehicle_aero_a,
            aero_b=vehicle_aero_b,
            aero_c=vehicle_aero_c,
            max_pedal_publish=self.max_pedal_publish,
        )
        safety_config = TrackingSafetyConfig(
            enabled=safety_enabled,
            degraded_thresholds=SafetyThresholds(
                lateral_error_enter=float(self.get_parameter("safety.degraded.lateral_error_enter").value),
                heading_error_enter=float(self.get_parameter("safety.degraded.heading_error_enter").value),
                speed_enter=float(self.get_parameter("safety.degraded.speed_enter").value),
                enter_persistence_sec=float(self.get_parameter("safety.degraded.enter_persistence_sec").value),
                lateral_error_exit=float(self.get_parameter("safety.degraded.lateral_error_exit").value),
                heading_error_exit=float(self.get_parameter("safety.degraded.heading_error_exit").value),
                speed_exit=float(self.get_parameter("safety.degraded.speed_exit").value),
                exit_persistence_sec=float(self.get_parameter("safety.degraded.exit_persistence_sec").value),
            ),
            protective_brake_thresholds=SafetyThresholds(
                lateral_error_enter=float(self.get_parameter("safety.protective_brake.lateral_error_enter").value),
                heading_error_enter=float(self.get_parameter("safety.protective_brake.heading_error_enter").value),
                speed_enter=float(self.get_parameter("safety.protective_brake.speed_enter").value),
                enter_persistence_sec=float(self.get_parameter("safety.protective_brake.enter_persistence_sec").value),
                lateral_error_exit=float(self.get_parameter("safety.protective_brake.lateral_error_exit").value),
                heading_error_exit=float(self.get_parameter("safety.protective_brake.heading_error_exit").value),
                speed_exit=float(self.get_parameter("safety.protective_brake.speed_exit").value),
                exit_persistence_sec=float(self.get_parameter("safety.protective_brake.exit_persistence_sec").value),
            ),
            controlled_stop_thresholds=SafetyThresholds(
                lateral_error_enter=float(self.get_parameter("safety.controlled_stop.lateral_error_enter").value),
                heading_error_enter=float(self.get_parameter("safety.controlled_stop.heading_error_enter").value),
                speed_enter=float(self.get_parameter("safety.controlled_stop.speed_enter").value),
                enter_persistence_sec=float(self.get_parameter("safety.controlled_stop.enter_persistence_sec").value),
                lateral_error_exit=float(self.get_parameter("safety.controlled_stop.lateral_error_exit").value),
                heading_error_exit=float(self.get_parameter("safety.controlled_stop.heading_error_exit").value),
                speed_exit=float(self.get_parameter("safety.controlled_stop.speed_exit").value),
                exit_persistence_sec=float(self.get_parameter("safety.controlled_stop.exit_persistence_sec").value),
            ),
            degraded_limits=SafetyActionLimits(
                target_speed_mps=float(self.get_parameter("safety.degraded.target_speed_mps").value),
                max_accel_mps2=float(self.get_parameter("safety.degraded.max_accel_mps2").value),
                max_throttle_norm=float(self.get_parameter("safety.degraded.max_throttle_norm").value),
                max_steer_rate_radps=float(self.get_parameter("safety.degraded.max_steer_rate_radps").value),
                max_steer_abs_rad=float(self.get_parameter("safety.degraded.max_steer_abs_rad").value),
                overspeed_brake_accel_mps2=float(self.get_parameter("safety.degraded.overspeed_brake_accel_mps2").value),
            ),
            protective_brake_limits=SafetyActionLimits(
                target_speed_mps=float(self.get_parameter("safety.protective_brake.target_speed_mps").value),
                max_accel_mps2=float(self.get_parameter("safety.protective_brake.max_accel_mps2").value),
                max_throttle_norm=float(self.get_parameter("safety.protective_brake.max_throttle_norm").value),
                max_steer_rate_radps=float(self.get_parameter("safety.protective_brake.max_steer_rate_radps").value),
                max_steer_abs_rad=float(self.get_parameter("safety.protective_brake.max_steer_abs_rad").value),
                brake_accel_mps2=float(self.get_parameter("safety.protective_brake.brake_accel_mps2").value),
            ),
            controlled_stop_limits=SafetyActionLimits(
                target_speed_mps=float(self.get_parameter("safety.controlled_stop.target_speed_mps").value),
                max_accel_mps2=float(self.get_parameter("safety.controlled_stop.max_accel_mps2").value),
                max_throttle_norm=float(self.get_parameter("safety.controlled_stop.max_throttle_norm").value),
                max_steer_rate_radps=float(self.get_parameter("safety.controlled_stop.max_steer_rate_radps").value),
                max_steer_abs_rad=float(self.get_parameter("safety.controlled_stop.max_steer_abs_rad").value),
                brake_accel_mps2=float(self.get_parameter("safety.controlled_stop.brake_accel_mps2").value),
                hold_brake_accel_mps2=float(self.get_parameter("safety.controlled_stop.hold_brake_accel_mps2").value),
                hold_speed_mps=float(self.get_parameter("safety.controlled_stop.hold_speed_mps").value),
            ),
        )
        self.safety_supervisor = TrackingSafetySupervisor(safety_config)
        self.control_dt = control_dt

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_callback, 10
        )
        self.command_pub = self.create_publisher(Float32MultiArray, command_topic, 10)
        self.enable_pub = self.create_publisher(Bool, enable_topic, 10)
        self.safety_debug_pub = self.create_publisher(String, safety_debug_topic, 10)
        self.control_timer = self.create_timer(control_dt, self.control_timer_callback)

        self._last_wait_log_sec = 0.0

        self.get_logger().info(
            f"vehicle_controller started, odom={odom_topic}, imu={imu_topic}, "
            f"command={command_topic}, enable={enable_topic}, safety_debug={safety_debug_topic}, dt={control_dt:.3f}s"
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.state_adapter.update_odometry(msg)

    def imu_callback(self, msg: Imu) -> None:
        self.state_adapter.update_imu(msg)

    def control_timer_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if not self.state_adapter.is_ready():
            self._publish_enable(False)
            self._throttled_wait_log(now_sec, "waiting for odometry and imu")
            return

        if not self.state_adapter.has_fresh_data(now_sec):
            self._publish_enable(False)
            self._throttled_wait_log(now_sec, "state data received but not fresh enough")
            return

        meas = self.state_adapter.build_measured_state(self.memory)
        dt_update = self._compute_update_dt(meas.stamp_sec)
        ref_point = self.reference_manager.query(meas, idx_hint=self.memory.idx_progress)
        self.memory.idx_progress = max(self.memory.idx_progress, ref_point.idx)

        steering_cmd_raw = self.lateral_controller.step(
            meas=meas,
            ref_point=ref_point,
            ref_path=self.reference_manager.ref,
            memory=self.memory,
        )
        steering_cmd_raw = self._clamp(steering_cmd_raw, -self.max_steer, self.max_steer)
        steering_cmd_delayed = self._apply_delay(steering_cmd_raw, self.memory.steer_delay_buffer)
        steering_exec = self._rate_limit(
            steering_cmd_delayed,
            meas.delta_prev,
            self.max_steer_rate,
            dt_update,
        )

        accel_cmd_raw = self.longitudinal_controller.step(
            v_ref=ref_point.v_ref,
            vx=meas.vx,
            memory=self.memory,
            dt=dt_update,
        )
        accel_cmd_exec = self._apply_delay(accel_cmd_raw, self.memory.lon_delay_buffer)
        safety_output = self.safety_supervisor.apply(
            steering_cmd_rad=steering_exec,
            accel_cmd=accel_cmd_exec,
            vx=meas.vx,
            lateral_error=ref_point.e_y,
            heading_error=ref_point.e_psi,
            v_ref=ref_point.v_ref,
            dt=dt_update,
        )

        cmd, act_dbg = self._build_command(
            safety_output.steering_cmd_rad,
            safety_output.accel_cmd,
            meas.vx,
        )
        cmd.throttle = min(cmd.throttle, safety_output.max_throttle_norm)
        act_dbg.throttle_norm = cmd.throttle
        act_dbg.throttle_publish = cmd.throttle * self.max_pedal_publish
        self.memory.last_steering_rad = safety_output.steering_cmd_rad
        self.memory.last_steering_norm = cmd.steering
        self.memory.last_accel_cmd = cmd.accel_cmd

        msg = Float32MultiArray()
        msg.data = cmd.as_command_array()
        self.command_pub.publish(msg)
        self._publish_enable(True)
        self._publish_safety_debug(safety_output.debug)

        self.get_logger().debug(
            "x=%.3f y=%.3f yaw=%.3f vx=%.3f | idx=%d ey=%.3f epsi=%.3f vref=%.3f "
            "steer_rad=%.3f steer_norm=%.3f ax_cmd=%.3f thr=%.3f brk=%.3f "
            "f_resist=%.3f f_req=%.3f thr_pub=%.3f brk_pub=%.3f"
            % (
                meas.x,
                meas.y,
                meas.yaw,
                meas.vx,
                ref_point.idx,
                ref_point.e_y,
                ref_point.e_psi,
                ref_point.v_ref,
                safety_output.steering_cmd_rad,
                cmd.steering,
                cmd.accel_cmd,
                cmd.throttle,
                cmd.brake,
                act_dbg.f_resist,
                act_dbg.f_required,
                act_dbg.throttle_publish,
                act_dbg.brake_publish,
            )
        )

    def _build_command(self, steering_cmd: float, accel_cmd: float, vx: float):
        return self.actuator_mapper.map_command(steering_cmd, accel_cmd, vx)

    def _publish_enable(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self.enable_pub.publish(msg)

    def _publish_safety_debug(self, debug) -> None:
        msg = String()
        msg.data = json.dumps(debug.to_dict(), ensure_ascii=True)
        self.safety_debug_pub.publish(msg)

    @staticmethod
    def _apply_delay(u_cmd: float, buffer: list[float]) -> float:
        if len(buffer) <= 1:
            if buffer:
                buffer[0] = u_cmd
            return u_cmd

        u_exec = buffer[0]
        for i in range(len(buffer) - 1):
            buffer[i] = buffer[i + 1]
        buffer[-1] = u_cmd
        return float(u_exec)

    @staticmethod
    def _rate_limit(u_cmd: float, u_prev: float, rate_limit: float, dt: float) -> float:
        delta_max = float(rate_limit) * float(dt)
        du = float(u_cmd) - float(u_prev)
        du = max(min(du, delta_max), -delta_max)
        return float(u_prev) + du

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(min(float(value), float(upper)), float(lower))

    def _compute_update_dt(self, stamp_sec: float) -> float:
        last_stamp_sec = self.memory.last_update_stamp_sec
        self.memory.last_update_stamp_sec = float(stamp_sec)

        if last_stamp_sec <= 0.0:
            return self.control_dt

        dt_update = float(stamp_sec) - float(last_stamp_sec)
        return self._clamp(dt_update, self.dt_update_min, self.dt_update_max)

    def _throttled_wait_log(self, now_sec: float, text: str) -> None:
        if now_sec - self._last_wait_log_sec >= 1.0:
            self.get_logger().info(text)
            self._last_wait_log_sec = now_sec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VehicleControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
