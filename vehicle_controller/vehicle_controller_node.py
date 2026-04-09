import math
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray

from vehicle_controller.actuator_mapper import ActuatorMapper
from vehicle_controller.lateral_mpc import LateralMPC
from vehicle_controller.longitudinal_lqr import LongitudinalLQR
from vehicle_controller.mpc_combined import MPCCombined
from vehicle_controller.recorder import VehicleRecorder
from vehicle_controller.reference_manager import ReferenceManager
from vehicle_controller.state_adapter import StateAdapter
from vehicle_controller.types import ControllerMemory


class VehicleControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("vehicle_controller")

        package_root = Path(__file__).resolve().parents[1]
        default_path_file = str(package_root / "data" / "path_ref.mat")

        self.declare_parameter("odom_topic", "/ins/odometry")
        self.declare_parameter("imu_topic", "/ins/imu")
        self.declare_parameter("command_topic", "/command")
        self.declare_parameter("enable_topic", "/enable")

        self.declare_parameter("controller.lateral", "mpc")
        self.declare_parameter("controller.longitudinal", "lqr")
        self.declare_parameter("sim.dt", 0.10)
        self.declare_parameter("sim.progress_window", 80)

        self.declare_parameter("ref.path_file", default_path_file)
        self.declare_parameter("speed.mode", "constant")
        self.declare_parameter("speed.constant_value", 10.0)
        self.declare_parameter("speed.profile", "")

        self.declare_parameter("accel_limits.a_max", 3.372)
        self.declare_parameter("accel_limits.a_min", -7.357)

        self.declare_parameter("mpc_combined.N", 25)
        self.declare_parameter("mpc_combined.Q", [15.0, 12.0, 8.0, 10.0, 8.0, 20.0])
        self.declare_parameter("mpc_combined.R", [5.0, 0.5])
        self.declare_parameter("mpc_combined.Rd", [15.0, 2.0])
        self.declare_parameter("mpc_combined.kappa_ff_gain", 0.5)
        self.declare_parameter("mpc_combined.max_steer", 0.6108652382)
        self.declare_parameter("mpc_combined.a_min", -7.357)
        self.declare_parameter("mpc_combined.a_max", 3.372)

        self.declare_parameter("mpc.N", 25)
        self.declare_parameter("mpc.Q", [15.0, 12.0, 8.0, 10.0])
        self.declare_parameter("mpc.R", 5.0)
        self.declare_parameter("mpc.Rd", 15.0)
        self.declare_parameter("mpc.kappa_ff_gain", 0.5)
        self.declare_parameter("mpc.max_steer", 0.6108652382)

        self.declare_parameter("lon_lqr.Q", [20.0, 10.0])
        self.declare_parameter("lon_lqr.R", 0.4)
        self.declare_parameter("lon_lqr.a_min", -7.357)
        self.declare_parameter("lon_lqr.a_max", 3.372)
        self.declare_parameter("lon_lqr.int_error_min", -10.0)
        self.declare_parameter("lon_lqr.int_error_max", 10.0)

        self.declare_parameter("vehicle.mass", 1948.0)
        self.declare_parameter("vehicle.wheelbase", 2.720)
        self.declare_parameter("vehicle.lf", 1.214)
        self.declare_parameter("vehicle.lr", 1.506)
        self.declare_parameter("vehicle.iz", 2500.0)
        self.declare_parameter("vehicle.aero_a", 45.0)
        self.declare_parameter("vehicle.aero_b", 10.0)
        self.declare_parameter("vehicle.aero_c", 0.518)
        self.declare_parameter("vehicle.max_steer", 0.6108652382)
        self.declare_parameter("vehicle.max_steer_rate", 0.6981317008)
        self.declare_parameter("vehicle.max_pedal_publish", 0.60)
        self.declare_parameter("vehicle.tire.front.b", 9.5)
        self.declare_parameter("vehicle.tire.front.c", 1.30)
        self.declare_parameter("vehicle.tire.front.d", 8000.0)
        self.declare_parameter("vehicle.tire.front.calpha", 98800.0)
        self.declare_parameter("vehicle.tire.rear.b", 10.5)
        self.declare_parameter("vehicle.tire.rear.c", 1.30)
        self.declare_parameter("vehicle.tire.rear.d", 8500.0)
        self.declare_parameter("vehicle.tire.rear.calpha", 116025.0)
        self.declare_parameter("vehicle.accel_map_file", "")
        self.declare_parameter("vehicle.brake_map_file", "")

        self.declare_parameter("timing.dt_min", 0.01)
        self.declare_parameter("timing.dt_max", 0.10)
        self.declare_parameter("timing.min_control_period", 0.10)
        self.declare_parameter("end_condition.max_lateral_error", 5.0)
        self.declare_parameter("end_condition.max_longitudinal_error", 5.0)
        self.declare_parameter("end_condition.goal_index_margin", 3)
        self.declare_parameter("end_condition.goal_lateral_error", 0.2)
        self.declare_parameter("end_condition.goal_longitudinal_error", 0.5)
        self.declare_parameter("end_condition.goal_heading_error", 0.0872664626)
        self.declare_parameter("end_condition.goal_speed_threshold", 0.2)
        self.declare_parameter("recorder.enabled", True)
        self.declare_parameter("recorder.output_dir", "")
        self.declare_parameter("recorder.file_prefix", "vehicle_record")

        odom_topic = self.get_parameter("odom_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        command_topic = self.get_parameter("command_topic").value
        enable_topic = self.get_parameter("enable_topic").value

        controller_lateral = str(self.get_parameter("controller.lateral").value)
        controller_longitudinal = str(self.get_parameter("controller.longitudinal").value)
        control_dt = float(self.get_parameter("sim.dt").value)
        progress_window = int(self.get_parameter("sim.progress_window").value)

        path_file = self.get_parameter("ref.path_file").value
        speed_mode = self.get_parameter("speed.mode").value
        constant_speed = float(self.get_parameter("speed.constant_value").value)
        speed_profile = self.get_parameter("speed.profile").value

        accel_limit_max = float(self.get_parameter("accel_limits.a_max").value)
        accel_limit_min = float(self.get_parameter("accel_limits.a_min").value)

        mpc_horizon = int(self.get_parameter("mpc_combined.N").value)
        mpc_q = self.get_parameter("mpc_combined.Q").value
        mpc_r = self.get_parameter("mpc_combined.R").value
        mpc_rd = self.get_parameter("mpc_combined.Rd").value
        mpc_kappa_ff_gain = float(self.get_parameter("mpc_combined.kappa_ff_gain").value)
        mpc_max_steer = float(self.get_parameter("mpc_combined.max_steer").value)
        mpc_a_min = float(self.get_parameter("mpc_combined.a_min").value)
        mpc_a_max = float(self.get_parameter("mpc_combined.a_max").value)

        lat_mpc_horizon = int(self.get_parameter("mpc.N").value)
        lat_mpc_q = self.get_parameter("mpc.Q").value
        lat_mpc_r = float(self.get_parameter("mpc.R").value)
        lat_mpc_rd = float(self.get_parameter("mpc.Rd").value)
        lat_mpc_kappa_ff_gain = float(self.get_parameter("mpc.kappa_ff_gain").value)
        lat_mpc_max_steer = float(self.get_parameter("mpc.max_steer").value)

        lon_lqr_q = self.get_parameter("lon_lqr.Q").value
        lon_lqr_r = float(self.get_parameter("lon_lqr.R").value)
        lon_lqr_a_min = float(self.get_parameter("lon_lqr.a_min").value)
        lon_lqr_a_max = float(self.get_parameter("lon_lqr.a_max").value)
        lon_lqr_int_error_min = float(self.get_parameter("lon_lqr.int_error_min").value)
        lon_lqr_int_error_max = float(self.get_parameter("lon_lqr.int_error_max").value)

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
        tire_front_calpha = float(self.get_parameter("vehicle.tire.front.calpha").value)
        tire_rear_calpha = float(self.get_parameter("vehicle.tire.rear.calpha").value)
        accel_map_file = self.get_parameter("vehicle.accel_map_file").value
        brake_map_file = self.get_parameter("vehicle.brake_map_file").value

        self.dt_update_min = float(self.get_parameter("timing.dt_min").value)
        self.dt_update_max = float(self.get_parameter("timing.dt_max").value)
        self.min_control_period = float(self.get_parameter("timing.min_control_period").value)
        self.max_lateral_error = float(self.get_parameter("end_condition.max_lateral_error").value)
        self.max_longitudinal_error = float(
            self.get_parameter("end_condition.max_longitudinal_error").value
        )
        self.goal_index_margin = int(self.get_parameter("end_condition.goal_index_margin").value)
        self.goal_lateral_error = float(self.get_parameter("end_condition.goal_lateral_error").value)
        self.goal_longitudinal_error = float(
            self.get_parameter("end_condition.goal_longitudinal_error").value
        )
        self.goal_heading_error = float(self.get_parameter("end_condition.goal_heading_error").value)
        self.goal_speed_threshold = float(
            self.get_parameter("end_condition.goal_speed_threshold").value
        )
        recorder_enabled = bool(self.get_parameter("recorder.enabled").value)
        recorder_output_dir = str(self.get_parameter("recorder.output_dir").value)
        recorder_file_prefix = str(self.get_parameter("recorder.file_prefix").value)

        if mpc_a_min < accel_limit_min or mpc_a_max > accel_limit_max:
            raise ValueError(
                "mpc_combined acceleration limits must stay within accel_limits bounds."
            )
        if lon_lqr_a_min < accel_limit_min or lon_lqr_a_max > accel_limit_max:
            raise ValueError("lon_lqr acceleration limits must stay within accel_limits bounds.")

        self.control_mode = self._resolve_control_mode(
            controller_lateral=controller_lateral,
            controller_longitudinal=controller_longitudinal,
        )

        self.state_adapter = StateAdapter(max_state_age_sec=max(4.0 * control_dt, 0.2))
        self.reference_manager = ReferenceManager(
            path_file=path_file,
            speed_profile=speed_profile,
            speed_mode=speed_mode,
            constant_speed=constant_speed,
            search_window=progress_window,
        )
        self.memory = ControllerMemory()
        self.recorder = None

        self.combined_controller = MPCCombined(
            initial_dt=control_dt,
            horizon=mpc_horizon,
            q=mpc_q,
            r=mpc_r,
            rd=mpc_rd,
            kappa_ff_gain=mpc_kappa_ff_gain,
            max_steer=mpc_max_steer,
            a_min=mpc_a_min,
            a_max=mpc_a_max,
            wheelbase=vehicle_wheelbase,
            mass=vehicle_mass,
            iz=vehicle_iz,
            lf=vehicle_lf,
            lr=vehicle_lr,
            cf=tire_front_calpha,
            cr=tire_rear_calpha,
        )
        self.lateral_controller = LateralMPC(
            initial_dt=control_dt,
            horizon=lat_mpc_horizon,
            q=lat_mpc_q,
            r=lat_mpc_r,
            rd=lat_mpc_rd,
            kappa_ff_gain=lat_mpc_kappa_ff_gain,
            max_steer=lat_mpc_max_steer,
            wheelbase=vehicle_wheelbase,
            mass=vehicle_mass,
            iz=vehicle_iz,
            lf=vehicle_lf,
            lr=vehicle_lr,
            cf=tire_front_calpha,
            cr=tire_rear_calpha,
        )
        self.longitudinal_controller = LongitudinalLQR(
            initial_dt=control_dt,
            q=lon_lqr_q,
            r=lon_lqr_r,
            a_min=lon_lqr_a_min,
            a_max=lon_lqr_a_max,
            int_error_min=lon_lqr_int_error_min,
            int_error_max=lon_lqr_int_error_max,
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
        self.control_dt = control_dt
        default_record_dir = Path.cwd() / "records"
        recorder_dir = recorder_output_dir if recorder_output_dir else str(default_record_dir)
        if recorder_enabled:
            self.recorder = VehicleRecorder(
                output_dir=recorder_dir,
                file_prefix=recorder_file_prefix,
            )

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.command_pub = self.create_publisher(Float32MultiArray, command_topic, 10)
        self.enable_pub = self.create_publisher(Bool, enable_topic, 10)
        self.control_timer = self.create_timer(control_dt, self.control_timer_callback)

        self._last_wait_log_sec = 0.0
        self._prev_control_start_sec: float | None = None
        self._last_control_update_sec: float | None = None
        self._controller_stopped = False

        self.get_logger().info(
            f"vehicle_controller started, odom={odom_topic}, imu={imu_topic}, "
            f"command={command_topic}, enable={enable_topic}, "
            f"initial_dt={control_dt:.3f}s, min_control_period={self.min_control_period:.3f}s, "
            f"controller={self.control_mode}"
        )
        if self.recorder is not None:
            self.get_logger().info(f"recording vehicle data to {self.recorder.path}")

    def odom_callback(self, msg: Odometry) -> None:
        self.state_adapter.update_odometry(msg)

    def imu_callback(self, msg: Imu) -> None:
        self.state_adapter.update_imu(msg)

    def control_timer_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        actual_loop_dt = self._compute_actual_loop_dt(now_sec)

        if not self.state_adapter.is_ready():
            self._publish_enable(False)
            self._throttled_wait_log(now_sec, "waiting for odometry and imu")
            return

        if not self.state_adapter.has_fresh_data(now_sec):
            self._publish_enable(False)
            self._throttled_wait_log(now_sec, "state data received but not fresh enough")
            return

        if self._controller_stopped:
            return

        if not self._should_update_control(now_sec):
            return

        control_update_dt = self._compute_control_update_dt(now_sec)

        meas = self.state_adapter.build_measured_state(self.memory)
        ref_point = self.reference_manager.query(meas, idx_hint=self.memory.idx_progress)
        self.memory.idx_progress = max(self.memory.idx_progress, ref_point.idx)
        e_longitudinal = self._compute_longitudinal_error(meas.x, meas.y, ref_point.xr, ref_point.yr, ref_point.psi_ref)

        stop_reason = self._check_end_condition(ref_point, meas.vx, e_longitudinal)
        if stop_reason is not None:
            self._stop_controller(stop_reason)
            return

        self.memory.int_speed_error += (ref_point.v_ref - meas.vx) * control_update_dt

        if self.control_mode == "mpc_combined":
            delta_cmd, accel_cmd = self.combined_controller.step(
                meas=meas,
                ref_point=ref_point,
                ref_path=self.reference_manager.ref,
                memory=self.memory,
                dt=control_update_dt,
            )
        else:
            delta_cmd = self.lateral_controller.step(
                meas=meas,
                ref_point=ref_point,
                ref_path=self.reference_manager.ref,
                memory=self.memory,
                dt=control_update_dt,
            )
            accel_cmd = self.longitudinal_controller.step(
                v_ref=ref_point.v_ref,
                vx=meas.vx,
                memory=self.memory,
                dt=control_update_dt,
            )

        steering_cmd_raw = self._clamp(delta_cmd, -self.max_steer, self.max_steer)
        steering_exec = self._rate_limit(
            steering_cmd_raw,
            meas.delta_prev,
            self.max_steer_rate,
            control_update_dt,
        )

        accel_cmd_exec = accel_cmd
        cmd, act_dbg = self._build_command(steering_exec, accel_cmd_exec, meas.vx)
        self.memory.last_steering_rad = steering_exec
        self.memory.last_steering_norm = cmd.steering
        self.memory.last_accel_cmd = cmd.accel_cmd

        msg = Float32MultiArray()
        msg.data = cmd.as_command_array()
        self.command_pub.publish(msg)
        self._publish_enable(True)
        self._record_step(
            meas=meas,
            ref_point=ref_point,
            actual_loop_dt=actual_loop_dt,
            control_update_dt=control_update_dt,
            e_longitudinal=e_longitudinal,
            steering_exec=steering_exec,
            cmd=cmd,
            act_dbg=act_dbg,
        )

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
                steering_exec,
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

    @staticmethod
    def _rate_limit(u_cmd: float, u_prev: float, rate_limit: float, dt: float) -> float:
        delta_max = float(rate_limit) * float(dt)
        du = float(u_cmd) - float(u_prev)
        du = max(min(du, delta_max), -delta_max)
        return float(u_prev) + du

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(min(float(value), float(upper)), float(lower))

    def _publish_zero_command(self) -> None:
        msg = Float32MultiArray()
        msg.data = [0.0] * 7
        self.command_pub.publish(msg)

    def _resolve_control_mode(self, controller_lateral: str, controller_longitudinal: str) -> str:
        if controller_lateral == "mpc" and controller_longitudinal == "mpc":
            return "mpc_combined"
        if controller_lateral == "mpc" and controller_longitudinal == "lqr":
            return "mpc+lqr"
        raise ValueError(
            "Supported controller selections are "
            "'controller.lateral=mpc' with 'controller.longitudinal=mpc' or "
            "'controller.lateral=mpc' with 'controller.longitudinal=lqr'."
        )

    def _compute_actual_loop_dt(self, control_start_sec: float) -> float:
        # Initialize with the nominal loop period until we have two callback timestamps.
        if self._prev_control_start_sec is None:
            actual_loop_dt = self.control_dt
        else:
            actual_loop_dt = float(control_start_sec) - float(self._prev_control_start_sec)

        # Update immediately so the stored time always marks the latest callback start.
        self._prev_control_start_sec = float(control_start_sec)

        if actual_loop_dt <= 0.0:
            return self.control_dt

        return self._clamp(actual_loop_dt, self.dt_update_min, self.dt_update_max)

    def _should_update_control(self, now_sec: float) -> bool:
        if self._last_control_update_sec is None:
            return True
        return (float(now_sec) - float(self._last_control_update_sec)) >= self.min_control_period

    def _compute_control_update_dt(self, control_start_sec: float) -> float:
        # Initialize the first true control update with the nominal period.
        if self._last_control_update_sec is None:
            control_update_dt = self.control_dt
        else:
            control_update_dt = float(control_start_sec) - float(self._last_control_update_sec)

        # Store the update timestamp immediately so it tracks the latest solved command.
        self._last_control_update_sec = float(control_start_sec)

        if control_update_dt <= 0.0:
            return self.control_dt

        return self._clamp(control_update_dt, self.dt_update_min, self.dt_update_max)

    def _throttled_wait_log(self, now_sec: float, text: str) -> None:
        if now_sec - self._last_wait_log_sec >= 1.0:
            self.get_logger().info(text)
            self._last_wait_log_sec = now_sec

    @staticmethod
    def _compute_longitudinal_error(
        x: float, y: float, xr: float, yr: float, psi_ref: float
    ) -> float:
        dx = float(x) - float(xr)
        dy = float(y) - float(yr)
        return math.cos(float(psi_ref)) * dx + math.sin(float(psi_ref)) * dy

    def _check_end_condition(self, ref_point, vx: float, e_longitudinal: float) -> str | None:
        if abs(ref_point.e_y) > self.max_lateral_error:
            return f"lateral error exceeded limit: {ref_point.e_y:.3f} m"
        if abs(e_longitudinal) > self.max_longitudinal_error:
            return f"longitudinal error exceeded limit: {e_longitudinal:.3f} m"

        final_ref_idx = len(self.reference_manager.ref.x) - 1
        near_goal = ref_point.idx >= max(final_ref_idx - self.goal_index_margin, 0)
        if not near_goal:
            return None

        if abs(ref_point.e_y) > self.goal_lateral_error:
            return None
        if abs(e_longitudinal) > self.goal_longitudinal_error:
            return None
        if abs(ref_point.e_psi) > self.goal_heading_error:
            return None
        if abs(float(vx)) > self.goal_speed_threshold:
            return None
        return "goal condition satisfied"

    def _stop_controller(self, reason: str) -> None:
        if self._controller_stopped:
            return
        self._controller_stopped = True
        self._publish_zero_command()
        self._publish_enable(False)
        self.get_logger().info(f"controller stopped: {reason}")

    def _record_step(
        self,
        meas,
        ref_point,
        actual_loop_dt: float,
        control_update_dt: float,
        e_longitudinal: float,
        steering_exec: float,
        cmd,
        act_dbg,
    ) -> None:
        if self.recorder is None:
            return

        self.recorder.write(
            {
                "stamp_sec": meas.stamp_sec,
                "actual_loop_dt": actual_loop_dt,
                "control_update_dt": control_update_dt,
                "ref_idx": ref_point.idx,
                "x": meas.x,
                "y": meas.y,
                "yaw": meas.yaw,
                "vx": meas.vx,
                "vy": meas.vy,
                "yaw_rate": meas.yaw_rate,
                "ax": meas.ax,
                "xr": ref_point.xr,
                "yr": ref_point.yr,
                "psi_ref": ref_point.psi_ref,
                "kappa_ref": ref_point.kappa_ref,
                "v_ref": ref_point.v_ref,
                "e_longitudinal": e_longitudinal,
                "e_lateral": ref_point.e_y,
                "e_heading": ref_point.e_psi,
                "steering_rad": steering_exec,
                "steering_norm": cmd.steering,
                "accel_cmd": cmd.accel_cmd,
                "throttle": cmd.throttle,
                "brake": cmd.brake,
                "f_resist": act_dbg.f_resist,
                "f_required": act_dbg.f_required,
            }
        )

    def destroy_node(self):
        if self.recorder is not None:
            self.recorder.close()
        return super().destroy_node()


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
