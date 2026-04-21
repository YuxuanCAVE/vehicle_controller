import math
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sygnal_msgs.msg import FaultState, InterfaceCommand, InterfaceEnable, State

from vehicle_controller.actuator_mapper import ActuatorMapper
from vehicle_controller.longitudinal_pid import LongitudinalPID
from vehicle_controller.nmpc_kbm_lateral import NMPCKBMLateral
from vehicle_controller.reference_manager import ReferenceManager
from vehicle_controller.state_adapter import StateAdapter
from vehicle_controller.types import ControllerMemory


CONTROLLER_RECORD_FIELDS = [
    "stamp_sec",
    "actual_loop_dt",
    "control_update_dt",
    "ref_idx",
    "x",
    "y",
    "yaw",
    "vx",
    "vy",
    "yaw_rate",
    "ax",
    "xr",
    "yr",
    "psi_ref",
    "kappa_ref",
    "v_ref",
    "e_longitudinal",
    "e_lateral",
    "e_heading",
    "steering_rad",
    "steering_command",
    "accel_cmd",
    "throttle",
    "brake",
    "f_resist",
    "f_required",
]


class VehicleControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("vehicle_controller")

        package_share_dir = Path(get_package_share_directory("vehicle_controller"))
        package_source_dir = Path(__file__).resolve().parents[1]
        default_path_file = str(package_share_dir / "data" / "path_ref.mat")
        default_accel_map_file = str(package_share_dir / "data" / "Acc_mapData_noSlope.mat")
        default_brake_map_file = str(package_share_dir / "data" / "brake_mapData_noSlope.mat")

        self.declare_parameter("odom_topic", "/ins/odometry")
        self.declare_parameter("imu_topic", "/ins/imu")
        self.declare_parameter("command_topic", "/command")
        self.declare_parameter("enable_topic", "/enable")
        self.declare_parameter("record_topic", "/controller_record")
        self.declare_parameter("sygnal_state_topic", "/sygnal_state")
        self.declare_parameter("sygnal_fault_topic", "/sygnal_fault")

        self.declare_parameter("controller.lateral", "nmpc_kbm")
        self.declare_parameter("controller.longitudinal", "pid")
        self.declare_parameter("sim.dt", 0.10)
        self.declare_parameter("sim.progress_window", 5)

        self.declare_parameter("ref.path_file", default_path_file)
        self.declare_parameter("speed.mode", "constant")
        self.declare_parameter("speed.constant_value", 3.0)
        self.declare_parameter("speed.profile", "")
        self.declare_parameter("speed.smoothing_enabled", True)
        self.declare_parameter("speed.smoothing_accel_scale", 0.85)
        self.declare_parameter("speed.initial_smoothed_value", 0.5)

        self.declare_parameter("accel_limits.a_max", 3.372)
        self.declare_parameter("accel_limits.a_min", -7.357)

        self.declare_parameter("nmpc_kbm.N", 5)
        self.declare_parameter("nmpc_kbm.q_x", 32.0)
        self.declare_parameter("nmpc_kbm.q_y", 32.0)
        self.declare_parameter("nmpc_kbm.q_psi", 4.0)
        self.declare_parameter("nmpc_kbm.r_delta", 32.0)
        self.declare_parameter("nmpc_kbm.r_du", 32.0)
        self.declare_parameter("nmpc_kbm.max_iterations", 100)
        self.declare_parameter("nmpc_kbm.max_fun_evals", 4000)

        self.declare_parameter("lon_pid.kp", 1.6)
        self.declare_parameter("lon_pid.ki", 0.0)
        self.declare_parameter("lon_pid.kd", 0.0)
        self.declare_parameter("lon_pid.a_min", -7.357)
        self.declare_parameter("lon_pid.a_max", 3.372)
        self.declare_parameter("lon_pid.int_error_min", -10.0)
        self.declare_parameter("lon_pid.int_error_max", 10.0)

        self.declare_parameter("vehicle.mass", 1948.0)
        self.declare_parameter("vehicle.wheelbase", 2.720)
        self.declare_parameter("vehicle.lr", 1.506)
        self.declare_parameter("vehicle.aero_a", 45.0)
        self.declare_parameter("vehicle.aero_b", 10.0)
        self.declare_parameter("vehicle.aero_c", 0.518)
        self.declare_parameter("vehicle.max_steer", 0.3490658504)
        self.declare_parameter("vehicle.max_steer_rate", 0.2617993878)
        self.declare_parameter("vehicle.steering_sign", -1.0)
        self.declare_parameter("vehicle.max_pedal_publish", 0.60)
        self.declare_parameter("vehicle.sensor_offset_x", 1.14)
        self.declare_parameter("vehicle.sensor_offset_y", 0.32)
        self.declare_parameter("vehicle.sensor_offset_z", 0.0)
        self.declare_parameter("vehicle.accel_map_file", default_accel_map_file)
        self.declare_parameter("vehicle.brake_map_file", default_brake_map_file)

        self.declare_parameter("timing.dt_min", 0.01)
        self.declare_parameter("timing.dt_max", 0.10)
        self.declare_parameter("timing.min_control_period", 0.10)
        self.declare_parameter("timing.startup_warmup_sec", 0.0)
        self.declare_parameter("end_condition.max_lateral_error", 7.0)
        self.declare_parameter("end_condition.max_longitudinal_error", 7.0)
        self.declare_parameter("end_condition.goal_index_margin", 3)
        self.declare_parameter("end_condition.goal_lateral_error", 2.0)
        self.declare_parameter("end_condition.goal_longitudinal_error", 2.0)
        self.declare_parameter("end_condition.goal_heading_error", 0.0872664626)
        self.declare_parameter("end_condition.goal_speed_threshold", 2.0)

        odom_topic = self.get_parameter("odom_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        command_topic = self.get_parameter("command_topic").value
        enable_topic = self.get_parameter("enable_topic").value
        record_topic = self.get_parameter("record_topic").value
        sygnal_state_topic = self.get_parameter("sygnal_state_topic").value
        sygnal_fault_topic = self.get_parameter("sygnal_fault_topic").value

        controller_lateral = str(self.get_parameter("controller.lateral").value)
        controller_longitudinal = str(self.get_parameter("controller.longitudinal").value)
        control_dt = float(self.get_parameter("sim.dt").value)
        progress_window = int(self.get_parameter("sim.progress_window").value)

        path_file = self._resolve_param_path(
            self.get_parameter("ref.path_file").value,
            package_share_dir=package_share_dir,
            package_source_dir=package_source_dir,
        )
        speed_mode = self.get_parameter("speed.mode").value
        constant_speed = float(self.get_parameter("speed.constant_value").value)
        speed_profile = self.get_parameter("speed.profile").value
        speed_smoothing_enabled = bool(self.get_parameter("speed.smoothing_enabled").value)
        speed_smoothing_accel_scale = float(
            self.get_parameter("speed.smoothing_accel_scale").value
        )
        speed_initial_smoothed_value = float(
            self.get_parameter("speed.initial_smoothed_value").value
        )

        accel_limit_max = float(self.get_parameter("accel_limits.a_max").value)
        accel_limit_min = float(self.get_parameter("accel_limits.a_min").value)
        self.accel_limit_max = accel_limit_max
        self.accel_limit_min = accel_limit_min

        nmpc_horizon = int(self.get_parameter("nmpc_kbm.N").value)
        nmpc_q_x = float(self.get_parameter("nmpc_kbm.q_x").value)
        nmpc_q_y = float(self.get_parameter("nmpc_kbm.q_y").value)
        nmpc_q_psi = float(self.get_parameter("nmpc_kbm.q_psi").value)
        nmpc_r_delta = float(self.get_parameter("nmpc_kbm.r_delta").value)
        nmpc_r_du = float(self.get_parameter("nmpc_kbm.r_du").value)
        nmpc_max_iterations = int(self.get_parameter("nmpc_kbm.max_iterations").value)
        nmpc_max_fun_evals = int(self.get_parameter("nmpc_kbm.max_fun_evals").value)

        lon_pid_kp = float(self.get_parameter("lon_pid.kp").value)
        lon_pid_ki = float(self.get_parameter("lon_pid.ki").value)
        lon_pid_kd = float(self.get_parameter("lon_pid.kd").value)
        lon_pid_a_min = float(self.get_parameter("lon_pid.a_min").value)
        lon_pid_a_max = float(self.get_parameter("lon_pid.a_max").value)
        lon_pid_int_error_min = float(self.get_parameter("lon_pid.int_error_min").value)
        lon_pid_int_error_max = float(self.get_parameter("lon_pid.int_error_max").value)

        vehicle_mass = float(self.get_parameter("vehicle.mass").value)
        vehicle_wheelbase = float(self.get_parameter("vehicle.wheelbase").value)
        vehicle_lr = float(self.get_parameter("vehicle.lr").value)
        vehicle_aero_a = float(self.get_parameter("vehicle.aero_a").value)
        vehicle_aero_b = float(self.get_parameter("vehicle.aero_b").value)
        vehicle_aero_c = float(self.get_parameter("vehicle.aero_c").value)
        self.max_steer = float(self.get_parameter("vehicle.max_steer").value)
        self.max_steer_rate = float(self.get_parameter("vehicle.max_steer_rate").value)
        vehicle_steering_sign = float(self.get_parameter("vehicle.steering_sign").value)
        self.max_pedal_publish = float(self.get_parameter("vehicle.max_pedal_publish").value)
        vehicle_sensor_offset_x = float(self.get_parameter("vehicle.sensor_offset_x").value)
        vehicle_sensor_offset_y = float(self.get_parameter("vehicle.sensor_offset_y").value)
        vehicle_sensor_offset_z = float(self.get_parameter("vehicle.sensor_offset_z").value)
        accel_map_file = self._resolve_param_path(
            self.get_parameter("vehicle.accel_map_file").value,
            package_share_dir=package_share_dir,
            package_source_dir=package_source_dir,
        )
        brake_map_file = self._resolve_param_path(
            self.get_parameter("vehicle.brake_map_file").value,
            package_share_dir=package_share_dir,
            package_source_dir=package_source_dir,
        )

        self.dt_update_min = float(self.get_parameter("timing.dt_min").value)
        self.dt_update_max = float(self.get_parameter("timing.dt_max").value)
        self.min_control_period = float(self.get_parameter("timing.min_control_period").value)
        self.startup_warmup_sec = max(
            float(self.get_parameter("timing.startup_warmup_sec").value),
            0.0,
        )
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
        self.speed_smoothing_enabled = speed_smoothing_enabled
        self.speed_smoothing_accel_scale = speed_smoothing_accel_scale
        self.speed_initial_smoothed_value = speed_initial_smoothed_value

        if lon_pid_a_min < accel_limit_min or lon_pid_a_max > accel_limit_max:
            raise ValueError("lon_pid acceleration limits must stay within accel_limits bounds.")

        self.control_mode = self._resolve_control_mode(
            controller_lateral=controller_lateral,
            controller_longitudinal=controller_longitudinal,
        )

        self.state_adapter = StateAdapter(
            max_state_age_sec=max(4.0 * control_dt, 0.2),
            sensor_offset_x=vehicle_sensor_offset_x,
            sensor_offset_y=vehicle_sensor_offset_y,
            sensor_offset_z=vehicle_sensor_offset_z,
        )
        self.reference_manager = ReferenceManager(
            path_file=path_file,
            speed_profile=speed_profile,
            speed_mode=speed_mode,
            constant_speed=constant_speed,
            search_window=progress_window,
        )
        self.memory = ControllerMemory()

        self.lateral_controller = NMPCKBMLateral(
            initial_dt=control_dt,
            horizon=nmpc_horizon,
            q_x=nmpc_q_x,
            q_y=nmpc_q_y,
            q_psi=nmpc_q_psi,
            r_delta=nmpc_r_delta,
            r_du=nmpc_r_du,
            max_steer=self.max_steer,
            wheelbase=vehicle_wheelbase,
            lr=vehicle_lr,
            delta_rate_max=self.max_steer_rate,
            max_iterations=nmpc_max_iterations,
            max_fun_evals=nmpc_max_fun_evals,
        )
        if controller_longitudinal != "pid":
            raise ValueError(f"Unsupported longitudinal controller: {controller_longitudinal}")
        self.longitudinal_controller = LongitudinalPID(
            kp=lon_pid_kp,
            ki=lon_pid_ki,
            kd=lon_pid_kd,
            a_min=lon_pid_a_min,
            a_max=lon_pid_a_max,
            int_error_min=lon_pid_int_error_min,
            int_error_max=lon_pid_int_error_max,
        )
        self.actuator_mapper = ActuatorMapper(
            accel_map_file=accel_map_file,
            brake_map_file=brake_map_file,
            max_steer=self.max_steer,
            steering_sign=vehicle_steering_sign,
            mass=vehicle_mass,
            aero_a=vehicle_aero_a,
            aero_b=vehicle_aero_b,
            aero_c=vehicle_aero_c,
            max_pedal_publish=self.max_pedal_publish,
        )
        self.control_dt = control_dt

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.sygnal_state_sub = self.create_subscription(
            State, sygnal_state_topic, self.sygnal_state_callback, 10
        )
        self.sygnal_fault_sub = self.create_subscription(
            FaultState, sygnal_fault_topic, self.sygnal_fault_callback, 10
        )
        self.command_pub = self.create_publisher(InterfaceCommand, command_topic, 10)
        self.enable_pub = self.create_publisher(InterfaceEnable, enable_topic, 10)
        self.record_pub = self.create_publisher(Float32MultiArray, record_topic, 10)
        self.control_timer = self.create_timer(control_dt, self.control_timer_callback)

        self.sygnal_system_state = ["UNKNOWN", "UNKNOWN"]
        self.sygnal_interface_state = [False] * 7
        self.sygnal_fault_active = False
        self.sygnal_fault_summary = "none"
        self._last_wait_log_sec = 0.0
        self._prev_control_start_sec: float | None = None
        self._last_control_update_sec: float | None = None
        self._startup_warmup_start_sec: float | None = None

        self.get_logger().info(
            f"vehicle_controller started, odom={odom_topic}, imu={imu_topic}, "
            f"command={command_topic}, enable={enable_topic}, record={record_topic}, "
            f"sygnal_state={sygnal_state_topic}, sygnal_fault={sygnal_fault_topic}, "
            f"initial_dt={control_dt:.3f}s, min_control_period={self.min_control_period:.3f}s, "
            f"startup_warmup={self.startup_warmup_sec:.3f}s, "
            f"controller={self.control_mode}"
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.state_adapter.update_odometry(msg)

    def imu_callback(self, msg: Imu) -> None:
        self.state_adapter.update_imu(msg)

    def sygnal_state_callback(self, msg: State) -> None:
        self.sygnal_system_state = list(msg.system_state)
        self.sygnal_interface_state = [bool(value) for value in msg.interface_state]

    def sygnal_fault_callback(self, msg: FaultState) -> None:
        active_faults = sum(int(fault.fault_count) for fault in msg.fault_list)
        self.sygnal_fault_active = (
            int(msg.op1_cause) != 0
            or int(msg.op2_cause) != 0
            or int(msg.hard_cause) != 0
            or active_faults > 0
        )
        self.sygnal_fault_summary = (
            f"op1={int(msg.op1_cause)} op2={int(msg.op2_cause)} "
            f"hard={int(msg.hard_cause)} count={active_faults}"
        )

    def control_timer_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        actual_loop_dt = self._compute_actual_loop_dt(now_sec)

        if self.sygnal_fault_active:
            self._reset_startup_warmup()
            self._publish_zero_command()
            self._publish_enable(False)
            self._throttled_wait_log(
                now_sec,
                f"sygnal fault active: {self.sygnal_fault_summary}",
            )
            return

        if not self.state_adapter.is_ready():
            self._reset_startup_warmup()
            self._publish_enable(False)
            self._throttled_wait_log(now_sec, "waiting for odometry and imu")
            return

        if not self.state_adapter.has_fresh_data(now_sec):
            self._reset_startup_warmup()
            self._publish_enable(False)
            self._throttled_wait_log(now_sec, "state data received but not fresh enough")
            return

        if not self._should_update_control(now_sec):
            return

        control_update_dt = self._compute_control_update_dt(now_sec)

        meas = self.state_adapter.build_measured_state(self.memory)
        idx_hint = self.memory.idx_progress if self.memory.has_reference_lock else None
        ref_point = self.reference_manager.query(meas, idx_hint=idx_hint)
        self.memory.has_reference_lock = True
        self.memory.idx_progress = max(self.memory.idx_progress, ref_point.idx)
        ref_point.v_ref = self._smooth_reference_speed(ref_point.v_ref, meas.vx, control_update_dt)
        e_longitudinal = self._compute_longitudinal_error(meas.x, meas.y, ref_point.xr, ref_point.yr, ref_point.psi_ref)

        stop_reason = self._check_end_condition(ref_point, meas.vx, e_longitudinal)
        if stop_reason is not None:
            self._publish_zero_command()
            self._publish_enable(False)
            self.get_logger().info(stop_reason)
            return

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

        if self._in_startup_warmup(now_sec):
            self._publish_zero_command()
            self._publish_enable(False)
            self._publish_record(
                meas=meas,
                ref_point=ref_point,
                actual_loop_dt=actual_loop_dt,
                control_update_dt=control_update_dt,
                e_longitudinal=e_longitudinal,
                steering_exec=steering_exec,
                cmd=cmd,
                act_dbg=act_dbg,
            )
            self._throttled_wait_log(
                now_sec,
                (
                    "warming up controller before enabling commands "
                    f"({self.startup_warmup_sec:.2f}s)"
                ),
            )
            return

        self.memory.last_steering_rad = steering_exec
        self.memory.last_steering_command = cmd.steering
        self.memory.last_accel_cmd = cmd.accel_cmd

        msg = InterfaceCommand()
        msg.command = cmd.as_command_array()
        self.command_pub.publish(msg)
        self._publish_enable(True)
        self._publish_record(
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
            "steer_rad=%.3f steer_cmd=%.3f ax_cmd=%.3f thr=%.3f brk=%.3f "
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
        msg = InterfaceEnable()
        msg.enable = [bool(enabled), bool(enabled), bool(enabled), False, False, False, False]
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

    @staticmethod
    def _resolve_param_path(
        path_value: str,
        package_share_dir: Path,
        package_source_dir: Path,
    ) -> str:
        path = Path(str(path_value)).expanduser()
        if path.is_absolute():
            return str(path)

        share_candidate = package_share_dir / path
        if share_candidate.exists():
            return str(share_candidate)

        source_candidate = package_source_dir / path
        if source_candidate.exists():
            return str(source_candidate)

        return str(path)

    def _publish_zero_command(self) -> None:
        msg = InterfaceCommand()
        msg.command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.command_pub.publish(msg)

    def _resolve_control_mode(self, controller_lateral: str, controller_longitudinal: str) -> str:
        if controller_lateral in {"nmpc_kbm", "kinematic_mpc"} and controller_longitudinal == "pid":
            return "nmpc_kbm+pid"
        raise ValueError(
            "Supported controller selection is "
            "'controller.lateral=nmpc_kbm' with 'controller.longitudinal=pid'. "
            "The legacy value 'kinematic_mpc' is also accepted as an alias."
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

    def _reset_startup_warmup(self) -> None:
        self._startup_warmup_start_sec = None

    def _in_startup_warmup(self, now_sec: float) -> bool:
        if self.startup_warmup_sec <= 0.0:
            return False
        if self._startup_warmup_start_sec is None:
            self._startup_warmup_start_sec = float(now_sec)
            return True
        return (float(now_sec) - float(self._startup_warmup_start_sec)) < self.startup_warmup_sec

    def _smooth_reference_speed(self, v_ref_raw: float, vx: float, dt: float) -> float:
        v_ref_raw = max(float(v_ref_raw), 0.0)
        if not self.speed_smoothing_enabled:
            self.memory.v_ref_smooth = v_ref_raw
            self.memory.has_speed_ref_lock = True
            return v_ref_raw

        if not self.memory.has_speed_ref_lock:
            self.memory.v_ref_smooth = max(float(vx), self.speed_initial_smoothed_value, 0.0)
            self.memory.has_speed_ref_lock = True

        a_max_ref = self.speed_smoothing_accel_scale * self.accel_limit_max
        a_min_ref = self.speed_smoothing_accel_scale * self.accel_limit_min
        dv_desired = v_ref_raw - self.memory.v_ref_smooth
        dv_max = a_max_ref * float(dt)
        dv_min = a_min_ref * float(dt)
        dv_clamped = min(max(dv_desired, dv_min), dv_max)
        self.memory.v_ref_smooth = max(self.memory.v_ref_smooth + dv_clamped, 0.0)
        return self.memory.v_ref_smooth

    @staticmethod
    def _compute_longitudinal_error(
        x: float, y: float, xr: float, yr: float, psi_ref: float
    ) -> float:
        dx = float(x) - float(xr)
        dy = float(y) - float(yr)
        return math.cos(float(psi_ref)) * dx + math.sin(float(psi_ref)) * dy

    def _check_end_condition(self, ref_point, vx: float, e_longitudinal: float) -> str | None:
        final_ref_idx = len(self.reference_manager.ref.x) - 1
        near_goal = ref_point.idx >= max(final_ref_idx - self.goal_index_margin, 0)
        if near_goal:
            if abs(ref_point.e_y) <= self.goal_lateral_error and \
               abs(e_longitudinal) <= self.goal_longitudinal_error and \
               abs(ref_point.e_psi) <= self.goal_heading_error and \
               abs(float(vx)) <= self.goal_speed_threshold:
                return "goal condition satisfied"

        e_lateral = float(ref_point.e_y)
        if abs(e_lateral) > self.max_lateral_error:
            return f"lateral error exceeded limit: {e_lateral:.3f} m"
        if abs(e_longitudinal) > self.max_longitudinal_error:
            return f"longitudinal error exceeded limit: {e_longitudinal:.3f} m"
        return None

    def _publish_record(
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
        msg = Float32MultiArray()
        record_values = [
            float(meas.stamp_sec),
            float(actual_loop_dt),
            float(control_update_dt),
            float(ref_point.idx),
            float(meas.x),
            float(meas.y),
            float(meas.yaw),
            float(meas.vx),
            float(meas.vy),
            float(meas.yaw_rate),
            float(meas.ax),
            float(ref_point.xr),
            float(ref_point.yr),
            float(ref_point.psi_ref),
            float(ref_point.kappa_ref),
            float(ref_point.v_ref),
            float(e_longitudinal),
            float(ref_point.e_y),
            float(ref_point.e_psi),
            float(steering_exec),
            float(cmd.steering),
            float(cmd.accel_cmd),
            float(cmd.throttle),
            float(cmd.brake),
            float(act_dbg.f_resist),
            float(act_dbg.f_required),
        ]
        msg.layout.dim = [
            MultiArrayDimension(label=name, size=1, stride=1)
            for name in CONTROLLER_RECORD_FIELDS
        ]
        msg.layout.data_offset = 0
        msg.data = record_values
        self.record_pub.publish(msg)


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
