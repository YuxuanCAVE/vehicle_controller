import math
from typing import Optional

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from vehicle_controller.types import ControllerMemory, MeasuredState


class StateAdapter:
    """Adapt OxTS odometry/IMU messages to the controller state.

    Frame assumptions used by this adapter:

    - ``odom.pose.pose`` is the sensor pose in the world/LRF frame.
    - ``odom.twist.twist.linear`` is treated as linear velocity expressed in the
      odometry child frame (``oxts_link``), which we use as the sensor/body frame
      for planar control.
    - ``imu.angular_velocity`` is treated as angular velocity in the IMU/body
      frame. For planar control we use ``imu.angular_velocity.z`` as yaw rate.
    - The configured sensor offset ``(x, y, z)`` is the rigid-body transform from
      the sensor origin to the controller tracking point, expressed in the same
      body frame convention as the velocity components.

    Under these assumptions:

    - position is compensated using ``p_control = p_sensor + R * offset``
    - linear velocity is compensated using rigid-body kinematics
      ``v_control = v_sensor + omega x offset``
    """

    def __init__(
        self,
        max_state_age_sec: float = 0.2,
        sensor_offset_x: float = 0.0,
        sensor_offset_y: float = 0.0,
        sensor_offset_z: float = 0.0,
    ):
        self.max_state_age_sec = max_state_age_sec
        self.sensor_offset_x = float(sensor_offset_x)
        self.sensor_offset_y = float(sensor_offset_y)
        self.sensor_offset_z = float(sensor_offset_z)
        self._last_odom: Optional[Odometry] = None
        self._last_imu: Optional[Imu] = None

    def update_odometry(self, msg: Odometry) -> None:
        self._last_odom = msg

    def update_imu(self, msg: Imu) -> None:
        self._last_imu = msg

    def is_ready(self) -> bool:
        return self._last_odom is not None and self._last_imu is not None

    def has_fresh_data(self, now_sec: float) -> bool:
        if not self.is_ready():
            return False

        odom_age = now_sec - self._stamp_to_sec(self._last_odom.header.stamp)
        imu_age = now_sec - self._stamp_to_sec(self._last_imu.header.stamp)
        return odom_age <= self.max_state_age_sec and imu_age <= self.max_state_age_sec

    def build_measured_state(self, memory: Optional[ControllerMemory] = None) -> MeasuredState:
        if not self.is_ready():
            raise RuntimeError("StateAdapter requires both odometry and imu before building state.")

        odom = self._last_odom
        imu = self._last_imu

        quat = odom.pose.pose.orientation
        rot = self._rotation_matrix_from_quaternion(quat.x, quat.y, quat.z, quat.w)
        yaw = self._yaw_from_quaternion(quat.x, quat.y, quat.z, quat.w)
        x_center, y_center, _ = self._transform_sensor_to_center(
            position_x=float(odom.pose.pose.position.x),
            position_y=float(odom.pose.pose.position.y),
            position_z=float(odom.pose.pose.position.z),
            rotation_matrix=rot,
        )
        yaw_rate = float(imu.angular_velocity.z)
        vx_center, vy_center = self._compensate_linear_velocity_to_control_point(
            vx_sensor=float(odom.twist.twist.linear.x),
            vy_sensor=float(odom.twist.twist.linear.y),
            yaw_rate=yaw_rate,
        )

        delta_prev = 0.0
        if memory is not None:
            delta_prev = memory.last_steering_rad

        stamp_sec = max(
            self._stamp_to_sec(odom.header.stamp),
            self._stamp_to_sec(imu.header.stamp),
        )

        return MeasuredState(
            x=x_center,
            y=y_center,
            yaw=yaw,
            vx=vx_center,
            vy=vy_center,
            yaw_rate=yaw_rate,
            ax=imu.linear_acceleration.x,
            delta_prev=delta_prev,
            stamp_sec=stamp_sec,
        )

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _transform_sensor_to_center(
        self,
        position_x: float,
        position_y: float,
        position_z: float,
        rotation_matrix: tuple[tuple[float, float, float], ...],
    ) -> tuple[float, float, float]:
        tx = self.sensor_offset_x
        ty = self.sensor_offset_y
        tz = self.sensor_offset_z

        dx = rotation_matrix[0][0] * tx + rotation_matrix[0][1] * ty + rotation_matrix[0][2] * tz
        dy = rotation_matrix[1][0] * tx + rotation_matrix[1][1] * ty + rotation_matrix[1][2] * tz
        dz = rotation_matrix[2][0] * tx + rotation_matrix[2][1] * ty + rotation_matrix[2][2] * tz
        return position_x + dx, position_y + dy, position_z + dz

    def _compensate_linear_velocity_to_control_point(
        self,
        vx_sensor: float,
        vy_sensor: float,
        yaw_rate: float,
    ) -> tuple[float, float]:
        """Shift body-frame linear velocity from the sensor origin to the control point.

        For planar rigid-body motion:

        ``v_point = v_sensor + omega x r``

        with:
        - ``omega = [0, 0, yaw_rate]``
        - ``r = [sensor_offset_x, sensor_offset_y, 0]``

        which yields:

        - ``vx_point = vx_sensor - yaw_rate * sensor_offset_y``
        - ``vy_point = vy_sensor + yaw_rate * sensor_offset_x``
        """

        vx_point = float(vx_sensor) - float(yaw_rate) * self.sensor_offset_y
        vy_point = float(vy_sensor) + float(yaw_rate) * self.sensor_offset_x
        return vx_point, vy_point

    @staticmethod
    def _rotation_matrix_from_quaternion(
        x: float, y: float, z: float, w: float
    ) -> tuple[tuple[float, float, float], ...]:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return (
            (
                1.0 - 2.0 * (yy + zz),
                2.0 * (xy - wz),
                2.0 * (xz + wy),
            ),
            (
                2.0 * (xy + wz),
                1.0 - 2.0 * (xx + zz),
                2.0 * (yz - wx),
            ),
            (
                2.0 * (xz - wy),
                2.0 * (yz + wx),
                1.0 - 2.0 * (xx + yy),
            ),
        )
