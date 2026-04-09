import math
from typing import Optional

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from vehicle_controller.types import ControllerMemory, MeasuredState


class StateAdapter:
    def __init__(self, max_state_age_sec: float = 0.2):
        self.max_state_age_sec = max_state_age_sec
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
        yaw = self._yaw_from_quaternion(quat.x, quat.y, quat.z, quat.w)

        delta_prev = 0.0
        if memory is not None:
            delta_prev = memory.last_steering_rad

        stamp_sec = max(
            self._stamp_to_sec(odom.header.stamp),
            self._stamp_to_sec(imu.header.stamp),
        )

        return MeasuredState(
            x=odom.pose.pose.position.x,
            y=odom.pose.pose.position.y,
            yaw=yaw,
            vx=odom.twist.twist.linear.x,
            vy=odom.twist.twist.linear.y,
            yaw_rate=imu.angular_velocity.z,
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
