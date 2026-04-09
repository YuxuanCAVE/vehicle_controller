# MATLAB simulation reference

https://github.com/YuxuanCAVE/Controller_GDP

# vehicle_controller

ROS 2 Python vehicle controller package for OxTS-based state input.

Current controller stack:
- combined MPC for steering and acceleration tracking
- lateral MPC + longitudinal LQR as an alternative runtime mode
- lookup-table-based throttle/brake mapping
- dynamic update `dt` for integral and rate-limit logic
- ROS2 record topic for vehicle state, dynamics, and tracking errors
- normalized command output plus controller enable topic

## Topics

Inputs:
- `/ins/odometry` (`nav_msgs/msg/Odometry`)
- `/ins/imu` (`sensor_msgs/msg/Imu`)

Outputs:
- `/command` (`std_msgs/msg/Float32MultiArray`)
- `/enable` (`std_msgs/msg/Bool`)
- `/controller_record` (`std_msgs/msg/Float32MultiArray`)

`/command` layout:
- `data[0]`: brake in `[0, 1]`
- `data[1]`: throttle in `[0, 1]`
- `data[2]`: steering in `[-1, 1]`
- `data[3:7]`: reserved, currently zero

`/enable` semantics:
- `true`: the controller produced and published a valid command this cycle
- `false`: the controller is waiting for required inputs or state freshness is not acceptable

`/controller_record` layout:
- `data[0]`: `stamp_sec`
- `data[1]`: `actual_loop_dt`
- `data[2]`: `control_update_dt`
- `data[3]`: `ref_idx`
- `data[4:7]`: `x`, `y`, `yaw`
- `data[7:11]`: `vx`, `vy`, `yaw_rate`, `ax`
- `data[11:16]`: `xr`, `yr`, `psi_ref`, `kappa_ref`, `v_ref`
- `data[16:19]`: `e_longitudinal`, `e_lateral`, `e_heading`
- `data[19:24]`: `steering_rad`, `steering_norm`, `accel_cmd`, `throttle`, `brake`
- `data[24:26]`: `f_resist`, `f_required`

## Controller flow

The runtime path is:

1. Read odometry and IMU from OxTS topics.
2. Build measured state and reference point on the path.
3. Compute control with either combined `mpc + mpc` or decoupled `mpc + lqr` using the actual loop `dt`.
4. Apply steering rate limit.
5. Map the final acceleration command to normalized throttle/brake.
6. Publish command and enable topics.

The ROS runtime now follows the MATLAB `mpc_combined` structure directly.
The controller initializes with `sim.dt = 0.10` and then updates the loop `dt`
from the actual control callback timing.
The MPC, integral update, and steering rate-limit logic all use this floating loop `dt`,
with clamping:
- `timing.dt_min`
- `timing.dt_max`

This keeps the first step well-defined while letting the controller follow actual ROS timing.
To avoid running the control law faster than intended, the node also enforces
`timing.min_control_period`. If callbacks arrive faster than that period, the node
skips command publication and only recomputes and publishes a new command when the minimum
period elapses.

The ROS runtime parameters are now aligned with the newer MATLAB config naming:
- `controller.*`
- `sim.*`
- `ref.*`
- `speed.*`
- `mpc_combined.*`
- `mpc.*`
- `lon_lqr.*`
- `accel_limits.*`

## Configuration

Main configuration file:
- [config/controller.yaml](/f:/vehicle_controller/config/controller.yaml)

Important parameter groups:
- `controller.*`: choose `lateral=mpc, longitudinal=mpc` or `lateral=mpc, longitudinal=lqr`
- `sim.*`: initial control-loop dt and progress window
- `ref.*`: reference path location
- `speed.*`: constant speed or profile configuration
- `mpc_combined.*`: combined controller horizon and Q/R/Rd weights
- `mpc.*`: lateral MPC horizon and weights
- `lon_lqr.*`: longitudinal LQR weights and acceleration limits
- `accel_limits.*`: MATLAB-synced acceleration bounds from actuator maps
- `timing.*`: measured `dt` clamp range for update logic
  and the minimum control update period
- `end_condition.*`: controller stop conditions for goal completion and excessive tracking error
- `record_topic`: debug/record topic for ros2 bag logging

The launch file injects these runtime data files automatically:
- `data/path_ref.mat`
- `data/Acc_mapData_noSlope.mat`
- `data/brake_mapData_noSlope.mat`

## Data file requirements

Required files in `data/`:
- `path_ref.mat`
- `Acc_mapData_noSlope.mat`
- `brake_mapData_noSlope.mat`

`path_ref.mat` must contain:
- `x_opt`
- `y_opt`

If a speed profile is used, the configured speed profile MAT file must contain:
- `pathv_ref`

Speed profile selection supports:
- `speed.mode: "constant"` with `speed.constant_value`
- `speed.mode: "profile"` with `speed.profile`

For the current packaged data, use:
- `speed.profile: "referencePath_Velocity_peak_velocity_3"`

Numeric shorthand is also accepted for `speed.profile`, for example:
- `"3"`
- `"4"`
- `"5"`
- `"7"`

The acceleration and brake MAT files are used for lookup-table-based longitudinal actuator mapping.

## End Condition

The controller now has two explicit stop conditions:
- goal reached: near the final reference index, with small lateral, longitudinal, and heading error,
  and vehicle speed below a threshold
- tracking failure: absolute lateral error or absolute longitudinal error exceeds the configured limit

The default failure thresholds are:
- `end_condition.max_lateral_error = 5.0`
- `end_condition.max_longitudinal_error = 5.0`

When either stop condition is triggered, the node publishes a zero command and sets `/enable` to
`false`.

## Recording

The node now publishes `/controller_record` for logging with ROS2 bags instead of writing CSV files
itself. This keeps the workflow aligned with standard `ros2 bag record` output, which is stored as
`.db3`.

Example:

```bash
ros2 bag record /ins/odometry /ins/imu /command /enable /controller_record
```

## Build

```bash
cd ~/ros_ws
colcon build --packages-select vehicle_controller
source install/setup.bash
```

## Run

```bash
ros2 launch vehicle_controller vehicle_controller.launch.py
```

## Quick checks

```bash
ros2 topic echo /command
ros2 topic echo /enable
```

## Notes

- Internal steering is kept in radians inside the controller.
- Published steering is normalized to `[-1, 1]`.
- Internal longitudinal output is desired acceleration in `m/s^2`.
- Published throttle and brake are normalized to `[0, 1]`.
- Launch defaults come from `config/controller.yaml` and `config/vehicle_params.yaml`.
