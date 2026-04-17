# MATLAB simulation reference

https://github.com/YuxuanCAVE/Controller_GDP

# vehicle_controller

ROS 2 Python vehicle controller package for OxTS-based state input.

Current controller stack:
- combined MPC for steering and acceleration tracking
- lateral MPC or kinematic lateral MPC with longitudinal PID or LQR as alternative runtime modes
- 1D lookup-table-based throttle/brake mapping with linear interpolation
- sensor-position compensation from the OxTS measurement point to the vehicle control point
- dynamic update `dt` for control, integral, and rate-limit logic
- ROS2 record topic for vehicle state, dynamics, and tracking errors
- normalized command output plus controller enable topic

## Topics

Inputs:
- `/ins/odometry` (`nav_msgs/msg/Odometry`)
- `/ins/imu` (`sensor_msgs/msg/Imu`)
- `/sygnal_state` (`sygnal_msgs/msg/State`)
- `/sygnal_fault` (`sygnal_msgs/msg/FaultState`)

Outputs:
- `/command` (`sygnal_msgs/msg/InterfaceCommand`)
- `/enable` (`sygnal_msgs/msg/InterfaceEnable`)
- `/controller_record` (`std_msgs/msg/Float32MultiArray`)

`/command.command` layout:
- `command[0]`: brake in `[0, 1]`
- `command[1]`: throttle in `[0, 1]`
- `command[2]`: steering in `[-1, 1]`
- `command[3:7]`: reserved, currently zero

`/enable.enable` semantics:
- first 3 entries `true`: brake, throttle, and steering channels are enabled
- all 7 entries `false`: the controller is waiting for required inputs or state freshness is not acceptable

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
2. Convert the OxTS position to the vehicle control point using `p_center = P + R * t`.
3. Find the nearest reference point. The first query is global; later queries use a local progress window.
4. Smooth the reference speed in time when speed smoothing is enabled.
5. Compute control with either combined `mpc + mpc` or decoupled lateral + longitudinal control using the actual loop `dt`.
6. Apply steering rate limit.
7. Map the final acceleration command to normalized throttle/brake using 1D actuator lookup tables.
8. Publish command and enable topics.

The ROS runtime is aligned with the current MATLAB closed-loop structure for the active controller
and actuator chain. The controller initializes with `sim.dt = 0.10` and then updates the loop `dt`
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
- `kinematic_mpc.*`
- `lon_pid.*`
- `lon_lqr.*`
- `accel_limits.*`
- `vehicle.*`

## Configuration

Main configuration file:
- `config/controller.yaml`
- `config/vehicle_params.yaml`

Important parameter groups:
- `controller.*`: choose `lateral=mpc, longitudinal=mpc`, `lateral=mpc, longitudinal=lqr`,
  `lateral=mpc, longitudinal=pid`, `lateral=kinematic_mpc, longitudinal=lqr`, or
  `lateral=kinematic_mpc, longitudinal=pid`
- `sim.*`: initial control-loop dt and reference search window
- `ref.*`: reference path location
- `speed.*`: constant speed or profile configuration plus optional reference-speed smoothing
- `mpc_combined.*`: combined controller horizon and Q/R/Rd weights
- `mpc.*`: lateral MPC horizon and weights
- `kinematic_mpc.*`: kinematic lateral MPC horizon and weights
- `lon_pid.*`: longitudinal PID gains and acceleration limits
- `lon_lqr.*`: longitudinal LQR weights and acceleration limits
- `accel_limits.*`: MATLAB-synced acceleration bounds from actuator maps
- `vehicle.*`: vehicle dimensions, steering limits, actuator files, and sensor offset
- `timing.*`: measured `dt` clamp range for update logic
  and the minimum control update period
- `end_condition.*`: controller stop conditions for goal completion and excessive tracking error
- `record_topic`: debug/record topic for ros2 bag logging

The launch file injects these runtime data files automatically:
- `data/path_ref.mat`
- `data/Acc_mapData_noSlope.mat`
- `data/brake_mapData_noSlope.mat`

If you run the node without explicit file overrides, it resolves these data files from the
installed package share directory instead of assuming a source-tree path.

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

The acceleration and brake MAT files are used for 1D longitudinal actuator mapping. The runtime
expects each sample set to include:
- command value (`Acc_Full` or `Break_Full`)
- longitudinal force (`Force_full`)
- vehicle speed (`Vel_Full`)

At load time, the ROS runtime reduces the raw map samples to monotonic 1D lookup tables and then
uses linear interpolation at runtime for both:
- inverse lookup: force -> command
- forward lookup: command -> achieved force

This matches the current MATLAB structure where inverse lookup and forward execution use the same
reduced 1D map.

## State Handling

The controller does not use the raw OxTS sensor position directly as the vehicle tracking point.
Instead, it converts the measured sensor position to the vehicle control point using:

```text
p_center = P + R * t
```

where:
- `P` is the OxTS position from odometry
- `R` is the `3x3` rotation matrix from the odometry quaternion
- `t` is the configured sensor offset in vehicle coordinates

The current default offset is configured in `config/vehicle_params.yaml` as:
- `vehicle.sensor_offset_x = 1.14`
- `vehicle.sensor_offset_y = 0.32`
- `vehicle.sensor_offset_z = 0.0`

After this compensation, the controller computes:
- nearest reference point
- lateral error `e_y`
- heading error `e_psi`
- longitudinal error

all relative to the compensated vehicle control point rather than the raw sensor location.

At startup, the reference match is found globally. Once the controller locks onto the path, it
switches back to local-window tracking using `sim.progress_window`. This allows the runtime to start
correctly even when the vehicle is not physically located near reference index `0`.

## End Condition

The controller now has two explicit stop conditions:
- goal reached: near the final reference index, with small lateral, longitudinal, and heading error,
  and vehicle speed below a threshold
- tracking failure: absolute lateral error or absolute longitudinal error exceeds the configured limit

The default failure thresholds are:
- `end_condition.max_lateral_error = 7.0`
- `end_condition.max_longitudinal_error = 7.0`

When either stop condition is triggered, the node publishes a zero command and sets `/enable` to
`false`.

The default goal thresholds are:
- `end_condition.goal_index_margin = 3`
- `end_condition.goal_lateral_error = 2.0`
- `end_condition.goal_longitudinal_error = 2.0`
- `end_condition.goal_heading_error = 0.0872664626`
- `end_condition.goal_speed_threshold = 2.0`

## Recording

The node now publishes `/controller_record` for logging with ROS2 bags instead of writing CSV files
itself. This keeps the workflow aligned with standard `ros2 bag record` output, which is stored as
`.db3`.

Example:

```bash
ros2 bag record \
  /ins/odometry \
  /ins/imu \
  /ins/nav_sat_fix \
  /ins/nav_sat_ref \
  /sygnal_state \
  /sygnal_fault \
  /command \
  /enable \
  /controller_record \
  /tf \
  /tf_static \
  --output output_path
```

You can also use the helper script:

```bash
bash scripts/record_vehicle_bag.sh cave_runs_record
```

The script treats `output_path` as a base name and appends a timestamp suffix, so repeated runs do
not overwrite earlier recordings. For example, `output_path` becomes
`output_path_20260414_153000`.

To expand `/controller_record` into a CSV with column headers after recording:

```bash
python3 scripts/export_controller_record_csv.py /path/to/bag_dir
```

By default this writes `controller_record.csv` inside the bag directory. You can also override the
destination:

```bash
python3 scripts/export_controller_record_csv.py /path/to/bag_dir --output /tmp/controller_record.csv
```

To estimate a reasonable `sim.progress_window` from the reference path spacing:

```bash
python3 scripts/analyze_progress_window.py data/path_ref.mat --dt 0.10 --speeds 1 3 5 8
```

This reports path spacing statistics and a suggested search-window range for representative speeds.

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
- Published steering is normalized to `[-1, 1]`, with positive values mapped directly from
  positive steering radians.
- Internal longitudinal output is desired acceleration in `m/s^2`.
- Published throttle and brake are normalized to `[0, 1]`.
- Launch defaults come from `config/controller.yaml` and `config/vehicle_params.yaml`.
- The current default runtime configuration is `lateral=kinematic_mpc` with `longitudinal=pid`.
