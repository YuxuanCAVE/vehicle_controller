# MATLAB simulation reference

https://github.com/YuxuanCAVE/Controller_GDP

# vehicle_controller

ROS 2 Python vehicle controller package for OxTS-based state input.

Current controller stack:
- lateral KBM-based NMPC for steering tracking
- longitudinal PID for speed tracking
- 1D lookup-table-based throttle/brake mapping with linear interpolation
- sensor-position compensation from the OxTS measurement point to the vehicle control point
- dynamic update `dt` for control, integral, and rate-limit logic
- ROS2 record topic for vehicle state, dynamics, and tracking errors
- DBW command output plus controller enable topic

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
- `command[0]`: brake in `[0, 0.6]`
- `command[1]`: throttle in `[0, 0.6]`
- `command[2]`: steering command in radians, limited to `[-vehicle.max_steer, vehicle.max_steer]`
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
- `data[19:24]`: `steering_rad`, `steering_command`, `accel_cmd`, `throttle`, `brake`
- `data[24:26]`: `f_resist`, `f_required`

## Controller flow

The runtime path is:

1. Read odometry and IMU from OxTS topics.
2. Convert the OxTS position to the vehicle control point using `p_center = P + R * t`.
3. Convert the OxTS linear velocity to the controller point using rigid-body compensation.
4. Find the nearest reference point. The first query is global; later queries use a local progress window.
   The current implementation matches MATLAB by selecting the nearest waypoint index directly.
5. Smooth the reference speed in time when speed smoothing is enabled.
6. Compute steering with lateral NMPC-KBM and acceleration with longitudinal PID.
7. Apply steering angle and steering-rate limits.
8. Map the final acceleration command to throttle/brake using 1D actuator lookup tables.
   The published throttle and brake are the actual DBW pedal requests, capped by
   `vehicle.max_pedal_publish`, without a second normalization step.
9. Publish command and enable topics.

The ROS runtime is aligned with the current MATLAB controller and actuator chain, but is used for
real-vehicle topic I/O rather than closed-loop simulation. The controller initializes with
`sim.dt = 0.10` for the first control step, then updates the loop `dt` from the actual control
callback timing from the second step onward.
The MPC, integral update, and steering rate-limit logic all use this floating loop `dt`,
with clamping:
- `timing.dt_min`
- `timing.dt_max`

This keeps the first step well-defined while letting the controller follow actual ROS timing.

The ROS runtime parameters are now aligned with the current MATLAB config naming:
- `controller.*`
- `sim.*`
- `ref.*`
- `speed.*`
- `nmpc_kbm.*`
- `lon_pid.*`
- `accel_limits.*`
- `vehicle.*`

## Configuration

Main configuration file:
- `config/controller.yaml`
- `config/vehicle_params.yaml`

Important parameter groups:
- `controller.*`: current supported stack is `lateral=nmpc_kbm` with `longitudinal=pid`
- `sim.*`: initial control-loop dt and reference search window
- `ref.*`: reference path location
- `speed.*`: constant speed or profile configuration plus optional reference-speed smoothing
- `nmpc_kbm.*`: lateral NMPC horizon and weights
- `lon_pid.*`: longitudinal PID gains and acceleration limits
- `accel_limits.*`: MATLAB-synced acceleration bounds from actuator maps
- `vehicle.*`: vehicle dimensions, steering limits, actuator files, and sensor offset
- `vehicle.steering_sign`: final published steering sign
  and the current configuration uses `-1.0`, which means positive controller steering angle is
  flipped before publish while remaining in radians
- `vehicle.max_pedal_publish`: DBW pedal ceiling, currently `0.60`
- `timing.*`: measured `dt` clamp range for update logic
  and optional startup warmup time before commands are enabled
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

The current nearest-point logic follows the MATLAB implementation style:
- choose the nearest waypoint index directly
- use that waypoint as the reference point
- compute the reference heading from neighboring waypoints

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

The node publishes `/controller_record` for logging, and the launch file now starts bag recording
automatically by default. Each run creates a new timestamped bag directory, so previous experiments
are not overwritten.

Default behavior:

```bash
ros2 launch vehicle_controller vehicle_controller.launch.py
```

This will:
- start the controller node
- start `ros2 bag record`
- save the bag under `bags/vehicle_controller_YYYYMMDD_HHMMSS` when launched from the source tree
- otherwise save under `<current_working_directory>/bags/vehicle_controller_YYYYMMDD_HHMMSS`
- export `/controller_record` to `controller_record.csv` inside that same bag directory after the
  recording process stops

You can disable automatic recording if needed:

```bash
ros2 launch vehicle_controller vehicle_controller.launch.py record:=false
```

You can also override the output root:

```bash
ros2 launch vehicle_controller vehicle_controller.launch.py record_root:=/path/to/bags
```

If you want to export a bag manually, the helper is still available:

```bash
python3 scripts/export_controller_record_csv.py /path/to/bag_dir
```

By default this writes `controller_record.csv` inside the bag directory. You can also override the
destination:

```bash
python3 scripts/export_controller_record_csv.py /path/to/bag_dir --output /tmp/controller_record.csv
```

To find the latest experiment quickly from the source tree:

```bash
ls -dt bags/vehicle_controller_* | head -n 1
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

If you only changed YAML or launch parameters, you do not need to rebuild. Restart the launch is
enough. Rebuild is only needed after Python package layout changes or install-space data changes.

## Run

```bash
ros2 launch vehicle_controller vehicle_controller.launch.py
```

Run with a custom bag output root:

```bash
ros2 launch vehicle_controller vehicle_controller.launch.py record_root:=/home/yuxuan/test_bags
```

Run without automatic bag recording:

```bash
ros2 launch vehicle_controller vehicle_controller.launch.py record:=false
```

Run after only changing YAML:

```bash
cd ~/ros_ws
source install/setup.bash
ros2 launch vehicle_controller vehicle_controller.launch.py
```

## Quick checks

Check that the node is alive:

```bash
ros2 node list | grep vehicle_controller
```

Check command output:

```bash
ros2 topic echo /command
```

Check controller record stream:

```bash
ros2 topic echo /controller_record
```

Check odometry rate:

```bash
ros2 topic hz /ins/odometry
```

Check IMU rate:

```bash
ros2 topic hz /ins/imu
```

Foxglove is a good choice for live monitoring during experiments. The most useful topics to watch
live are:
- `/ins/odometry`
- `/ins/imu`
- `/command`
- `/enable`
- `/controller_record`

```bash
ros2 topic echo /command
ros2 topic echo /enable
```

## Notes

- Internal steering is kept in radians inside the controller.
- Published steering is also in radians, with `vehicle.steering_sign` applied at publish time.
- Internal longitudinal output is desired acceleration in `m/s^2`.
- Published throttle and brake are normalized to `[0, 1]`.
- Launch defaults come from `config/controller.yaml` and `config/vehicle_params.yaml`.
- The current default runtime configuration is `lateral=kinematic_mpc` with `longitudinal=pid`.
