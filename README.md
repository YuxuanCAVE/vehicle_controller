# MATLAB simulation reference

https://github.com/YuxuanCAVE/Controller_GDP

# vehicle_controller

ROS 2 Python vehicle controller package for OxTS-based state input.

Current controller stack:
- lateral MPC for path tracking
- longitudinal LQR for speed tracking
- lookup-table-based throttle/brake mapping
- dynamic update `dt` for integral and rate-limit logic
- tracking safety supervisor with multi-stage fallback modes
- normalized command output plus controller enable and safety debug topics

## Topics

Inputs:
- `/ins/odometry` (`nav_msgs/msg/Odometry`)
- `/ins/imu` (`sensor_msgs/msg/Imu`)

Outputs:
- `/command` (`std_msgs/msg/Float32MultiArray`)
- `/enable` (`std_msgs/msg/Bool`)
- `/safety/debug` (`std_msgs/msg/String`)

`/command` layout:
- `data[0]`: brake in `[0, 1]`
- `data[1]`: throttle in `[0, 1]`
- `data[2]`: steering in `[-1, 1]`
- `data[3:7]`: reserved, currently zero

`/enable` semantics:
- `true`: the controller produced and published a valid command this cycle
- `false`: the controller is waiting for required inputs or state freshness is not acceptable

`/safety/debug` payload:
- JSON string with `mode`, `mode_code`, `lateral_error`, `heading_error`, `speed_mps`
- includes persistence timers, current safety target speed, and `override_active`

## Controller flow

The runtime path is:

1. Read odometry and IMU from OxTS topics.
2. Build measured state and reference point on the path.
3. Compute raw lateral steering and longitudinal acceleration.
4. Apply delay and steering rate limit.
5. Run the tracking safety supervisor.
6. Map the final acceleration command to normalized throttle/brake.
7. Publish command, enable, and safety debug topics.

The lateral MPC and longitudinal LQR still use a fixed model `control_dt`.
The integral update and steering rate-limit update use measured timestamp gaps with clamping:
- `timing.dt_min`
- `timing.dt_max`

This keeps the controller model fixed while making update logic more robust to real message jitter.

## Safety supervisor

The tracking safety supervisor sits after the raw controller output and before final actuator command publication.

Monitored signals:
- lateral tracking error
- heading error
- vehicle speed
- persistence time of safety threshold violations

Modes:
- `NORMAL`: no safety override
- `DEGRADED`: reduce target speed, positive acceleration, throttle, and steering rate
- `PROTECTIVE_BRAKE`: force throttle to zero, apply moderate braking, keep steering smooth
- `CONTROLLED_STOP`: perform a controlled stop, hold brake at low speed, and avoid aggressive steering

Each mode has YAML-configurable:
- enter thresholds
- exit thresholds
- enter persistence time
- exit persistence time
- command limits

Hysteresis is implemented through separate enter and exit thresholds plus exit persistence timers.

## Configuration

Main configuration file:
- [config/controller.yaml](/f:/vehicle_controller/config/controller.yaml)

Important parameter groups:
- `reference.*`: path file, speed mode, constant speed, optional speed profile
- `lateral_mpc.*`: horizon and weights
- `longitudinal_lqr.*`: LQR weights, acceleration bounds, integral windup bounds
- `timing.*`: measured `dt` clamp range for update logic
- `safety.degraded.*`
- `safety.protective_brake.*`
- `safety.controlled_stop.*`

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

The acceleration and brake MAT files are used for lookup-table-based longitudinal actuator mapping.

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
ros2 topic echo /safety/debug
```

## Notes

- Internal steering is kept in radians inside the controller.
- Published steering is normalized to `[-1, 1]`.
- Internal longitudinal output is desired acceleration in `m/s^2`.
- Published throttle and brake are normalized to `[0, 1]`.
- Launch defaults come from `config/controller.yaml` and `config/vehicle_params.yaml`.
