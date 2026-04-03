# vehicle_controller

ROS 2 Python vehicle controller package for OxTS-based state input.

This package currently implements:
- lateral MPC
- longitudinal LQR
- lookup-table-based longitudinal actuator mapping
- normalized command output

## Inputs

- `/ins/odometry` (`nav_msgs/msg/Odometry`)
- `/ins/imu` (`sensor_msgs/msg/Imu`)

## Output

- `/control/command` (`std_msgs/msg/Float32MultiArray`)

Command layout:
- `data[0]`: brake in `[0, 1]`
- `data[1]`: throttle in `[0, 1]`
- `data[2]`: steering in `[-1, 1]`
- `data[3:7]`: reserved, currently zero

## Data files

The package expects these files in `data/`:
- `path_ref.mat`
- `Acc_mapData_noSlope.mat`
- `brake_mapData_noSlope.mat`

`path_ref.mat` must contain:
- `x_opt`
- `y_opt`

The actuator map files are used for lookup-table-based throttle/brake mapping.

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

## Notes

- Internal steering is kept in radians inside the controller.
- Published steering is normalized to `[-1, 1]`.
- Internal longitudinal output is desired acceleration in `m/s^2`.
- Published throttle and brake are normalized to `[0, 1]`.
# vehicle_controller
