from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import os
from datetime import datetime
from pathlib import Path


def _resolve_package_root(package_share_dir: Path) -> Path:
    launch_package_root = Path(__file__).resolve().parents[1]
    if (launch_package_root / "vehicle_controller").is_dir():
        return launch_package_root
    return package_share_dir


def _default_record_root(package_root: Path) -> Path:
    if (package_root / "vehicle_controller").is_dir():
        return package_root / "bags"
    return Path.cwd() / "bags"


def _build_record_actions(context, package_root: Path, default_record_root: Path):
    record_enabled = LaunchConfiguration("record").perform(context).strip().lower()
    if record_enabled not in {"1", "true", "yes", "on"}:
        return []

    record_root_value = LaunchConfiguration("record_root").perform(context).strip()
    if record_root_value:
        record_root = Path(record_root_value).expanduser()
    else:
        record_root = default_record_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_dir = record_root / f"vehicle_controller_{timestamp}"
    suffix = 1
    while bag_dir.exists():
        bag_dir = record_root / f"vehicle_controller_{timestamp}_{suffix}"
        suffix += 1

    bag_dir.parent.mkdir(parents=True, exist_ok=True)
    export_script = package_root / "scripts" / "export_controller_record_csv.py"

    record_process = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "/ins/odometry",
            "/ins/imu",
            "/ins/nav_sat_fix",
            "/ins/nav_sat_ref",
            "/sygnal_state",
            "/sygnal_fault",
            "/command",
            "/enable",
            "/controller_record",
            "/tf",
            "/tf_static",
            "--output",
            str(bag_dir),
        ],
        output="screen",
        sigterm_timeout="5",
        sigkill_timeout="5",
    )

    return [
        LogInfo(msg=f"[vehicle_controller] Recording rosbag to: {bag_dir}"),
        record_process,
        RegisterEventHandler(
            OnProcessExit(
                target_action=record_process,
                on_exit=[
                    LogInfo(msg=f"[vehicle_controller] Exporting controller_record.csv to: {bag_dir}"),
                    ExecuteProcess(
                        cmd=[
                            "python3",
                            str(export_script),
                            str(bag_dir),
                        ],
                        output="screen",
                    ),
                ],
            )
        ),
    ]


def generate_launch_description():
    package_share_dir = get_package_share_directory("vehicle_controller")
    package_share_path = Path(package_share_dir)
    package_root = _resolve_package_root(package_share_path)
    default_record_root = _default_record_root(package_root)
    path_ref_file = os.path.join(package_share_dir, "data", "path_ref.mat")
    accel_map_file = os.path.join(package_share_dir, "data", "Acc_mapData_noSlope.mat")
    brake_map_file = os.path.join(package_share_dir, "data", "brake_mapData_noSlope.mat")

    source_config_dir = package_root / "config"
    install_config_dir = package_share_path / "config"

    controller_config_path = source_config_dir / "controller.yaml"
    vehicle_params_path = source_config_dir / "vehicle_params.yaml"

    if not controller_config_path.exists():
        controller_config_path = install_config_dir / "controller.yaml"
    if not vehicle_params_path.exists():
        vehicle_params_path = install_config_dir / "vehicle_params.yaml"

    controller_config = str(controller_config_path)
    vehicle_params = str(vehicle_params_path)

    controller_node = Node(
        package="vehicle_controller",
        executable="vehicle_controller_node",
        name="vehicle_controller",
        output="screen",
        parameters=[
            controller_config,
            vehicle_params,
            {"ref.path_file": path_ref_file},
            {"vehicle.accel_map_file": accel_map_file},
            {"vehicle.brake_map_file": brake_map_file},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "record",
                default_value="true",
                description="Automatically start rosbag recording with a unique timestamped output directory.",
            ),
            DeclareLaunchArgument(
                "record_root",
                default_value=str(default_record_root),
                description="Directory used to store timestamped rosbag recordings.",
            ),
            controller_node,
            OpaqueFunction(
                function=lambda context: _build_record_actions(
                    context,
                    package_root,
                    default_record_root,
                )
            ),
        ]
    )
