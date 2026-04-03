from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

import os


def generate_launch_description():
    package_share_dir = get_package_share_directory("vehicle_controller")
    controller_config = os.path.join(package_share_dir, "config", "controller.yaml")
    vehicle_params = os.path.join(package_share_dir, "config", "vehicle_params.yaml")
    path_ref_file = os.path.join(package_share_dir, "data", "path_ref.mat")
    accel_map_file = os.path.join(package_share_dir, "data", "Acc_mapData_noSlope.mat")
    brake_map_file = os.path.join(package_share_dir, "data", "brake_mapData_noSlope.mat")

    controller_node = Node(
        package="vehicle_controller",
        executable="vehicle_controller_node",
        name="vehicle_controller",
        output="screen",
        parameters=[
            controller_config,
            vehicle_params,
            {"reference.path_file": path_ref_file},
            {"vehicle.accel_map_file": accel_map_file},
            {"vehicle.brake_map_file": brake_map_file},
        ],
    )

    return LaunchDescription([controller_node])
