from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("wodowx_control"),
        "config",
        "pipeline.yaml",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "dry_run",
                default_value="true",
                description="Log planned motions without commanding the physical arm.",
            ),
            DeclareLaunchArgument(
                "start_camera_driver",
                default_value="true",
                description=(
                    "Start the Azure Kinect driver; set false if it is already running."
                ),
            ),
            DeclareLaunchArgument(
                "perception_python",
                default_value="/home/user/anaconda3/envs/env_perception/bin/python",
                description=(
                    "Python interpreter containing Ultralytics and the ROS Python "
                    "bindings; the default is this workspace's env_perception."
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("azure_kinect_ros_driver"),
                            "launch",
                            "driver.launch.py",
                        ]
                    )
                ),
                condition=IfCondition(LaunchConfiguration("start_camera_driver")),
                launch_arguments={
                    "depth_enabled": "true",
                    "color_enabled": "true",
                    "depth_unit": "16UC1",
                    "color_format": "bgra",
                }.items(),
            ),
            Node(
                package="wodowx_control",
                executable="leaf_perception",
                name="leaf_perception",
                output="screen",
                parameters=[config],
                prefix=LaunchConfiguration("perception_python"),
            ),
            Node(
                package="wodowx_control",
                executable="leaf_control",
                name="leaf_control",
                output="screen",
                parameters=[config, {"dry_run": LaunchConfiguration("dry_run")}],
            ),
            Node(
                package="wodowx_control",
                executable="leaf_workflow",
                name="leaf_workflow",
                output="screen",
                parameters=[config],
            ),
        ]
    )
