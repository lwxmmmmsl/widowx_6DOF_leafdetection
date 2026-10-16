#!/usr/bin/env bash
# 只在本工程 artifacts 下编译自定义 ROS 消息，不改动真机工作区。
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/humble/setup.bash
cd "$ROOT"
colcon --log-base artifacts/colcon_log build --base-paths reference/wodowx_interfaces --build-base artifacts/ros_build --install-base artifacts/ros_install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
