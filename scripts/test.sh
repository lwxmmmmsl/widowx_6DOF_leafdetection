#!/usr/bin/env bash
# Python 3.10：验证原工程算法、ROS 消息/TF 和仿真适配层。
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/humble/setup.bash
source "$ROOT/artifacts/ros_install/setup.bash"
# 测试话题与正在运行的仿真管线（77）隔离，避免 RGB-D 帧串入测试。
export ROS_DOMAIN_ID=${WIDOWX_TEST_ROS_DOMAIN_ID:-179} ROS_LOCALHOST_ONLY=1
export PYTHONPATH="$ROOT:$ROOT/reference/wodowx_control:${PYTHONPATH:-}"
cd "$ROOT"
exec /usr/bin/python3 -m pytest -q --ignore=tests/test_plant_asset.py tests reference/wodowx_control/test "$@"
