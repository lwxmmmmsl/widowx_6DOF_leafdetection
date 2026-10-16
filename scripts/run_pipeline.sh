#!/usr/bin/env bash
# 启动 ROS 仿真桥接、检测与测量节点；域 77 与常用真机域分开。
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/humble/setup.bash
if [[ ! -f "$ROOT/artifacts/ros_install/setup.bash" ]]; then
  echo 'Run scripts/build_ros.sh first.' >&2; exit 1
fi
source "$ROOT/artifacts/ros_install/setup.bash"
# Always isolate simulated frames from the real pipeline, preserving topic names.
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
export WIDOWX_RUN_DIR=${WIDOWX_RUN_DIR:-/tmp/widowx_codex}
export PYTHONPATH="$ROOT:$ROOT/reference/wodowx_control:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="$ROOT/artifacts/ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"
cd "$ROOT"
exec /usr/bin/python3 -m widowx_codex.launch_pipeline "$@"
