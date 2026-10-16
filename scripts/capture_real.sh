#!/usr/bin/env bash
# 只读导出真机相机/关节参数，沿用当前终端的真实 ROS 域。
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/humble/setup.bash
cd "$ROOT"
exec /usr/bin/python3 -m widowx_codex.capture_real "$@"
