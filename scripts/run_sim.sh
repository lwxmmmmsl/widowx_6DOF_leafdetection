#!/usr/bin/env bash
# 启动独立 Isaac 进程；清除 ROS Python 路径，并保留完整启动日志。
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
# Humble's Python 3.10 packages must not contaminate Isaac's Python 3.11.
unset PYTHONPATH PYTHONHOME
mkdir -p "$ROOT/artifacts/logs"
WIDOWX_SIM_LOG="$ROOT/artifacts/logs/simulator_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Full simulator output: $WIDOWX_SIM_LOG"
# Keep stdout AND stderr, including Python exceptions missing from Kit's log.
# pipefail preserves Python's failure status through tee.
/home/user/anaconda3/envs/env_isaaclab/bin/python -u -m widowx_codex.simulator --enable_cameras "$@" 2>&1 | tee "$WIDOWX_SIM_LOG"

# Kit 的立即关闭可能把 Python 异常退出码覆盖为 0；以持久化错误事件补充判断。
if grep -q '"event": "simulation_failed"' "$WIDOWX_SIM_LOG"; then
    exit 1
fi
