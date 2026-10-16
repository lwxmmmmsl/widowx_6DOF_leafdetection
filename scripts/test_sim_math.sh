#!/usr/bin/env bash
# Python 3.11：离线验证运动学、通信和 USD 资产；不启动物理仿真。
# CPU-only validation using Isaac's Python/USD libraries; no AppLauncher or GPU.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WIDOWX_ISAAC_ENV=/home/user/anaconda3/envs/env_isaaclab
WIDOWX_USD_LIB=$("$WIDOWX_ISAAC_ENV/bin/python" -c 'import glob,sys; paths=glob.glob(sys.prefix+"/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*/pxr"); assert len(paths)==1, paths; print(paths[0][:-4])')
export PYTHONPATH="$ROOT:$WIDOWX_USD_LIB"
export LD_LIBRARY_PATH="$WIDOWX_ISAAC_ENV/lib:$WIDOWX_USD_LIB/bin:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
exec "$WIDOWX_ISAAC_ENV/bin/python" -m pytest -q tests/test_contracts.py tests/test_plant_asset.py tests/test_initial_state.py tests/test_import_config.py tests/test_pose_tolerance.py "$@"
