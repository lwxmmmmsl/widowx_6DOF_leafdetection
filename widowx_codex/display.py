"""ROS 图像解码和面板文件协议；不依赖 ROS/Isaac，便于离线测试。"""
import json
import time
from pathlib import Path
import cv2
import numpy as np


def atomic_json(path, data):
    """每个文件只有一个写入者，原子替换避免 UI 读到半条状态。"""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False))
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def image_to_rgba(message):
    """按 encoding 和 step 解码 ROS Image，正确处理带行填充的图像。"""
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(message.encoding)
    if channels is None:
        raise ValueError(f'Unsupported debug image encoding: {message.encoding}')
    h, w, step = int(message.height), int(message.width), int(message.step)
    if min(h, w) <= 0 or step < w*channels or len(message.data) != h*step:
        raise ValueError('Malformed debug image dimensions/step/buffer')
    pixels = np.frombuffer(message.data, np.uint8).reshape(h, step)
    pixels = pixels[:, :w*channels].reshape(h, w, channels)
    codes = {'rgb8': cv2.COLOR_RGB2RGBA, 'bgr8': cv2.COLOR_BGR2RGBA,
             'bgra8': cv2.COLOR_BGRA2RGBA}
    return np.ascontiguousarray(pixels) if message.encoding == 'rgba8' else cv2.cvtColor(pixels, codes[message.encoding])


def panel_status(directory):
    """使用墙钟心跳判断 ROS bridge 是否在线，不把旧文件当成在线状态。"""
    state = read_json(Path(directory)/'panel_status.json')
    if time.time()-state.get('heartbeat', 0) > 4:
        return 'ROS pipeline offline: run ./scripts/run_pipeline.sh (or --manual --dry-run)'
    workflow = state.get('workflow', {})
    production = state.get('production', {})
    return (f"Detect service: {'ready' if state.get('detect_ready') else 'loading/offline'} | "
            f"{state.get('preview', '')}\n"
            f"Production detection (conf=0.20): valid 3D leaves={production.get('valid_leaves', 'not run')}, "
            f"sequence={production.get('sequence', '-')} | Preview boxes do NOT command motion.\n"
            f"Workflow: {workflow.get('message', 'waiting for workflow / manual mode')} | "
            f"planned={workflow.get('planned_targets', 0)}, measured={workflow.get('measured_targets', 0)}")
