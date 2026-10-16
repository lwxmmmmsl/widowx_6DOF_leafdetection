"""公共路径、配置校验及事件日志。

位姿矩阵约定为 T_parent_child，平移单位为米；本模块不依赖 ROS/Isaac。

Paths and validated shared configuration; no ROS or Isaac imports."""
from pathlib import Path
import json
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[1]
ARM = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

def read_yaml(path):
    with open(path, encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def profile(path):
    """Reject incomplete real camera profiles rather than invent calibration."""
    cfg = read_yaml(path)
    c = cfg['camera']
    k = np.asarray(c['k'], dtype=float).reshape(3, 3)
    t = np.asarray(c['T_base_optical'], dtype=float)
    if c['width'] <= 0 or c['height'] <= 0 or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid CameraInfo dimensions/intrinsics')
    validate_pose(t)
    if c.get('distortion_model', 'plumb_bob') not in ('plumb_bob', 'rational_polynomial', ''):
        raise ValueError('Only OpenCV Brown/rational distortion is supported')
    if not c['frame_id'] or c['frame_id'] == 'wx250s/base_link':
        raise ValueError('Camera optical frame must be distinct from robot base')
    if any(n not in cfg['initial_joints'] for n in ARM):
        raise ValueError('Profile must contain all measured initial arm joints')
    if not np.isfinite(list(cfg['initial_joints'].values())).all():
        raise ValueError('Nonfinite initial joints')
    return cfg

def validate_pose(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Expected a finite 4x4 pose')
    if not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(t[:3,:3]), 1):
        raise ValueError('Invalid rigid transformation')
    return t

def event(name, **fields):
    print(json.dumps({'event': name, **fields}, ensure_ascii=False), flush=True)
