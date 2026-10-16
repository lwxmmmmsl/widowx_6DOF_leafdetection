"""仿真目标姿态调整：基座 YZ 法向投影和有界、固定姿态的 IK 搜索。"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .common import validate_pose


def project_normal_yz(normal):
    """只能传入基座坐标系法向；退化投影拒绝，避免人为选取错误朝向。"""
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError('Invalid base-frame leaf normal')
    projected = np.array([0., normal[1], normal[2]])
    length = np.linalg.norm(projected)
    if length < 1e-6:
        raise ValueError('Leaf normal YZ projection is degenerate (normal parallel to base X)')
    return projected / length


def resolve_leaf_pose(model, start_q, approach, contact, tolerance_deg=15., step_deg=5., deadline=None):
    """原姿态优先，随后按角度递增绕工具三轴尝试正负旋转。

    角度为 SO(3) 总旋转角，不是各欧拉角分别的容差。位置不偏移；每个候选
    同时检查接近、插入及退出路径。三段采用同一个姿态，不在每个路点单独放宽。
    有限采样失败不代表容差范围内不存在数学解。此函数只规划，不执行运动。
    """
    approach, contact = validate_pose(approach).copy(), validate_pose(contact).copy()
    tolerance_deg, step_deg = float(tolerance_deg), float(step_deg)
    if not (np.isfinite(tolerance_deg) and 0 <= tolerance_deg <= 45):
        raise ValueError('orientation_tolerance_deg must be finite and in [0, 45]')
    if not (np.isfinite(step_deg) and step_deg > 0):
        raise ValueError('orientation_search_step_deg must be finite and positive')
    if tolerance_deg and int(np.ceil(tolerance_deg / step_deg)) > 9:
        raise ValueError('Orientation search allows at most 9 angle levels')
    if not np.allclose(approach[:3,:3], contact[:3,:3], atol=1e-8):
        raise ValueError('Approach and contact must share one nominal orientation')
    nominal = approach[:3,:3].copy()
    rotations = [(0., nominal)]
    levels = list(np.arange(step_deg, tolerance_deg, step_deg)) + ([tolerance_deg] if tolerance_deg else [])
    for angle in levels:
        for axis in np.eye(3):
            for sign in (-1, 1):
                rotations.append((angle, nominal @ Rotation.from_rotvec(axis * np.deg2rad(sign * angle)).as_matrix()))
    attempts = []
    for angle, orientation in rotations:
        if deadline is not None and time.monotonic() >= deadline:
            return dict(reachable=False, reason='orientation_search_timeout', attempts=attempts)
        a, c = approach.copy(), contact.copy()
        a[:3,:3] = c[:3,:3] = orientation
        row = dict(sample_angle_deg=float(angle), orientation_matrix=orientation.tolist())
        attempts.append(row)
        stage = 'approach_ik'
        try:
            path = model.plan(start_q, a)
            stage = 'insertion_ik'
            insertion = model.plan(path[-1], c, True, a)
            stage = 'retreat_ik'
            retreat = model.plan(insertion[-1], a, True, c)
            # 显式验证 FK 残差：角度放宽不能带来额外的位置漂移。
            max_position, max_angle = 0., 0.
            for points, begin, end in ((path[-1:], a, a), (insertion, a, c), (retreat, c, a)):
                for q, u in zip(points, np.linspace(0, 1, len(points))):
                    actual = model.fk(q)
                    position = (1-u)*begin[:3,3] + u*end[:3,3]
                    max_position = max(max_position, float(np.linalg.norm(actual[:3,3]-position)))
                    max_angle = max(max_angle, float(Rotation.from_matrix(nominal.T @ actual[:3,:3]).magnitude()))
            row.update(max_position_error_m=max_position, max_orientation_error_deg=float(np.rad2deg(max_angle)))
            # 0.001 rad 为原数值 IK 的角度残差；默认 1 mm 位置精度不放宽。
            if max_position > .001 or max_angle > np.deg2rad(tolerance_deg) + .001:
                row.update(reachable=False, failure_stage='fk_residual')
                continue
            row['reachable'] = True
            return dict(reachable=True, approach_pose=a.tolist(), contact_pose=c.tolist(),
                        orientation_error_deg=float(angle), attempts=attempts,
                        tolerance_deg=tolerance_deg)
        except RuntimeError as exc:
            row.update(reachable=False, failure_stage=stage, error=str(exc))
    return dict(reachable=False, reason='orientation_samples_exhausted', attempts=attempts,
                tolerance_deg=tolerance_deg)
