"""WX250S 工具轴：+X 沿夹指伸出，Y 为双指开合轴。"""
from dataclasses import replace
import numpy as np


def align_gripper_with_insertion(target):
    """保留接触点和基座侧接近点，将工具 +X 对齐直线伸入方向。"""
    direction = target.contact_position - target.approach_position
    length = np.linalg.norm(direction)
    if length < 1e-8:
        raise ValueError('Approach and contact must be distinct')
    tool_x = direction / length
    tool_y = target.orientation[:, 1]
    if abs(np.dot(tool_x, tool_y)) > 1e-6:
        raise ValueError('Insertion must lie in the leaf plane')
    tool_z = np.cross(tool_x, tool_y)
    tool_z /= np.linalg.norm(tool_z)
    return replace(target, orientation=np.column_stack((tool_x, tool_y, tool_z)))


def align_spad_with_insertion(target):
    """SPAD 测量口从夹持器侧面 -Z 接近，+Y 沿叶面法向。"""
    direction=target.contact_position-target.approach_position
    length=np.linalg.norm(direction)
    if length<1e-8:raise ValueError('Approach and contact must be distinct')
    z=-direction/length;y=target.orientation[:,1]
    if abs(z@y)>1e-6:raise ValueError('Insertion must lie in the leaf plane')
    x=np.cross(y,z);x/=np.linalg.norm(x)
    return replace(target,orientation=np.column_stack((x,y,z)))
