"""固定安装开度下的 SPAD 测量 TCP；姿态轴仍使用夹持器坐标轴。"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import Model
from .common import ROOT

ASSEMBLY = ROOT.parent/'widowx_spad_assembly/wx250s_with_spad.urdf'
FINGER = .0369
JAW = .1


def measurement_offset(urdf=ASSEMBLY, finger=FINGER):
    """读取装配树，计算 ee_gripper→测量口平移；不复制手填的安装尺寸。"""
    root=ET.parse(urdf).getroot()
    poses={'wx250s_fingers_link':np.eye(4)}
    for _ in range(len(root.findall('joint'))):
        for j in root.findall('joint'):
            parent=j.find('parent').get('link');child=j.find('child').get('link')
            if parent not in poses or child in poses:continue
            o=j.find('origin');t=np.eye(4)
            if o is not None:
                t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            if j.get('name')=='right_finger':t[:3,3]+=[0,-finger,0]
            elif j.get('name')=='left_finger':t[:3,3]+=[0,finger,0]
            poses[child]=poses[parent]@t
    t=np.eye(4)
    # TCP 坐标轴保留 EE 方向，因此进叶侧向轴为 -Z，闭合轴为 +Y。
    t[:3,3]=(np.linalg.inv(poses['wx250s_ee_gripper_link'])@poses['spad_measurement_tip'])[:3,3]
    return t


class SpadModel(Model):
    """对外收发测量口位姿，对内仍使用原 WX250S 六关节运动学。"""
    def __init__(self, **kwargs):
        super().__init__(urdf=ASSEMBLY, **kwargs)
        self.offset=measurement_offset()
    def fk(self,q):
        return super().fk(q)@self.offset
    def ik(self,pose,command_q,seed=None):
        return super().ik(np.asarray(pose)@np.linalg.inv(self.offset),command_q,seed)
