"""使用本机 Interbotix 参数及 Modern Robotics 算法计算 WX250S 运动学。

EE 为 ee_gripper_link，参考系为 base_link；关节角为 rad，位置为 m。

WX250S kinematics using the same local Modern Robotics code and SDK data.

Position/velocity screening and the three default IK seeds mirror Interbotix.
Physics drives approximate motor dynamics; they are not a Dynamixel emulator.
"""
import sys
import xml.etree.ElementTree as ET
import numpy as np
from .common import ROOT, ARM, validate_pose
sys.path.insert(0, str(ROOT / 'reference'))
import modern_robotics as mr
from mr_descriptions import wx250s

class Model:
    def __init__(self, urdf=None, moving_time=2.0, accel_time=0.3):
        root = ET.parse(urdf or ROOT/'assets/wx250s.original.urdf').getroot()
        self.limits = {j.attrib['name']: {k: float(v) for k,v in j.find('limit').attrib.items()}
                       for j in root.findall('joint') if j.find('limit') is not None}
        self.lower = np.array([self.limits[n]['lower'] for n in ARM])
        self.upper = np.array([self.limits[n]['upper'] for n in ARM])
        self.velocity = np.array([self.limits[n]['velocity'] for n in ARM])
        self.moving_time, self.accel_time = moving_time, accel_time
        if not 0 < accel_time <= moving_time/2:
            raise ValueError('Require 0 < accel_time <= moving_time/2')

    def fk(self, q):
        return mr.FKinSpace(wx250s.M, wx250s.Slist, np.asarray(q))

    def ik(self, pose, command_q, seed=None):
        pose = validate_pose(pose)
        # 普通位姿沿用 SDK 的三组初猜；笛卡尔路点用上一路点解保持连续。
        seeds = [np.zeros(6), np.array([-2*np.pi/3,0,0,0,0,0]), np.array([2*np.pi/3,0,0,0,0,0])] if seed is None else [seed]
        for guess in seeds:
            q, ok = mr.IKinSpace(wx250s.Slist, wx250s.M, pose, np.asarray(guess), .001, .001)
            if not ok or not np.isfinite(q).all():
                continue
            # 先归一化到 [-pi, pi)，再按各关节限位选择等价角。
            q = (q + np.pi) % (2*np.pi) - np.pi
            for i in range(6):
                if round(q[i],3) < round(self.lower[i],3): q[i] += 2*np.pi
                elif round(q[i],3) > round(self.upper[i],3): q[i] -= 2*np.pi
            # 保留 SDK 的千分位截断检查，同时避免原始解本身穿过硬限位。
            rounded = np.trunc(q*1000)/1000
            if (np.all(rounded >= self.lower) and np.all(rounded <= self.upper)
                    and np.all(np.abs(rounded-command_q)/self.moving_time <= self.velocity)
                    and np.all(q >= self.lower) and np.all(q <= self.upper)):
                return q
        return None

    def plan(self, start_q, target, cartesian=False, command_pose=None):
        """Plan completely before execution. Cartesian tool orientation stays fixed."""
        target = validate_pose(target)
        if not cartesian:
            q = self.ik(target, start_q)
            if q is None: raise RuntimeError('IK/limits rejected EE pose')
            return np.array([start_q, q])
        start = self.fk(start_q) if command_pose is None else command_pose
        if not np.allclose(start[:3,:3], target[:3,:3], atol=1e-3):
            raise ValueError('Cartesian motion requires unchanged orientation')
        points = [np.array(start_q)]
        # 沿 base 坐标中的直线插值平移，保持工具朝向不变。
        for u in np.linspace(0, 1, int(self.moving_time/.05)+1)[1:]:
            t = start.copy(); t[:3,3] = (1-u)*start[:3,3] + u*target[:3,3]
            q = self.ik(t, start_q, points[-1])
            if q is None: raise RuntimeError('Cartesian waypoint rejected before motion')
            points.append(q)
        return np.array(points)

    def interpolate(self, points, elapsed, cartesian=False):
        t = np.clip(elapsed, 0, self.moving_time)
        # 笛卡尔路点按等时间插值；关节点到点运动采用梯形速度曲线。
        if cartesian:
            u = t/self.moving_time
        else:
            a, duration = self.accel_time, self.moving_time
            if t < a: distance = t*t/(2*a)
            elif t > duration-a: distance = duration-a-(duration-t)**2/(2*a)
            else: distance = t-a/2
            u = distance/(duration-a)
        x = u*(len(points)-1); i = min(int(x), len(points)-2)
        return points[i] + (x-i)*(points[i+1]-points[i])
