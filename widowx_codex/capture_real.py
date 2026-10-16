"""只读采集真机相机内参、base→optical TF、URDF 和关节位置。

仅订阅和查询；不启动驱动、不发布机械臂运动指令。原始采样值原样保存。

Read-only capture of hardware camera calibration, TF, URDF and joint state.

Run in the REAL ROS domain with the already running Kinect and robot publishers.
This node has subscribers/TF listeners only and cannot move any joint.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from .common import ROOT, ARM, event
from .urdf_compare import compare_joint_trees

def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo,JointState
    from std_msgs.msg import String
    from tf2_ros import Buffer,TransformListener,TransformException
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'config/hardware.yaml')
    parser.add_argument('--timeout',type=float,default=30)
    parser.add_argument('--description-topic',default='/wx250s/robot_description')
    args=parser.parse_args()
    rclpy.init(); node=Node('widowx_read_only_profile_capture'); data={}
    buffer=Buffer(); listener=TransformListener(buffer,node)
    # 读取与彩色图对齐的深度内参，避免误用原始深度相机的内参/光学系。
    node.create_subscription(CameraInfo,'/depth_to_rgb/camera_info',lambda m:data.update(info=m),qos_profile_sensor_data)
    node.create_subscription(JointState,'/wx250s/joint_states',lambda m:data.update(joints=m),qos_profile_sensor_data)
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    node.create_subscription(String,args.description_topic,lambda m:data.update(urdf=m.data),qos)
    deadline=time.monotonic()+args.timeout
    try:
        transform=None
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if all(key in data for key in ('info','joints','urdf')):
                try:
                    transform=buffer.lookup_transform('wx250s/base_link',data['info'].header.frame_id,Time())
                    if all(n in data['joints'].name for n in ARM): break
                except TransformException: pass
        if transform is None or not all(k in data for k in ('info','joints','urdf')):
            raise RuntimeError(f'Capture incomplete: received {list(data)}. Check CameraInfo, joint_states, robot_description and full optical TF; no default calibration was written.')
        # 导出前检查机器人关节树，但允许等价的零偏移 world 固定连接。
        differences, accepted = compare_joint_trees(
            (ROOT/'assets/wx250s.original.urdf').read_text(), data['urdf']
        )
        if differences:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            mismatch = args.output.with_suffix('.unmatched.urdf')
            mismatch.write_text(data['urdf'])
            raise RuntimeError(
                'Live URDF joint tree differs from bundled WX250S:\n  '
                + '\n  '.join(differences)
                + f'\nSaved {mismatch}; reconcile these differences before exporting.'
            )
        if accepted:
            event('urdf_equivalent_world_attachment', joints=accepted,
                  reason='Identity world -> wx250s/base_link fixed attachment')
        info=data['info']; t=transform.transform.translation; q=transform.transform.rotation
        matrix=np.eye(4); matrix[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix(); matrix[:3,3]=[t.x,t.y,t.z]
        # 编码器读数原样落盘；仿真初值投影在 initial_state.py 中另行处理。
        cfg={'source':'captured_from_live_ros','base_frame':'wx250s/base_link','captured_unix':time.time(),
            'camera':{'width':info.width,'height':info.height,'k':list(map(float,info.k)), 'd':list(map(float,info.d)),
              'r':list(map(float,info.r)),'p':list(map(float,info.p)), 'distortion_model':info.distortion_model,
              'frame_id':info.header.frame_id,'T_base_optical':matrix.tolist()},
            'initial_joints':dict(zip(data['joints'].name,map(float,data['joints'].position)))}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
        args.output.with_suffix('.urdf').write_text(data['urdf'])
        event('hardware_profile_saved',path=str(args.output))
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()
