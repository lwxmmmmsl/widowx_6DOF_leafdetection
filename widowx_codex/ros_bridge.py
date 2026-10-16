"""将仿真 RGB-D、关节状态与时间发布成原工程使用的 ROS 消息。

同一帧的彩色图、深度图和 CameraInfo 共用时间戳及 RGB optical frame。

Publish Isaac frames into the unchanged real pipeline's ROS message contract."""
import json
import os
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from sensor_msgs.msg import Image,CameraInfo,JointState
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation
from .common import ROOT, read_yaml
from .display import atomic_json, read_json, image_to_rgba
from .ipc import save_frame
from std_srvs.srv import Trigger
from wodowx_interfaces.msg import WorkflowStatus, LeafDetectionArray

class Bridge(Node):
    def __init__(self):
        super().__init__('widowx_isaac_bridge')
        self.directory=Path(os.environ['WIDOWX_RUN_DIR'])
        self.meta=json.loads((self.directory/'ready.json').read_text())
        self.camera=self.meta['profile']['camera']
        topics=read_yaml(ROOT/'config/pipeline.yaml')['leaf_perception']['ros__parameters']
        qos=QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE)
        self.color_pub=self.create_publisher(Image,topics['color_topic'],qos)
        self.depth_pub=self.create_publisher(Image,topics['depth_topic'],qos)
        self.info_pub=self.create_publisher(CameraInfo,topics['camera_info_topic'],qos)
        self.joint_pub=self.create_publisher(JointState,'/wx250s/joint_states',qos)
        self.clock_pub=self.create_publisher(Clock,'/clock',10)
        # 发布 base→RGB optical 静态变换，供原控制节点转换叶片点云。
        self.broadcaster=StaticTransformBroadcaster(self)
        transform=TransformStamped(); transform.header.frame_id='wx250s/base_link'
        transform.child_frame_id=self.camera['frame_id']
        t=np.array(self.camera['T_base_optical']); q=Rotation.from_matrix(t[:3,:3]).as_quat()
        transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z=map(float,t[:3,3])
        transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z,transform.transform.rotation.w=map(float,q)
        world=TransformStamped(); world.header.frame_id='world'; world.child_frame_id='wx250s/base_link'
        world.transform.rotation.w=1.0
        self.broadcaster.sendTransform([world,transform])
        # 回传原 perception 的调试图，不在 bridge 内加载另一份 YOLO。
        self.create_subscription(Image, topics['debug_image_topic'], self.on_debug_image, qos)
        self.create_subscription(WorkflowStatus, '/workflow/status', self.on_workflow, 10)
        self.preview_client = self.create_client(Trigger, '/perception/debug_detect')
        self.preview_future = None
        self.preview_id = None
        self.preview_message = 'Preview button runs YOLO only; no motion'
        self.workflow = {}
        self.production = {}
        # 正式检测的三维叶片与低阈值 RGB 预览是两类结果，分别显示。
        detection_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                   reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(LeafDetectionArray, topics['detections_topic'],
                                 self.on_production_detections, detection_qos)
        self.create_timer(.5, self.update_panel_state)
        self.previous=-1; self.timer=self.create_timer(.01,self.publish_frame)

    def on_debug_image(self, message):
        """保留推理图原始时间戳，防止重发旧图掩盖检测停滞。"""
        try:
            rgba = image_to_rgba(message)
            stamp = message.header.stamp.sec + message.header.stamp.nanosec/1e9
            save_frame(self.directory/'yolo_debug.npz', rgba=rgba, stamp=stamp)
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def on_production_detections(self, message):
        self.production = {'sequence': int(message.sequence),
                           'valid_leaves': len(message.detections)}

    def on_workflow(self, message):
        self.workflow = {name: getattr(message, name) for name in
                         ('finished', 'success', 'target_shortfall',
                          'planned_targets', 'measured_targets', 'message')}

    def update_panel_state(self):
        request = read_json(self.directory/'preview_request.json')
        if self.preview_future is not None and self.preview_future.done():
            try:
                result = self.preview_future.result()
                self.preview_message = result.message
            except Exception as exc:
                self.preview_message = f'Preview failed: {exc}'
            self.preview_future = None
        if request.get('id') and request['id'] != self.preview_id and self.preview_future is None:
            if self.preview_client.service_is_ready():
                self.preview_id = request['id']
                self.preview_future = self.preview_client.call_async(Trigger.Request())
                self.preview_message = 'YOLO preview running...'
            else:
                self.preview_message = 'Waiting for perception service; start run_pipeline.sh'
        atomic_json(self.directory/'panel_status.json', {
            'heartbeat': time.time(), 'detect_ready': self.preview_client.service_is_ready(),
            'preview': self.preview_message, 'workflow': self.workflow,
            'production': self.production,
        })

    def publish_frame(self):
        path=self.directory/'latest.npz'
        if not path.exists(): return
        # np.load reads one atomically replaced file; no color/depth timestamp mixing.
        with np.load(path,allow_pickle=False) as frame:
            sequence=int(frame['sequence'])
            if sequence==self.previous: return
            self.previous=sequence
            # 从帧内仿真时间产生 ROS 时间；不能使用 bridge 本机墙钟替代。
            nanos=round(float(frame['stamp'])*1e9); stamp=Time(sec=nanos//10**9,nanosec=nanos%10**9)
            clock=Clock(); clock.clock=stamp; self.clock_pub.publish(clock)
            for key,encoding,publisher in [('color','bgra8',self.color_pub),('depth','16UC1',self.depth_pub)]:
                data=np.ascontiguousarray(frame[key]); msg=Image()
                msg.header.stamp=stamp; msg.header.frame_id=self.camera['frame_id']
                msg.height,msg.width=data.shape[:2]; msg.encoding=encoding; msg.is_bigendian=0
                msg.step=data.strides[0]; msg.data=data.tobytes(); publisher.publish(msg)
            info=CameraInfo(); info.header=msg.header
            info.width=self.camera['width']; info.height=self.camera['height']
            info.distortion_model=self.camera.get('distortion_model','plumb_bob')
            info.k=list(map(float,self.camera['k'])); info.d=list(map(float,self.camera.get('d',[])))
            info.r=list(map(float,self.camera.get('r',np.eye(3).flatten())))
            info.p=list(map(float,self.camera.get('p',np.column_stack([np.array(info.k).reshape(3,3),np.zeros(3)]).flatten())))
            self.info_pub.publish(info)
            joints=JointState(); joints.header.stamp=stamp; joints.name=self.meta['joint_names']
            joints.position=frame['joints'].astype(float).tolist(); self.joint_pub.publish(joints)

def main():
    rclpy.init(); node=Bridge()
    try: rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__=='__main__': main()
