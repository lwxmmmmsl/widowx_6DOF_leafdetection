"""ROS transport tests: only isolated domain 77; never import hardware SDK."""
import json
import os
import time
from types import SimpleNamespace
import numpy as np
import pytest
rclpy=pytest.importorskip('rclpy')
from sensor_msgs.msg import Image,CameraInfo,JointState
from rclpy.executors import SingleThreadedExecutor
from tf2_ros import Buffer,TransformListener
from rclpy.time import Time
from widowx_codex.common import ROOT,profile,ARM
from widowx_codex.ipc import save_frame,SimBackend
from widowx_codex.ros_bridge import Bridge

def test_bridge_message_contract(tmp_path,monkeypatch):
    if os.environ.get('ROS_DOMAIN_ID') != os.environ.get('WIDOWX_TEST_ROS_DOMAIN_ID', '179'):
        pytest.skip('Run ROS tests with scripts/test.sh in the isolated test domain')
    cfg=profile(ROOT/'config/demo_camera.yaml')
    (tmp_path/'ready.json').write_text(json.dumps({'profile':cfg,'joint_names':ARM}))
    monkeypatch.setenv('WIDOWX_RUN_DIR',str(tmp_path))
    rclpy.init(); bridge=Bridge(); observer=rclpy.create_node('sim_contract_observer'); received={}
    observer.create_subscription(Image,'/rgb/image_raw',lambda m:received.update(color=m),10)
    observer.create_subscription(Image,'/depth_to_rgb/image_raw',lambda m:received.update(depth=m),10)
    observer.create_subscription(CameraInfo,'/depth_to_rgb/camera_info',lambda m:received.update(info=m),10)
    observer.create_subscription(JointState,'/wx250s/joint_states',lambda m:received.update(joints=m),10)
    buffer=Buffer(); listener=TransformListener(buffer,observer)
    executor=SingleThreadedExecutor(); executor.add_node(bridge); executor.add_node(observer)
    deadline=time.monotonic()+5; i=0
    try:
        while time.monotonic()<deadline and len(received)<4:
            save_frame(tmp_path/'latest.npz',sequence=i,stamp=2+i*.1,color=np.zeros((480,640,4),np.uint8),depth=np.full((480,640),800,np.uint16),joints=np.zeros(6))
            i+=1
            for _ in range(10): executor.spin_once(timeout_sec=.01)
        assert set(received)=={'color','depth','info','joints'}
        assert received['depth'].encoding=='16UC1'
        assert received['color'].encoding=='bgra8'
        assert received['color'].header==received['depth'].header==received['info'].header
        np.testing.assert_array_equal(received['info'].k,cfg['camera']['k'])
        assert np.frombuffer(received['depth'].data,np.uint16)[0]==800
        transform=buffer.lookup_transform('wx250s/base_link',cfg['camera']['frame_id'],Time())
        assert transform.transform.translation.y==-.6
    finally:
        executor.shutdown(); bridge.destroy_node(); observer.destroy_node(); rclpy.shutdown()

def test_control_backend_cannot_instantiate_hardware(monkeypatch,tmp_path):
    from widowx_codex.ros_control import SimulationControlNode
    import wodowx_control.control_node as original
    def forbidden(*args,**kwargs): raise AssertionError('Hardware backend called!')
    monkeypatch.setattr(original,'InterbotixArmBackend',forbidden)
    monkeypatch.setenv('WIDOWX_RUN_DIR',str(tmp_path))
    node=SimpleNamespace(_backend=None)
    assert isinstance(SimulationControlNode._get_backend(node),SimBackend)


def test_empty_selection_never_commands_gripper_or_arm():
    from unittest.mock import Mock
    from widowx_codex.ros_control import SimulationControlNode
    backend = Mock()
    node = SimpleNamespace(get_logger=lambda: Mock())
    assert SimulationControlNode._execute_measurements(node, backend, []) == 0
    assert backend.mock_calls == []


def test_execution_failure_preserves_planned_count(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from widowx_codex.ros_control import SimulationControlNode
    from wodowx_control.control_node import LeafControlNode
    node = SimulationControlNode.__new__(SimulationControlNode)
    node._base_frame = 'wx250s/base_link'
    monkeypatch.setenv('WIDOWX_RUN_DIR', str(tmp_path))
    monkeypatch.setattr(node, 'get_logger', lambda: Mock())
    monkeypatch.setattr(node, 'get_parameter', lambda _: SimpleNamespace(value=3))
    def fail_execution(self, request, response):
        self._report.update(stage='execution', collision={'actual_selected_ids':[0]})
        return response
    monkeypatch.setattr(LeafControlNode, '_measure_callback', fail_execution)
    response = SimpleNamespace(success=False, planned_targets=0, measured_targets=0,
                               target_shortfall=False, message='open_gripper timeout')
    result = node._measure_callback(SimpleNamespace(sequence=1), response)
    assert not result.success and result.planned_targets == 1 and result.target_shortfall
    assert node._report['failure_stage'] == 'execution'
    assert node._report['selected_count'] == node._report['planned'] == 1


def test_ik_trace_records_both_poses(monkeypatch):
    from unittest.mock import Mock
    from widowx_codex.ros_control import SimulationControlNode
    from wodowx_control.target_planning import LeafTarget
    node = SimulationControlNode.__new__(SimulationControlNode)
    node._report = {'ik': []}
    monkeypatch.setattr(node, 'get_logger', lambda: Mock())
    target = LeafTarget(6, .8, np.zeros((1,3)), np.array([.2,0,.2]),
                        np.array([.12,0,.2]), np.eye(3))
    backend = SimpleNamespace(pose_reachable=Mock(side_effect=[False, True]))
    assert node._keep_ik_reachable_candidates(backend, [target]) == []
    assert [(row['pose'], row['reachable']) for row in node._report['ik']] == [('approach',False),('contact',True)]
    assert node._report['ik'][0]['position_m'] == [.12,0,.2]


def test_perception_diagnostic_preserves_original_drawing(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from widowx_codex.ros_perception import DiagnosticPerceptionNode
    from wodowx_control.perception_node import LeafPerceptionNode
    from std_srvs.srv import Trigger
    node = DiagnosticPerceptionNode.__new__(DiagnosticPerceptionNode)
    node._sequence = 2
    node._target_label = 'leaf'
    monkeypatch.setenv('WIDOWX_RUN_DIR', str(tmp_path))
    monkeypatch.setattr(node,'get_parameter',lambda _:SimpleNamespace(value=.2))
    monkeypatch.setattr(node,'get_logger',lambda:Mock())
    def process(self,response):
        self._sequence += 1
        self._draw_prediction(np.zeros((20,20,3),np.uint8),None,np.array([1,1,5,5]),
                              'Leaf',7,.7,(0,255,255),'no segmentation mask')
        response.success=True; response.message='valid_leaf_instances=0'
        return response
    monkeypatch.setattr(LeafPerceptionNode,'_process_detection_request',process)
    node._process_detection_request(Trigger.Response())
    data=json.loads((tmp_path/'perception_diagnostics.json').read_text())
    assert data['sequence']==3 and data['reason']=='leaf_geometry_rejected'
    assert data['predictions'][0]['result']=='no segmentation mask'


def test_target_projection_occurs_after_base_transform(monkeypatch):
    from unittest.mock import Mock
    from widowx_codex.ros_control import SimulationControlNode
    from wodowx_control.control_node import LeafControlNode
    from wodowx_control.target_planning import make_leaf_target
    node = SimulationControlNode.__new__(SimulationControlNode)
    node._report = {'targets': []}
    parameters={'project_normal_to_base_yz':True, 'approach_distance_m':.08, 'spad_mode':False}
    monkeypatch.setattr(node,'get_parameter',lambda key:SimpleNamespace(value=parameters[key]))
    monkeypatch.setattr(node,'get_logger',lambda:Mock())
    contact=np.array([.3,.1,.2]); normal=np.array([.8,.3,-.4])
    transformed=make_leaf_target(0,.8,np.array([contact]),contact,normal,.08)
    # 模拟原函数的输出：其 orientation[:,1] 已经在基座系中。
    monkeypatch.setattr(LeafControlNode,'_build_candidates',lambda *_:[transformed])
    leaf=SimpleNamespace(id=0,cloud=SimpleNamespace(header=SimpleNamespace(frame_id='camera')))
    result=node._build_candidates(SimpleNamespace(detections=[leaf],header=SimpleNamespace(frame_id='camera')))
    np.testing.assert_allclose(result[0].orientation[:,1],[0,.6,-.8])
    np.testing.assert_array_equal(result[0].contact_position,contact)
    assert abs(np.dot(result[0].contact_position-result[0].approach_position,result[0].orientation[:,1]))<1e-12


def test_selected_relaxed_orientation_reaches_execution_targets(monkeypatch):
    from widowx_codex.ros_control import SimulationControlNode
    from wodowx_control.target_planning import make_leaf_target
    from scipy.spatial.transform import Rotation
    node=SimulationControlNode.__new__(SimulationControlNode)
    node._report={'targets':[{'leaf_id':0}], 'ik':[]}
    monkeypatch.setattr(node,'get_parameter',lambda key:SimpleNamespace(value=15. if key=='orientation_tolerance_deg' else 5.))
    contact=np.array([.3,.1,.2])
    target=make_leaf_target(0,.8,np.array([contact,contact+[.01,0,0],contact+[0,.01,0]]),contact,np.array([0,0,1.]),.08)
    selected=target.orientation @ Rotation.from_rotvec([.1,0,0]).as_matrix()
    def resolve(approach,contact,tolerance,step):
        approach[:3,:3]=selected;contact[:3,:3]=selected
        return dict(reachable=True,approach_pose=approach.tolist(),contact_pose=contact.tolist(),orientation_error_deg=np.rad2deg(.1))
    results=node._keep_ik_reachable_candidates(SimpleNamespace(resolve_leaf=resolve),[target])
    np.testing.assert_allclose(results[0].orientation,selected)
    np.testing.assert_array_equal(results[0].approach_position,target.approach_position)
    np.testing.assert_array_equal(results[0].contact_position,target.contact_position)
    assert node._report['ik'][0]['reachable']


def test_spad_alignment_never_closes_gripper(monkeypatch):
    from unittest.mock import Mock
    from widowx_codex import ros_control
    from wodowx_control.target_planning import make_leaf_target
    node=ros_control.SimulationControlNode.__new__(ros_control.SimulationControlNode)
    node._report={}
    monkeypatch.setattr(node,'get_parameter',lambda key:SimpleNamespace(value={'spad_mode':True,'gripper_hold_s':0}[key]))
    monkeypatch.setattr(node,'get_logger',lambda:Mock())
    monkeypatch.setattr(ros_control,'record_report',lambda *args:None)
    p=np.array([.3,.1,.2]);t=make_leaf_target(0,.8,np.array([p,p+[.01,0,0],p+[0,.01,0]]),p,np.array([0.,0.,1.]),.08)
    backend=Mock()
    assert node._execute_measurements(backend,[t])==0
    backend.open_gripper.assert_not_called();backend.close_gripper.assert_not_called()
    assert backend.move_pose.call_count==1
    assert backend.move_cartesian_to_pose.call_count==2
    assert node._report['aligned_targets']==1
    assert node._report['measurement_performed'] is False
