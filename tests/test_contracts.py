"""Numerical/transport checks with no GPU or real robot required."""
import threading
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from widowx_codex.common import ROOT,ARM,profile,validate_pose
from widowx_codex.kinematics import Model
from widowx_codex.camera import CameraMapping
from widowx_codex.ipc import Server,Client,save_frame

def test_reference_is_exact_copy():
    original=ROOT.parent/'wodowx_control/src'
    for name in ('wodowx_control','wodowx_interfaces'):
        for path in (ROOT/'reference'/name).rglob('*'):
            if path.is_file() and '__pycache__' not in str(path) and path.suffix!='.pyc':
                assert path.read_bytes()==(original/name/path.relative_to(ROOT/'reference'/name)).read_bytes()
    assert (ROOT/'config/pipeline.yaml').read_bytes()==(original/'wodowx_control/config/pipeline.yaml').read_bytes()

def urdf_fk(q):
    root=ET.parse(ROOT/'assets/wx250s.original.urdf').getroot()
    joints={j.find('child').attrib['link']:j for j in root.findall('joint')}
    child='wx250s/ee_gripper_link'; chain=[]
    while child in joints:
        j=joints[child]; chain.append(j); child=j.find('parent').attrib['link']
    t=np.eye(4); positions=dict(zip(ARM,q))
    for j in reversed(chain):
        origin=j.find('origin'); step=np.eye(4)
        if origin is not None:
            step[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            step[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        t=t@step
        if j.attrib['name'] in positions:
            motion=np.eye(4); axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
            motion[:3,:3]=Rotation.from_rotvec(axis*positions[j.attrib['name']]).as_matrix(); t=t@motion
    return t

def test_sdk_fk_matches_urdf():
    model=Model(); rng=np.random.default_rng(123)
    for q in [np.zeros(6),*rng.uniform(model.lower,model.upper,(20,6))]:
        np.testing.assert_allclose(model.fk(q),urdf_fk(q),atol=1e-8)

def test_ik_cartesian_unreachable():
    m=Model(); q=np.array([.1,-.6,.5,.1,.1,-.1]); t=m.fk(q)
    solved=m.ik(t,q); assert solved is not None
    np.testing.assert_allclose(m.fk(solved),t,atol=.002)
    goal=t.copy(); goal[2,3]+=.01; path=m.plan(q,goal,True,t)
    assert len(path)==41
    for u,p in zip(np.linspace(0,1,41),path):
        np.testing.assert_allclose(m.fk(p)[:3,3],t[:3,3]+u*(goal[:3,3]-t[:3,3]),atol=.0011)
    goal[0,3]=10; assert m.ik(goal,q) is None

def test_motion_profile_and_pose_validation():
    m=Model(); points=np.array([np.zeros(6),np.ones(6)])
    np.testing.assert_allclose(m.interpolate(points,0),points[0])
    np.testing.assert_allclose(m.interpolate(points,2),points[1])
    values=np.array([m.interpolate(points,t)[0] for t in np.linspace(0,2,100)])
    assert np.all(np.diff(values)>=0)
    with pytest.raises(ValueError): validate_pose(np.zeros((4,4)))
    with pytest.raises(ValueError): validate_pose(np.full((4,4),np.nan))

def test_distorted_camera_and_depth():
    c=profile(ROOT/'config/demo_camera.yaml')['camera']; c.update(width=80,height=60,k=[65,0,37,0,69,29,0,0,1],d=[.15,-.02,.001,.002,0])
    m=CameraMapping(c); color=np.zeros((60,80,4),np.uint8); color[:,:,0]=255
    bgra,depth=m.convert(color,np.full((60,80),.75,dtype=np.float32))
    assert bgra.shape==(60,80,4) and (bgra[:,:,2]==255).all()
    assert depth.dtype==np.uint16 and (depth==750).all()
    np.testing.assert_allclose((m.map_x-m.render_k[0,2])/m.render_k[0,0],m.rays[:,:,0],atol=1e-6)
    _,bad=m.convert(color,np.full((60,80),np.inf,dtype=np.float32)); assert not bad.any()

def test_rpc_and_atomic_frames(tmp_path):
    server=Server(tmp_path/'test.sock')
    def worker():
        req=server.queue.get(timeout=2); assert req.method=='current_position'; req.finish([.1,.2,.3])
    thread=threading.Thread(target=worker); thread.start()
    try:
        assert Client(tmp_path/'test.sock').call('current_position')==[.1,.2,.3]
        with pytest.raises(RuntimeError,match='Unsupported'): Client(tmp_path/'test.sock').call('hardware_command')
    finally: thread.join(); server.close()
    path=tmp_path/'latest.npz'; save_frame(path,color=np.zeros((2,3,4),np.uint8),depth=np.zeros((2,3),np.uint16),sequence=1)
    with np.load(path,allow_pickle=False) as frame: assert int(frame['sequence'])==1
    assert not path.with_suffix('.tmp').exists()
