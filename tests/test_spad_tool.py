"""SPAD 侧向轴与测量 TCP 的回归验证。"""
import numpy as np
from widowx_codex.spad_tool import SpadModel
from widowx_codex.kinematics import Model
from widowx_codex.tool_geometry import align_spad_with_insertion
from wodowx_control.target_planning import make_leaf_target


def test_side_entry_keeps_leaf_contact():
    points=np.array([[.3,.1,.2],[.31,.1,.2],[.3,.11,.2]])
    old=make_leaf_target(0,.8,points,points.mean(0),np.array([0.,0.,1.]),.08)
    t=align_spad_with_insertion(old)
    d=t.contact_position-t.approach_position;d/=np.linalg.norm(d)
    np.testing.assert_allclose(-t.orientation[:,2],d)
    np.testing.assert_allclose(t.orientation[:,1],[0,0,1])
    assert abs(t.orientation[:,0]@d)<1e-12
    assert np.isclose(np.linalg.det(t.orientation),1)
    np.testing.assert_array_equal(t.contact_position,old.contact_position)


def test_measurement_tcp_fk_ik_and_cartesian_path():
    m=SpadModel();bare=Model();q=np.array([0.,-.7,.8,0.,-.1,0.])
    a=m.fk(q)
    np.testing.assert_allclose(a,bare.fk(q)@m.offset)
    c=a.copy();c[:3,3]-=.01*a[:3,2]
    path=m.plan(q,c,True,a)
    np.testing.assert_allclose(m.fk(path[-1])[:3,3],c[:3,3],atol=.001)
    np.testing.assert_allclose(bare.fk(path[-1])@m.offset,m.fk(path[-1]),atol=1e-10)


def test_mount_body_points_positive_z_and_mouth_negative_z():
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from widowx_codex.spad_tool import ASSEMBLY
    root=ET.parse(ASSEMBLY).getroot()
    origin=root.find("joint[@name='spad_cross_mount']/origin")
    rotation=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy'),sep=' ')).as_matrix()
    # SPAD 原模型尾部在测量点的 -X 侧；尾部应在夹持器 +Z 侧。
    np.testing.assert_allclose(rotation@[-1,0,0],[0,0,1],atol=1e-12)
    np.testing.assert_allclose(rotation@[1,0,0],[0,0,-1],atol=1e-12)
    np.testing.assert_allclose(rotation@[0,0,1],[0,1,0],atol=1e-12)
