"""回归：夹指前方必须朝向接触点，不能横着沿 +Z 插入。"""
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from widowx_codex.common import ROOT
from widowx_codex.tool_geometry import align_gripper_with_insertion
from wodowx_control.target_planning import make_leaf_target


def test_gripper_axes_match_urdf_and_base_side_insertion():
    urdf=ET.parse(ROOT/'assets/wx250s.original.urdf')
    offset=np.fromstring(urdf.find(".//joint[@name='ee_gripper']/origin").get('xyz'),sep=' ')
    np.testing.assert_allclose(offset/np.linalg.norm(offset),[1,0,0])
    np.testing.assert_allclose(np.fromstring(urdf.find(".//joint[@name='left_finger']/axis").get('xyz'),sep=' '),[0,1,0])
    points=np.array([[.3,.1,.2],[.31,.1,.2],[.3,.11,.2]])
    original=make_leaf_target(0,.8,points,points.mean(axis=0),np.array([0.,0.,1.]),.08)
    target=align_gripper_with_insertion(original)
    direction=target.contact_position-target.approach_position
    np.testing.assert_allclose(target.orientation[:,0],direction/np.linalg.norm(direction))
    np.testing.assert_allclose(target.orientation[:,1],[0,0,1])
    np.testing.assert_allclose(target.orientation.T@target.orientation,np.eye(3),atol=1e-12)
    assert np.isclose(np.linalg.det(target.orientation),1)
    assert np.linalg.norm(target.approach_position)<np.linalg.norm(target.contact_position)
    assert np.dot(target.approach_position-target.contact_position,-target.contact_position)>0
    np.testing.assert_array_equal(target.contact_position,original.contact_position)
    np.testing.assert_array_equal(target.approach_position,original.approach_position)
