"""Regression for robot_state_publisher's optional world attachment."""
import xml.etree.ElementTree as ET
import pytest
from widowx_codex.common import ROOT
from widowx_codex.urdf_compare import compare_joint_trees


def original():
    return (ROOT/'assets/wx250s.original.urdf').read_text()


def live_with_world(xyz=None):
    root = ET.fromstring(original())
    joint = ET.SubElement(root, 'joint', name='fixed', type='fixed')
    ET.SubElement(joint, 'parent', link='world')
    ET.SubElement(joint, 'child', link='wx250s/base_link')
    if xyz is not None:
        ET.SubElement(joint, 'origin', xyz=xyz, rpy='0 0 0')
    return root


def test_optional_identity_world_joint():
    for xyz in (None, '0 0.0 0e0'):
        differences, accepted = compare_joint_trees(original(), ET.tostring(live_with_world(xyz)))
        assert differences == []
        assert accepted == ['fixed']


def test_xml_order_numeric_spelling_and_origin_defaults():
    root = ET.fromstring(original())
    joint = root.find("joint[@name='waist']")
    joint[:] = list(reversed(list(joint)))
    joint.find('origin').set('rpy', '0.0 0e0 0')
    joint.find('limit').set('effort', '10.0')
    assert compare_joint_trees(original(), ET.tostring(root)) == ([], [])


@pytest.mark.parametrize('tag,attribute,value', [
    ('axis', 'xyz', '1 0 0'),
    ('limit', 'upper', '2.0'),
    ('origin', 'xyz', '0 0 0.073'),
    ('dynamics', 'damping', '0.2'),
])
def test_actual_parameter_changes_rejected(tag, attribute, value):
    root = live_with_world()
    root.find(f"joint[@name='waist']/{tag}").set(attribute, value)
    differences, _ = compare_joint_trees(original(), ET.tostring(root))
    assert any(f'waist/{tag}' in item for item in differences)


def test_nonzero_world_offset_and_extra_tool_joint_rejected():
    assert compare_joint_trees(original(), ET.tostring(live_with_world('0 0 0.1')))[0]
    root = live_with_world()
    joint = root.find("joint[@name='fixed']")
    joint.find('parent').set('link', 'wx250s/ee_gripper_link')
    joint.find('child').set('link', 'tool')
    assert compare_joint_trees(original(), ET.tostring(root))[0]


def test_missing_joint_rejected():
    root = ET.fromstring(original())
    root.remove(root.find("joint[@name='waist']"))
    assert 'waist: missing in live URDF' in compare_joint_trees(original(), ET.tostring(root))[0]
