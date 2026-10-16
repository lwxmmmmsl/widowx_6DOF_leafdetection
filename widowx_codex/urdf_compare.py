"""按关节语义比较真机与仿真 URDF，排除等价 XML 写法导致的误报。

仅允许多出的单位 world→base 固定连接；实际参数变化仍返回详细差异。

Compare joint definitions while accepting an identity world/base attachment.

The simulator fixes the robot base at the world origin. An explicit zero-offset
world joint therefore does not change its kinematics. Other extra joints and
nonzero world offsets must still be reconciled, not silently ignored.
"""
import xml.etree.ElementTree as ET
import numpy as np


def _value(value):
    """Normalize numeric spelling (0, 0.0, scientific notation), not precision."""
    try:
        numbers = tuple(float(part) for part in value.split())
        if numbers and np.isfinite(numbers).all():
            return numbers
    except ValueError:
        pass
    return value


def _signature(element):
    children = list(element)
    # URDF permits omitted origin and omitted xyz/rpy; both default to zero.
    if element.tag == 'joint' and element.find('origin') is None:
        children.append(ET.Element('origin'))
    attributes = dict(element.attrib)
    if element.tag == 'origin':
        attributes.setdefault('xyz', '0 0 0')
        attributes.setdefault('rpy', '0 0 0')
    return (
        element.tag,
        tuple(sorted((key, _value(value)) for key, value in attributes.items())),
        tuple(sorted((_signature(child) for child in children), key=repr)),
    )


def _identity_world_joint(joint):
    if joint.get('type') != 'fixed':
        return False
    parent, child = joint.find('parent'), joint.find('child')
    if parent is None or child is None:
        return False
    if parent.get('link') != 'world' or child.get('link') != 'wx250s/base_link':
        return False
    if any(element.tag not in ('parent', 'child', 'origin') for element in joint):
        return False
    origin = joint.find('origin')
    if origin is None:
        return True
    for name in ('xyz', 'rpy'):
        vector = _value(origin.get(name, '0 0 0'))
        if not isinstance(vector, tuple) or len(vector) != 3:
            return False
        if not np.allclose(vector, 0.0, atol=1e-12, rtol=0):
            return False
    return True


def compare_joint_trees(expected_xml, live_xml):
    """Return actionable differences and explicitly accepted extra joint names."""
    def joints(xml):
        result = {}
        for joint in ET.fromstring(xml).findall('joint'):
            name = joint.attrib['name']
            if name in result:
                raise ValueError(f'Duplicate URDF joint: {name}')
            result[name] = joint
        return result

    expected, live = joints(expected_xml), joints(live_xml)
    differences, accepted = [], []
    for name in sorted(expected.keys() - live.keys()):
        differences.append(f'{name}: missing in live URDF')
    for name in sorted(live.keys() - expected.keys()):
        if _identity_world_joint(live[name]) and not accepted:
            accepted.append(name)
        else:
            differences.append(f'{name}: unexpected joint (only one identity world -> wx250s/base_link fixed joint is allowed)')
    for name in sorted(expected.keys() & live.keys()):
        if _signature(expected[name]) == _signature(live[name]):
            continue
        if expected[name].attrib != live[name].attrib:
            differences.append(f'{name}: joint attributes expected={expected[name].attrib}, live={live[name].attrib}')
        for tag in sorted({e.tag for e in expected[name]} | {e.tag for e in live[name]}):
            left = [e for e in expected[name] if e.tag == tag]
            right = [e for e in live[name] if e.tag == tag]
            if tag == 'origin':
                left = left or [ET.Element('origin')]
                right = right or [ET.Element('origin')]
            if sorted(map(_signature, left), key=repr) != sorted(map(_signature, right), key=repr):
                differences.append(f'{name}/{tag}: expected={[e.attrib for e in left]}, live={[e.attrib for e in right]}')
    return differences, accepted
