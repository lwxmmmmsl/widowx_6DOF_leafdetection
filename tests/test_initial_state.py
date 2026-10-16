"""真实 sleep 附近越界样本及 PhysX float32 边界回归。"""
import copy
import numpy as np
import pytest
from widowx_codex.initial_state import prepare_initial_joints
from widowx_codex.kinematics import Model


def test_reported_hardware_sample_is_projected_without_changing_source():
    measured = {'shoulder': -1.9527575969696045,
                'left_finger': .03749222308397293,
                'right_finger': -.03749222308397293,
                'gripper': 1.356039047241211}
    limits = Model().limits
    old_measured, old_limits = copy.deepcopy(measured), copy.deepcopy(limits)
    initial, changes = prepare_initial_joints(measured, limits)
    assert measured == old_measured and limits == old_limits
    assert {c['joint'] for c in changes} == {'shoulder', 'left_finger', 'right_finger'}
    assert initial['gripper'] == measured['gripper']
    assert initial['left_finger'] == -initial['right_finger']
    for change in changes:
        assert np.float32(change['lower']) < np.float32(change['simulated']) < np.float32(change['upper'])
        assert change['reason'] == 'outside_urdf_limits'


def test_strict_mode_rejects_actual_out_of_bounds():
    with pytest.raises(ValueError, match='shoulder=.*outside URDF limits'):
        prepare_initial_joints({'shoulder': -1.9527575969696045}, Model().limits, 'strict')


def test_legal_positions_and_boundary_margin():
    limits = Model().limits
    result, changes = prepare_initial_joints({'shoulder': 0, 'left_finger': .025}, limits, 'strict')
    assert result == {'shoulder': 0, 'left_finger': .025} and changes == []
    result, changes = prepare_initial_joints({'left_finger': .037}, limits, 'strict')
    assert .03699 < result['left_finger'] < .037
    assert changes[0]['reason'] == 'float32_boundary_margin'


@pytest.mark.parametrize('sample,policy', [({'waist':float('nan')}, 'clamp'),
                                         ({'unknown':0}, 'clamp'),
                                         ({'waist':0}, 'ignore')])
def test_bad_input_rejected(sample, policy):
    with pytest.raises(ValueError):
        prepare_initial_joints(sample, Model().limits, policy)
