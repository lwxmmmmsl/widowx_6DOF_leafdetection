"""验证姿态放宽的边界、公共朝向及位置精度，不需要 GPU。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from widowx_codex.pose_tolerance import project_normal_yz, resolve_leaf_pose


class SampleModel:
    def __init__(self, reject_below_deg=0, drift=0, fail_retreat=False):
        self.reject_below_deg = reject_below_deg
        self.drift = drift
        self.fail_retreat = fail_retreat
        self.calls = []

    def fk(self, q):
        pose = np.asarray(q).reshape(4, 4).copy()
        pose[0, 3] += self.drift
        return pose

    def plan(self, q, target, cartesian=False, command_pose=None):
        self.calls.append((cartesian, target.copy(), command_pose))
        if np.rad2deg(Rotation.from_matrix(target[:3,:3]).magnitude()) < self.reject_below_deg:
            raise RuntimeError('Synthetic joint limit')
        if cartesian and self.fail_retreat and target[0,3] < command_pose[0,3]:
            raise RuntimeError('Synthetic retreat failure')
        if not cartesian:
            return np.array([q, target.ravel()])
        assert np.allclose(command_pose[:3,:3], target[:3,:3])
        return np.array([command_pose.ravel(), target.ravel()])


def poses():
    a=np.eye(4); a[:3,3]=[.2,.1,.3]
    c=a.copy(); c[0,3]+=.08
    return a,c


def test_base_yz_projection_and_degenerate_rejection():
    np.testing.assert_allclose(project_normal_yz([.8,.3,-.4]), [0,.6,-.8])
    for normal in ([1,0,0], [0,0,0], [0,float('nan'),1]):
        with pytest.raises(ValueError): project_normal_yz(normal)


def test_exact_orientation_first():
    a,c=poses(); model=SampleModel()
    result=resolve_leaf_pose(model,np.eye(4).ravel(),a,c)
    assert result['reachable'] and result['orientation_error_deg']==0
    assert len(result['attempts'])==1


def test_relaxed_pose_preserves_positions_and_common_orientation():
    a,c=poses(); model=SampleModel(reject_below_deg=6)
    result=resolve_leaf_pose(model,np.eye(4).ravel(),a,c,15,5)
    assert result['reachable'] and result['orientation_error_deg']==10
    selected_a,selected_c=np.array(result['approach_pose']),np.array(result['contact_pose'])
    np.testing.assert_array_equal(selected_a[:3,3], a[:3,3])
    np.testing.assert_array_equal(selected_c[:3,3], c[:3,3])
    np.testing.assert_array_equal(selected_a[:3,:3], selected_c[:3,:3])
    assert result['attempts'][-1]['max_orientation_error_deg'] <= 15
    assert len(model.calls[-3:]) == 3


def test_zero_tolerance_disables_search():
    a,c=poses()
    result=resolve_leaf_pose(SampleModel(reject_below_deg=1),np.eye(4).ravel(),a,c,0)
    assert not result['reachable'] and len(result['attempts'])==1


def test_position_drift_and_retreat_failure_are_rejected():
    a,c=poses()
    for model, stage in ((SampleModel(drift=.002),'fk_residual'),
                         (SampleModel(fail_retreat=True),'retreat_ik')):
        result=resolve_leaf_pose(model,np.eye(4).ravel(),a,c,0)
        assert not result['reachable']
        assert result['attempts'][0]['failure_stage']==stage


def test_invalid_config_and_deadline():
    a,c=poses()
    for tolerance,step in ((-1,5),(46,5),(15,0),(15,.01),(float('nan'),5)):
        with pytest.raises(ValueError): resolve_leaf_pose(SampleModel(),np.eye(4).ravel(),a,c,tolerance,step)
    assert resolve_leaf_pose(SampleModel(),np.eye(4).ravel(),a,c,deadline=0)['reason']=='orientation_search_timeout'
