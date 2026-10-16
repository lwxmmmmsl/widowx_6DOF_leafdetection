"""重放必须与原筛选结果一致；验证自身叶片阻挡与数量上限的区分。"""
import sys
import numpy as np
from widowx_codex.common import ROOT
sys.path.insert(0,str(ROOT/'reference/wodowx_control'))
from wodowx_control.target_planning import LeafTarget, choose_collision_free_targets
from widowx_codex.diagnostics import collision_trace, segment_evidence


def target(i, points, approach, contact):
    return LeafTarget(i,.8,np.array(points,dtype=float).reshape(-1,3),
                      np.array(contact,dtype=float),np.array(approach,dtype=float),np.eye(3))


def test_self_leaf_blocks_approach():
    t=target(4,[[.5,0,0]],[1,0,0],[1.1,0,0])
    trace=collision_trace([t],3,np.zeros(3),.035)
    step=trace['attempts'][0]
    assert trace['selected_ids']==[]
    assert step['current_to_approach']['nearest']['obstacle_leaf_id']==4
    assert step['current_to_approach']['nearest']['min_distance_m']==0
    assert step['approach_to_contact'] is None


def test_randomized_selection_matches_original():
    rng=np.random.default_rng(932)
    for _ in range(100):
        candidates=[target(i,rng.uniform(-1,1,(12,3)),rng.uniform(-1,1,3),rng.uniform(-1,1,3)) for i in range(5)]
        start=rng.uniform(-1,1,3); radius=float(rng.uniform(.01,.4)); count=int(rng.integers(0,6))
        expected=choose_collision_free_targets(candidates,count,start,radius)
        trace=collision_trace(candidates,count,start,radius)
        assert trace['selected_ids']==[t.leaf_id for t in expected]


def test_segment_boundary_is_collision():
    result=segment_evidence(np.zeros(3),np.array([1.,0,0]),[(7,np.array([[.5,.035,0]]))],.035)
    assert not result['clear']
    assert result['nearest']['obstacle_leaf_id']==7


def test_count_limit_does_not_claim_collision():
    candidates=[target(i,[],[i+1,0,0],[i+1.1,0,0]) for i in range(2)]
    trace=collision_trace(candidates,1,np.zeros(3),.035)
    assert trace['selected_ids']==[0] and trace['not_selected_ids']==[1]
    assert trace['stopped_reason']=='target_count_reached'


def test_available_leaf_selected_without_claiming_collision():
    trace=collision_trace([target(0,[],[1,0,0],[1.1,0,0])],3,np.zeros(3),.035)
    assert trace['selected_ids']==[0]
    assert trace['stopped_reason']=='candidates_exhausted'
