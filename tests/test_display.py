"""面板显示协议：颜色、行填充和过期心跳不能误导用户。"""
from types import SimpleNamespace
import time
import numpy as np
import pytest
from widowx_codex.display import image_to_rgba, atomic_json, panel_status


def test_bgr_padding_to_rgba():
    msg = SimpleNamespace(encoding='bgr8', height=1, width=2, step=8,
                          data=bytes([10,20,30,40,50,60,99,99]))
    np.testing.assert_array_equal(image_to_rgba(msg), [[[30,20,10,255],[60,50,40,255]]])
    msg.step=2
    with pytest.raises(ValueError): image_to_rgba(msg)


def test_offline_and_live_status(tmp_path):
    assert 'offline' in panel_status(tmp_path)
    atomic_json(tmp_path/'panel_status.json', {'heartbeat':time.time(), 'detect_ready':True,
        'workflow':{'message':'No reachable leaves','planned_targets':0}})
    assert 'No reachable leaves' in panel_status(tmp_path)
    atomic_json(tmp_path/'panel_status.json', {'heartbeat':time.time()-10})
    assert 'offline' in panel_status(tmp_path)
