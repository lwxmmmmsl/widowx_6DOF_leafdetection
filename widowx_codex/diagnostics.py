"""逐阶段诊断与碰撞筛选重放；仅记录，不改变原筛选决策。"""
import json
import os
import time
from pathlib import Path
import numpy as np
from .display import atomic_json


def record_report(kind, report, logger=None):
    """当前报告供 UI 读取，JSONL 保留每次检测/测量的历史。"""
    directory = Path(os.environ['WIDOWX_RUN_DIR'])
    report = dict(report, wall_time=time.time())
    atomic_json(directory/f'{kind}_diagnostics.json', report)
    line = json.dumps(report, ensure_ascii=False, allow_nan=False)
    with (directory/f'{kind}_diagnostics.jsonl').open('a') as stream:
        stream.write(line+'\n')
    if logger:
        logger.info(f'{kind.upper()}_DIAGNOSTICS {line}')


def segment_evidence(start, end, obstacles, radius):
    """与原点云走廊筛选相同的距离公式，输出最近阻挡点与所属叶片。

    距离 <= radius 即视为冲突；这不是整臂网格碰撞检测。
    """
    direction = end-start
    length2 = float(direction@direction)
    nearest = None
    for leaf_id, points in obstacles:
        if not points.size:
            continue
        fractions = np.zeros(len(points)) if length2 < 1e-12 else np.clip(((points-start)@direction)/length2, 0, 1)
        closest = start+fractions[:, None]*direction
        distances = np.linalg.norm(points-closest, axis=1)
        index = int(np.argmin(distances))
        if nearest is None or float(distances[index]) < nearest['min_distance_m']:
            nearest = {'obstacle_leaf_id':int(leaf_id), 'min_distance_m':float(distances[index]),
                       'obstacle_point_m':points[index].tolist(), 'path_point_m':closest[index].tolist()}
    return dict(start_m=start.tolist(), end_m=end.tolist(), radius_m=radius,
                clear=nearest is None or nearest['min_distance_m'] > radius,
                nearest=nearest)


def collision_trace(candidates, desired_count, start_position, radius):
    """重放原贪心次序；每轮记录每个实际尝试的候选和两个路径段。

    到接近点的路径也检查目标叶片本身；进入叶片的路径排除该目标。
    达到数量上限而未尝试的候选不会被误记为碰撞失败。
    """
    remaining = list(candidates)
    selected, attempts = [], []
    start = np.asarray(start_position, dtype=float)
    while remaining and len(selected) < desired_count:
        chosen = None
        ordered = sorted(remaining, key=lambda t:(float(np.linalg.norm(t.approach_position-start)), -t.confidence, t.leaf_id))
        for target in ordered:
            other = [(t.leaf_id,t.points) for t in candidates if t.leaf_id != target.leaf_id and t.points.size]
            entry = segment_evidence(start, target.approach_position, other+[(target.leaf_id,target.points)], radius)
            # 原算法在第一段失败后短路，因此第二段标注未检查。
            insertion = segment_evidence(target.approach_position, target.contact_position, other, radius) if entry['clear'] else None
            clear = entry['clear'] and insertion['clear']
            attempts.append({'round':len(selected)+1, 'leaf_id':int(target.leaf_id),
                             'current_to_approach':entry, 'approach_to_contact':insertion,
                             'result':'selected' if clear else 'collision_rejected'})
            if clear:
                chosen = target
                break
        if chosen is None:
            break
        selected.append(chosen.leaf_id)
        remaining = [t for t in remaining if t.leaf_id != chosen.leaf_id]
        start = chosen.approach_position
    return {'selected_ids':selected, 'attempts':attempts,
            'not_selected_ids':[t.leaf_id for t in remaining],
            'stopped_reason':('target_count_reached' if len(selected)>=desired_count else
                              'candidates_exhausted' if not remaining else 'no_clear_candidate')}
