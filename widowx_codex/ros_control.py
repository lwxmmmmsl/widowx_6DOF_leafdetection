"""原控制节点的仿真适配层：替换后端并跳过空目标运动。

目标筛选及测量顺序仍由原 LeafControlNode 执行，运动请求只发往本地仿真。

Run the original control node with a simulation-only execution backend."""
import os
import time
from dataclasses import replace
import numpy as np
from types import SimpleNamespace
from .common import ROOT
import sys
sys.path.insert(0,str(ROOT/'reference/wodowx_control'))
import rclpy
from wodowx_control.control_node import LeafControlNode
from .ipc import SimBackend
from .diagnostics import record_report, collision_trace
from .pose_tolerance import project_normal_yz
from .tool_geometry import align_gripper_with_insertion, align_spad_with_insertion
from wodowx_control.target_planning import make_leaf_target

class SimulationControlNode(LeafControlNode):
    def __init__(self):
        super().__init__()
        self.declare_parameter('spad_mode', False)
        self.declare_parameter('project_normal_to_base_yz', True)
        self.declare_parameter('orientation_tolerance_deg', 15.0)
        self.declare_parameter('orientation_search_step_deg', 5.0)

    def _measure_callback(self, request, response):
        # 单线程 ROS executor；每次测量建立独立报告，不与上次序号混淆。
        self._report = {'sequence':int(request.sequence), 'frame':self._base_frame,
                        'targets':[], 'ik':[], 'collision':None, 'stage':'request'}
        self._trace_candidates = []
        try:
            result = super()._measure_callback(request, response)
            # 原节点异常分支未填写 planned_targets；执行失败不等于规划数为零。
            selected_count = len((self._report.get('collision') or {}).get('actual_selected_ids', []))
            if not result.success and self._report['stage'] == 'execution':
                result.planned_targets = selected_count
                result.target_shortfall = selected_count < int(self.get_parameter('target_count').value)
            self._report.update(success=bool(result.success), planned=int(result.planned_targets),
                                measured=int(result.measured_targets), message=result.message)
            self._report['selected_count'] = selected_count
            if not result.success:
                self._report['failure_stage'] = self._report['stage']
            elif not result.planned_targets:
                self._report['no_motion_reason'] = (
                    'normal_projection_degenerate' if not self._report['targets'] and self._report.get('geometry_rejections') else
                    'no_valid_3d_leaf' if not self._report['targets'] else
                    'all_targets_failed_ik' if not self._trace_candidates else
                    'requested_target_count_zero' if int(self.get_parameter('target_count').value) == 0 else
                    'all_remaining_targets_blocked_by_collision')
            return result
        except Exception as exc:
            self._report.update(success=False, failure_stage=self._report['stage'], error=str(exc))
            raise
        finally:
            record_report('control', self._report, self.get_logger())

    def _build_candidates(self, detections):
        self._report['stage'] = 'tf_and_target_geometry'
        self._report['input_valid_leaves'] = len(detections.detections)
        candidates = []
        for leaf in detections.detections:
            try:
                # 用原函数执行 TF/目标几何，逐片捕获错误；失败仍按原逻辑终止。
                built = super()._build_candidates(SimpleNamespace(header=detections.header, detections=[leaf]))
            except Exception as exc:
                self._report['target_error'] = {'leaf_id':int(leaf.id), 'error':str(exc)}
                raise
            for target in built:
                # 原函数已完成相机→基座 TF；夹持闭合轴 +Y 正是归一化叶面法向。
                normal_original = target.orientation[:,1].copy()
                if self.get_parameter('project_normal_to_base_yz').value:
                    try:
                        normal = project_normal_yz(normal_original)
                    except ValueError as exc:
                        self._report.setdefault('geometry_rejections', []).append(
                            {'leaf_id':int(leaf.id), 'reason':str(exc),
                             'normal_base_original':normal_original.tolist()})
                        continue
                    target = make_leaf_target(target.leaf_id, target.confidence, target.points,
                        np.mean(target.points, axis=0), normal,
                        float(self.get_parameter('approach_distance_m').value))
                # 原快照误把 +Z 当夹指伸出轴；URDF 中实际为 +X。
                spad=bool(self.get_parameter('spad_mode').value)
                target = (align_spad_with_insertion if spad else align_gripper_with_insertion)(target)
                self._report['targets'].append({'leaf_id':int(target.leaf_id),
                    'confidence':float(target.confidence), 'cloud_points':len(target.points),
                    'source_frame':leaf.cloud.header.frame_id or detections.header.frame_id,
                    'approach_position_m':target.approach_position.tolist(),
                    'contact_position_m':target.contact_position.tolist(),
                    'normal_base_original':normal_original.tolist(),
                    'normal_base_used':target.orientation[:,1].tolist(),
                    'insertion_axis_base':(-target.orientation[:,2] if spad else target.orientation[:,0]).tolist(),
                    'tool_forward_axis':'-Z' if spad else '+X',
                    'task_tcp':'spad_measurement_tip' if spad else 'ee_gripper_link',
                    'orientation_matrix':target.orientation.tolist()})
                self.get_logger().info(
                    f'Leaf {target.leaf_id} simulation target: '
                    f'base_normal={target.orientation[:,1].tolist()}, '
                    f'approach={target.approach_position.tolist()}, contact={target.contact_position.tolist()}')
                candidates.append(target)
        self._trace_candidates = candidates
        return candidates

    def _keep_ik_reachable_candidates(self, backend, candidates):
        self._report['stage'] = 'ik'
        rows = self._report['ik']
        if hasattr(backend, 'resolve_leaf'):
            reachable = []
            for candidate in candidates:
                result = backend.resolve_leaf(
                    candidate.pose_matrix(candidate.approach_position),
                    candidate.pose_matrix(candidate.contact_position),
                    float(self.get_parameter('orientation_tolerance_deg').value),
                    float(self.get_parameter('orientation_search_step_deg').value))
                rows.append({'leaf_id':int(candidate.leaf_id), **result})
                if result['reachable']:
                    orientation = np.asarray(result['approach_pose'])[:3,:3]
                    reachable.append(replace(candidate, orientation=orientation))
                    for row in self._report['targets']:
                        if row['leaf_id'] == candidate.leaf_id:
                            row['selected_orientation_matrix'] = orientation.tolist()
                            row['orientation_error_deg'] = result['orientation_error_deg']
            self._trace_candidates = reachable
            return reachable
        # 包装调用以记录原函数的两个检查，不另行求解或改变选择顺序。
        class TracedBackend:
            def pose_reachable(proxy, pose):
                index = len(rows)
                row = {'leaf_id':int(candidates[index//2].leaf_id),
                       'pose':'approach' if index%2 == 0 else 'contact',
                       'position_m':pose[:3,3].tolist()}
                try:
                    row['reachable'] = bool(backend.pose_reachable(pose))
                    return row['reachable']
                except Exception as exc:
                    row['error'] = str(exc)
                    raise
                finally:
                    rows.append(row)
        reachable = super()._keep_ik_reachable_candidates(TracedBackend(), candidates)
        self._trace_candidates = reachable
        return reachable

    def _publish_debug_targets(self, stamp, start_position, targets):
        self._report['stage'] = 'collision'
        trace = collision_trace(self._trace_candidates,
            int(self.get_parameter('target_count').value), start_position,
            float(self.get_parameter('tool_clearance_radius_m').value))
        trace['stage_status'] = 'evaluated' if trace['attempts'] else 'not_evaluated'
        trace['input_candidates_after_ik'] = len(self._trace_candidates)
        trace['actual_selected_ids'] = [int(t.leaf_id) for t in targets]
        trace['matches_original_selection'] = trace['selected_ids'] == trace['actual_selected_ids']
        self._report['collision'] = trace
        self._report['start_position_m'] = start_position.tolist()
        record_report('control', self._report, self.get_logger())
        return super()._publish_debug_targets(stamp, start_position, targets)

    def _execute_measurements(self, backend, targets):
        """空列表不是测量任务，不执行原节点入口处的张开夹爪动作。

        保留原规划产生的 target_shortfall 和 planned_targets=0 返回值；
        非空目标仍完全沿用原节点的测量顺序。
        """
        if not targets:
            self.get_logger().warning(
                'No valid selected targets: skipping gripper and arm execution. '
                'YOLO debug preview does not publish measurement targets.'
            )
            return 0
        self._report['stage'] = 'execution'
        if self.get_parameter('spad_mode').value:
            self._report.update(tool_mode='spad_alignment',aligned_targets=0,
                                measurement_performed=False,collision_scope='TCP corridor only, not full SPAD body')
            for target in targets:
                a=target.pose_matrix(target.approach_position)
                c=target.pose_matrix(target.contact_position)
                for operation,pose in [('approach',a),('side_insert',c),('retreat',a)]:
                    self._report['execution']={'operation':operation,'target_pose':pose.tolist()}
                    record_report('control',self._report,self.get_logger())
                    if operation=='approach':backend.move_pose(pose)
                    else:backend.move_cartesian_to_pose(pose)
                    if operation=='side_insert':time.sleep(float(self.get_parameter('gripper_hold_s').value))
                self._report['aligned_targets']+=1
            return 0  # 对准验证不是测量，不伪造 measured_targets。
        owner = self
        class ExecutionTrace:
            def __getattr__(proxy, name):
                operation = getattr(backend, name)
                def call(*args, **kwargs):
                    owner._report['execution'] = {'operation':name, 'status':'running'}
                    if args and name in ('move_pose', 'move_cartesian_to_pose'):
                        owner._report['execution']['target_pose'] = args[0].tolist()
                    record_report('control', owner._report, owner.get_logger())
                    try:
                        result = operation(*args, **kwargs)
                        owner._report['execution']['status'] = 'completed'
                        return result
                    except Exception as exc:
                        owner._report['execution']['status'] = 'failed'
                        owner._report['execution']['error'] = str(exc)
                        raise
                return call
        return super()._execute_measurements(ExecutionTrace(), targets)

    def _get_backend(self):
        if self._backend is None:
            self._backend=SimBackend(os.path.join(os.environ['WIDOWX_RUN_DIR'],'rpc.sock'))
        return self._backend

def main():
    rclpy.init(); node=SimulationControlNode()
    try: rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__=='__main__': main()
