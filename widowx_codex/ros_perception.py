"""原检测节点的诊断适配：复用推理/深度/PCA，只截取已有淘汰原因。"""
import sys
from .common import ROOT
sys.path.insert(0, str(ROOT/'reference/wodowx_control'))
import rclpy
from wodowx_control.perception_node import LeafPerceptionNode
from .diagnostics import record_report


class DiagnosticPerceptionNode(LeafPerceptionNode):
    def _process_detection_request(self, response):
        self._diagnostic_rows = []
        report = {'stage':'perception', 'confidence_threshold':float(self.get_parameter('confidence_threshold').value)}
        before = self._sequence
        try:
            result = super()._process_detection_request(response)
            report.update(success=bool(result.success), message=result.message,
                          sequence=int(self._sequence) if self._sequence != before else None,
                          predictions=self._diagnostic_rows)
            rows = self._diagnostic_rows
            report['reason'] = (
                'camera_not_ready' if not result.success else
                'yolo_no_predictions' if not rows else
                'no_leaf_class' if not any(r['label'].lower() == self._target_label for r in rows) else
                'ready' if any(r['result'].startswith('3D valid') for r in rows) else
                'leaf_geometry_rejected')
            return result
        except Exception as exc:
            report.update(success=False, reason='perception_exception', error=str(exc), sequence=None,
                          predictions=self._diagnostic_rows)
            raise
        finally:
            record_report('perception', report, self.get_logger())
            self._diagnostic_rows = None

    def _draw_prediction(self, image, mask, box, label, instance_id, confidence, color, detail):
        """原节点逐实例 detail 已包括非目标类、无 mask、深度不足、PCA 失败。"""
        rows = getattr(self, '_diagnostic_rows', None)
        if rows is not None:
            rows.append({'leaf_id':int(instance_id), 'label':label, 'confidence':float(confidence),
                         'box_xyxy':box.tolist(), 'mask_available':mask is not None, 'result':detail})
        return super()._draw_prediction(image, mask, box, label, instance_id, confidence, color, detail)


def main():
    rclpy.init()
    node = DiagnosticPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()
