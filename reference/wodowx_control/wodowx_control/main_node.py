"""Orchestrate on-demand perception, measurement, and completion reporting."""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from wodowx_interfaces.msg import LeafDetectionArray, WorkflowStatus
from wodowx_interfaces.srv import MeasureLeaves


class LeafWorkflowNode(Node):
    """Drive one capture-and-measure cycle, with a start service for repeat runs."""

    def __init__(self) -> None:
        super().__init__("leaf_workflow")
        self.declare_parameter("detections_topic", "/perception/leaves")
        self.declare_parameter("status_topic", "/workflow/status")
        self.declare_parameter("auto_start", True)
        self.declare_parameter("start_delay_s", 2.0)
        self.declare_parameter("camera_retry_period_s", 0.5)
        self.declare_parameter("max_camera_retries", 30)
        self.declare_parameter("perception_service", "/perception/detect")
        self.declare_parameter("control_service", "/control/measure")

        self._state = "idle"
        self._latest_detections: LeafDetectionArray | None = None
        self._sequence_before_capture = 0
        self._camera_retries = 0
        self._next_attempt_time = 0.0
        self._auto_start_at = time.monotonic() + float(
            self.get_parameter("start_delay_s").value
        )
        self._auto_start_pending = bool(self.get_parameter("auto_start").value)
        self._perception_client = self.create_client(
            Trigger, str(self.get_parameter("perception_service").value)
        )
        self._control_client = self.create_client(
            MeasureLeaves, str(self.get_parameter("control_service").value)
        )
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            WorkflowStatus,
            str(self.get_parameter("status_topic").value),
            status_qos,
        )
        detections_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._detections_sub = self.create_subscription(
            LeafDetectionArray,
            str(self.get_parameter("detections_topic").value),
            self._on_detections,
            detections_qos,
        )
        self._start_service = self.create_service(
            Trigger, "/workflow/start", self._start_callback
        )
        self._timer = self.create_timer(0.1, self._step)
        self.get_logger().info(
            "Workflow ready; start manually with ros2 service call "
            "/workflow/start std_srvs/srv/Trigger '{}' if auto_start is disabled."
        )

    def _start_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if self._state not in ("idle", "finished"):
            response.success = False
            response.message = f"Workflow is currently {self._state}."
            return response
        self._auto_start_pending = False
        self._begin_capture()
        response.success = True
        response.message = "Workflow capture requested."
        return response

    def _on_detections(self, message: LeafDetectionArray) -> None:
        self._latest_detections = message
        if (
            self._state == "waiting_for_detections"
            and message.sequence > self._sequence_before_capture
        ):
            self._request_measurement(message.sequence)

    def _step(self) -> None:
        now = time.monotonic()
        if (
            self._auto_start_pending
            and self._state == "idle"
            and now >= self._auto_start_at
        ):
            self._begin_capture()
        elif self._state == "waiting_for_services" and (
            self._perception_client.service_is_ready()
            and self._control_client.service_is_ready()
        ):
            self._begin_capture()
        elif self._state == "retry_wait" and now >= self._next_attempt_time:
            self._call_perception()

    def _begin_capture(self) -> None:
        self._auto_start_pending = False
        if not (
            self._perception_client.service_is_ready()
            and self._control_client.service_is_ready()
        ):
            self._state = "waiting_for_services"
            self.get_logger().info(
                "Waiting for perception and control services before capture."
            )
            return
        self._sequence_before_capture = (
            self._latest_detections.sequence if self._latest_detections else 0
        )
        self._camera_retries = 0
        self._state = "capturing"
        self._call_perception()

    def _call_perception(self) -> None:
        if not self._perception_client.service_is_ready():
            self._state = "waiting_for_services"
            return
        self._state = "capturing"
        self._perception_client.call_async(Trigger.Request()).add_done_callback(
            self._on_perception_response
        )

    def _on_perception_response(self, future) -> None:
        exception = future.exception()
        if exception is not None:
            self._finish(False, f"Perception service call failed: {exception}")
            return
        result = future.result()
        if result.success:
            self._state = "waiting_for_detections"
            if (
                self._latest_detections is not None
                and self._latest_detections.sequence > self._sequence_before_capture
            ):
                self._request_measurement(self._latest_detections.sequence)
            return

        if result.message == "waiting for synchronized camera frame":
            self._camera_retries += 1
            max_retries = int(self.get_parameter("max_camera_retries").value)
            if self._camera_retries <= max_retries:
                self.get_logger().warning(
                    "No synchronized camera frame yet; retrying perception "
                    f"({self._camera_retries}/{max_retries})."
                )
                self._state = "retry_wait"
                self._next_attempt_time = time.monotonic() + float(
                    self.get_parameter("camera_retry_period_s").value
                )
                return
        self._finish(False, f"Perception failed: {result.message}")

    def _request_measurement(self, sequence: int) -> None:
        if not self._control_client.service_is_ready():
            self._finish(False, "Control measurement service is unavailable.")
            return
        request = MeasureLeaves.Request()
        request.sequence = sequence
        self._state = "measuring"
        self._control_client.call_async(request).add_done_callback(
            self._on_control_response
        )

    def _on_control_response(self, future) -> None:
        exception = future.exception()
        if exception is not None:
            self._finish(False, f"Control service call failed: {exception}")
            return
        result = future.result()
        self._finish(
            result.success,
            result.message,
            result.target_shortfall,
            result.planned_targets,
            result.measured_targets,
        )

    def _finish(
        self,
        success: bool,
        message: str,
        target_shortfall: bool = False,
        planned_targets: int = 0,
        measured_targets: int = 0,
    ) -> None:
        self._state = "finished"
        self._auto_start_pending = False
        status = WorkflowStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.finished = True
        status.success = success
        status.target_shortfall = target_shortfall
        status.planned_targets = planned_targets
        status.measured_targets = measured_targets
        status.message = message
        self._status_pub.publish(status)
        log = self.get_logger().info if success else self.get_logger().error
        log(f"Workflow finished: {message}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LeafWorkflowNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
