"""On-demand Azure Kinect image segmentation and leaf point-cloud extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from wodowx_interfaces.msg import LeafDetection, LeafDetectionArray

from wodowx_control.geometry import (
    decode_depth_buffer,
    depth_mask_to_points,
    fit_leaf_plane,
)


class LeafPerceptionNode(Node):
    """Run YOLO only when the workflow requests a synchronized RGB-D frame."""

    def __init__(self) -> None:
        super().__init__("leaf_perception")
        self.declare_parameter("color_topic", "/rgb/image_raw")
        self.declare_parameter("depth_topic", "/depth_to_rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/depth_to_rgb/camera_info")
        self.declare_parameter("detections_topic", "/perception/leaves")
        self.declare_parameter(
            "debug_camera_image_topic", "/perception/debug_camera_image"
        )
        self.declare_parameter("debug_image_topic", "/perception/debug_image")
        self.declare_parameter("debug_confidence_threshold", 0.05)
        self.declare_parameter("debug_image_republish_period_s", 1.0)
        self.declare_parameter(
            "model_path",
            "/home/user/workspace/lwx/perception/runs/leaf_seg-2/weights/best.pt",
        )
        self.declare_parameter("target_label", "leaf")
        self.declare_parameter("confidence_threshold", 0.4)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("inference_device", "cpu")
        self.declare_parameter("min_depth_m", 0.12)
        self.declare_parameter("max_depth_m", 1.5)
        self.declare_parameter("min_points", 80)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("sync_slop_s", 0.08)
        self.declare_parameter("debug_enabled", True)

        confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        debug_confidence_threshold = float(
            self.get_parameter("debug_confidence_threshold").value
        )
        iou_threshold = float(self.get_parameter("iou_threshold").value)
        min_depth_m = float(self.get_parameter("min_depth_m").value)
        max_depth_m = float(self.get_parameter("max_depth_m").value)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if not 0.0 <= debug_confidence_threshold <= 1.0:
            raise ValueError("debug_confidence_threshold must be in [0, 1].")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1].")
        if min_depth_m < 0.0 or max_depth_m <= min_depth_m:
            raise ValueError("Depth limits must satisfy 0 <= min_depth_m < max_depth_m.")
        if int(self.get_parameter("min_points").value) < 3:
            raise ValueError("min_points must be at least 3.")
        if int(self.get_parameter("point_stride").value) < 1:
            raise ValueError("point_stride must be at least 1.")
        if int(self.get_parameter("image_size").value) < 1:
            raise ValueError("image_size must be positive.")
        if float(self.get_parameter("sync_slop_s").value) < 0.0:
            raise ValueError("sync_slop_s cannot be negative.")
        if float(self.get_parameter("debug_image_republish_period_s").value) <= 0.0:
            raise ValueError("debug_image_republish_period_s must be positive.")

        model_path = Path(self.get_parameter("model_path").value).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO segmentation model not found: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required by leaf_perception; install it in the ROS "
                "Python environment before starting this node."
            ) from exc
        self._model = YOLO(str(model_path))
        self._latest_color_msg: Image | None = None
        self._latest_frame: tuple[Image, Image, CameraInfo] | None = None
        self._latest_debug_image: Image | None = None
        self._sequence = 0
        self._target_label = str(self.get_parameter("target_label").value).lower()
        self._debug_enabled = bool(self.get_parameter("debug_enabled").value)

        output_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._detections_pub = self.create_publisher(
            LeafDetectionArray,
            str(self.get_parameter("detections_topic").value),
            output_qos,
        )
        self._debug_camera_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("debug_camera_image_topic").value),
            1,
        )
        self._debug_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("debug_image_topic").value),
            1,
        )
        self._markers_pub = self.create_publisher(
            MarkerArray, "/perception/debug_markers", 1
        )
        self._detect_service = self.create_service(
            Trigger, "/perception/detect", self._detect_callback
        )
        self._debug_detect_service = self.create_service(
            Trigger, "/perception/debug_detect", self._debug_detect_callback
        )

        sensor_qos = QoSProfile(
            depth=5, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self._camera_debug_sub = self.create_subscription(
            Image,
            str(self.get_parameter("color_topic").value),
            self._on_camera_image,
            sensor_qos,
        )
        self._color_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("color_topic").value),
            qos_profile=sensor_qos,
        )
        self._depth_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("depth_topic").value),
            qos_profile=sensor_qos,
        )
        self._info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            qos_profile=sensor_qos,
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub, self._info_sub],
            queue_size=10,
            slop=float(self.get_parameter("sync_slop_s").value),
        )
        self._sync.registerCallback(self._on_synchronized_frame)
        self._debug_image_timer = self.create_timer(
            float(self.get_parameter("debug_image_republish_period_s").value),
            self._republish_debug_image,
        )
        self.get_logger().info(
            f"Loaded model {model_path}; services: /perception/detect and "
            "/perception/debug_detect."
        )

    def _on_camera_image(self, color: Image) -> None:
        """Publish a live preview to verify the color-camera input independently."""
        self._latest_color_msg = color
        if not self._debug_enabled:
            return
        try:
            image = self._decode_color(color)
        except ValueError as exc:
            self.get_logger().error(
                f"Cannot decode camera preview: {exc}",
                throttle_duration_sec=5,
            )
            return
        self._debug_camera_image_pub.publish(
            self._make_bgr_image_message(image, color.header)
        )

    def _on_synchronized_frame(
        self, color: Image, depth: Image, camera_info: CameraInfo
    ) -> None:
        """Cache the newest aligned color/depth pair and its camera intrinsics."""
        self._latest_frame = (color, depth, camera_info)

    def _republish_debug_image(self) -> None:
        """Keep the latest annotated frame available to late-starting viewers."""
        if self._debug_enabled and self._latest_debug_image is not None:
            self._debug_image_pub.publish(self._latest_debug_image)

    def _detect_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        return self._run_detection(response)

    def _debug_detect_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Run a low-threshold RGB-only inference without publishing targets."""
        try:
            return self._process_debug_inference(response)
        except (KeyError, OSError, RuntimeError, ValueError, cv2.error) as exc:
            response.success = False
            response.message = f"Debug inference failed: {exc}"
            self.get_logger().error(response.message)
            return response

    def _run_detection(
        self, response: Trigger.Response
    ) -> Trigger.Response:
        try:
            return self._process_detection_request(response)
        except (KeyError, OSError, RuntimeError, ValueError, cv2.error) as exc:
            response.success = False
            response.message = f"Perception failed: {exc}"
            self.get_logger().error(response.message)
            return response

    def _process_debug_inference(
        self, response: Trigger.Response
    ) -> Trigger.Response:
        """Annotate YOLO results from the latest RGB frame, independent of depth."""
        if self._latest_color_msg is None:
            response.success = False
            response.message = "waiting for color camera image"
            return response

        color_msg = self._latest_color_msg
        color = self._decode_color(color_msg)
        confidence_threshold = float(
            self.get_parameter("debug_confidence_threshold").value
        )
        result = self._model.predict(
            source=color,
            imgsz=int(self.get_parameter("image_size").value),
            conf=confidence_threshold,
            iou=float(self.get_parameter("iou_threshold").value),
            device=str(self.get_parameter("inference_device").value),
            verbose=False,
        )[0]
        debug_image = color.copy()
        prediction_count = 0
        leaf_prediction_count = 0
        if result.boxes is not None:
            prediction_count = len(result.boxes)
            class_ids = result.boxes.cls.int().cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            masks = (
                result.masks.data.cpu().numpy()
                if result.masks is not None
                else None
            )
            for instance_id, (class_id, confidence, box) in enumerate(
                zip(class_ids, confidences, boxes)
            ):
                label = self._class_name(result.names, int(class_id))
                is_target_label = label.lower() == self._target_label
                leaf_prediction_count += int(is_target_label)
                mask = None
                if masks is not None and instance_id < len(masks):
                    mask = cv2.resize(
                        masks[instance_id],
                        (color.shape[1], color.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0.5
                self._draw_prediction(
                    debug_image,
                    mask,
                    box,
                    label,
                    instance_id,
                    float(confidence),
                    (0, 255, 0) if is_target_label else (0, 165, 255),
                    "target class" if is_target_label else "ignored class",
                )

        summary = (
            f"YOLO RGB debug: {prediction_count} predictions "
            f"(conf >= {confidence_threshold:.2f}); "
            f"{self._target_label} candidates: {leaf_prediction_count}"
        )
        if prediction_count == 0:
            summary += " - no model detections at this threshold"
        cv2.rectangle(debug_image, (0, 0), (debug_image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            debug_image,
            summary,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if self._debug_enabled:
            debug_msg = self._make_bgr_image_message(
                debug_image, color_msg.header
            )
            self._latest_debug_image = debug_msg
            self._debug_image_pub.publish(debug_msg)

        response.success = True
        response.message = (
            f"{summary}. Production confidence threshold: "
            f"{float(self.get_parameter('confidence_threshold').value):.2f}; "
            "control detections were not published."
        )
        self.get_logger().info(response.message)
        return response

    def _process_detection_request(
        self, response: Trigger.Response
    ) -> Trigger.Response:
        if self._latest_frame is None:
            response.success = False
            response.message = "waiting for synchronized camera frame"
            return response

        color_msg, depth_msg, camera_info = self._latest_frame
        color = self._decode_color(color_msg)
        depth_raw = self._decode_depth(depth_msg)
        depth_m = self._depth_to_meters(depth_raw, depth_msg.encoding)
        if depth_m.shape != color.shape[:2]:
            raise ValueError(
                "Depth must be aligned to the color image; configure Azure Kinect "
                "depth_to_rgb/image_raw."
            )

        confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        results = self._model.predict(
            source=color,
            imgsz=int(self.get_parameter("image_size").value),
            conf=confidence_threshold,
            iou=float(self.get_parameter("iou_threshold").value),
            device=str(self.get_parameter("inference_device").value),
            verbose=False,
        )
        result = results[0]
        output = LeafDetectionArray()
        output.header = depth_msg.header
        self._sequence += 1
        output.sequence = self._sequence
        debug_image = color.copy()
        markers = MarkerArray()
        model_prediction_count = 0
        leaf_prediction_count = 0
        valid_leaf_count = 0

        if result.boxes is not None:
            boxes = result.boxes
            model_prediction_count = len(boxes)
            class_ids = result.boxes.cls.int().cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            masks = (
                result.masks.data.cpu().numpy()
                if result.masks is not None
                else None
            )
            names = result.names
            min_points = int(self.get_parameter("min_points").value)
            stride = int(self.get_parameter("point_stride").value)
            min_depth = float(self.get_parameter("min_depth_m").value)
            max_depth = float(self.get_parameter("max_depth_m").value)
            fx, fy, cx, cy = self._camera_intrinsics(camera_info)

            for detection_id, (class_id, confidence, box) in enumerate(
                zip(class_ids, confidences, boxes_xyxy)
            ):
                label = self._class_name(names, int(class_id))
                is_target_label = label.lower() == self._target_label
                mask = None
                if masks is not None and detection_id < len(masks):
                    resized_mask = cv2.resize(
                        masks[detection_id],
                        (color.shape[1], color.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask = resized_mask > 0.5

                detail = "ignored class"
                color_bgr = (0, 165, 255)
                if is_target_label:
                    leaf_prediction_count += 1
                    color_bgr = (0, 255, 255)
                    if mask is None:
                        detail = "no segmentation mask"
                    else:
                        points = depth_mask_to_points(
                            depth_m,
                            mask,
                            (fx, fy, cx, cy),
                            min_depth,
                            max_depth,
                            stride,
                            np.asarray(camera_info.d, dtype=np.float64),
                        )
                        if points.shape[0] < min_points:
                            detail = (
                                f"depth points {points.shape[0]} < {min_points}"
                            )
                        else:
                            try:
                                center, normal, _tangent = fit_leaf_plane(points)
                            except ValueError as exc:
                                detail = f"PCA failed: {exc}"
                            else:
                                detail = f"3D valid, {points.shape[0]} points"
                                color_bgr = (0, 255, 0)
                                valid_leaf_count += 1
                                leaf = LeafDetection()
                                leaf.id = detection_id
                                leaf.label = label
                                leaf.confidence = float(confidence)
                                leaf.center.x, leaf.center.y, leaf.center.z = map(
                                    float, center
                                )
                                leaf.normal.x, leaf.normal.y, leaf.normal.z = map(
                                    float, normal
                                )
                                leaf.cloud = point_cloud2.create_cloud_xyz32(
                                    depth_msg.header, points.tolist()
                                )
                                output.detections.append(leaf)
                                markers.markers.extend(
                                    self._make_markers(
                                        leaf,
                                        depth_msg.header,
                                        len(markers.markers),
                                    )
                                )

                self._draw_prediction(
                    debug_image,
                    mask,
                    box,
                    label,
                    detection_id,
                    float(confidence),
                    color_bgr,
                    detail,
                )

        summary = (
            f"YOLO predictions: {model_prediction_count} "
            f"(conf >= {confidence_threshold:.2f}); "
            f"{self._target_label} candidates: {leaf_prediction_count}; "
            f"valid 3D leaves: {valid_leaf_count}"
        )
        if model_prediction_count == 0:
            summary += " - no model detections at this threshold"
        cv2.rectangle(debug_image, (0, 0), (debug_image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            debug_image,
            summary,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        self._detections_pub.publish(output)
        if self._debug_enabled:
            debug_msg = self._make_bgr_image_message(debug_image, color_msg.header)
            self._latest_debug_image = debug_msg
            self._debug_image_pub.publish(debug_msg)
            self._markers_pub.publish(markers)
        response.success = True
        response.message = (
            f"sequence={output.sequence}, model_predictions={model_prediction_count}, "
            f"leaf_candidates={leaf_prediction_count}, "
            f"valid_leaf_instances={len(output.detections)}"
        )
        self.get_logger().info(response.message)
        return response

    @staticmethod
    def _make_bgr_image_message(image: np.ndarray, header: object) -> Image:
        """Encode a contiguous OpenCV BGR image as a ROS Image message."""
        image = np.ascontiguousarray(image)
        message = Image()
        message.header = header
        message.height, message.width = image.shape[:2]
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = message.width * 3
        message.data = image.tobytes()
        return message

    @staticmethod
    def _draw_prediction(
        image: np.ndarray,
        mask: np.ndarray | None,
        box: np.ndarray,
        label: str,
        instance_id: int,
        confidence: float,
        color: tuple[int, int, int],
        detail: str,
    ) -> None:
        """Draw a prediction and the reason it was accepted or discarded."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if mask is not None:
            overlay = image.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, 0.32, image, 0.68, 0.0, dst=image)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {instance_id}: {confidence:.2f} - {detail}"
        text_y = max(52, y1 - 6)
        cv2.putText(
            image,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _decode_color(message: Image) -> np.ndarray:
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "mono8": 1,
        }
        channels = channels_by_encoding.get(message.encoding)
        if channels is None:
            raise ValueError(
                f"Unsupported color encoding {message.encoding!r}; expected "
                "bgr8, rgb8, bgra8, rgba8, or mono8."
            )
        row = np.frombuffer(message.data, dtype=np.uint8)
        if message.step < message.width * channels or row.size < message.height * message.step:
            raise ValueError("Color Image message has an invalid step or data length.")
        pixels = row[: message.height * message.step].reshape(
            message.height, message.step
        )
        image = pixels[:, : message.width * channels].reshape(
            message.height, message.width, channels
        )
        if message.encoding == "bgr8":
            return np.ascontiguousarray(image)
        if message.encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if message.encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if message.encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _decode_depth(message: Image) -> np.ndarray:
        return decode_depth_buffer(
            message.data,
            message.width,
            message.height,
            message.step,
            message.encoding,
            message.is_bigendian,
        )

    @staticmethod
    def _depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
        if encoding in ("16UC1", "mono16"):
            return depth.astype(np.float32) * 0.001
        if encoding == "32FC1":
            return depth.astype(np.float32)
        raise ValueError(
            f"Unsupported depth encoding {encoding!r}; expected 16UC1 or 32FC1."
        )

    @staticmethod
    def _camera_intrinsics(info: CameraInfo) -> tuple[float, float, float, float]:
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("CameraInfo does not contain valid pinhole intrinsics.")
        return fx, fy, cx, cy

    @staticmethod
    def _class_name(names: object, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names[class_id])
        return str(names[class_id])  # Ultralytics also supports a list of class names.

    @staticmethod
    def _make_markers(
        leaf: LeafDetection, header: object, first_id: int
    ) -> list[Marker]:
        center = Marker()
        center.header = header
        center.ns = "leaf_centers"
        center.id = first_id
        center.type = Marker.SPHERE
        center.action = Marker.ADD
        center.pose.position = leaf.center
        center.pose.orientation.w = 1.0
        center.scale.x = center.scale.y = center.scale.z = 0.012
        center.color.r, center.color.g, center.color.a = 0.1, 1.0, 1.0

        normal = Marker()
        normal.header = header
        normal.ns = "leaf_normals"
        normal.id = first_id + 1
        normal.type = Marker.ARROW
        normal.action = Marker.ADD
        normal.pose.orientation.w = 1.0
        normal.scale.x, normal.scale.y, normal.scale.z = 0.004, 0.008, 0.012
        normal.color.r, normal.color.g, normal.color.a = 1.0, 0.2, 1.0
        from geometry_msgs.msg import Point

        start = Point(x=leaf.center.x, y=leaf.center.y, z=leaf.center.z)
        end = Point(
            x=leaf.center.x + 0.08 * leaf.normal.x,
            y=leaf.center.y + 0.08 * leaf.normal.y,
            z=leaf.center.z + 0.08 * leaf.normal.z,
        )
        normal.points = [start, end]
        return [center, normal]


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LeafPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
