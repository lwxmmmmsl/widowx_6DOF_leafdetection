"""Transform, screen, and measure detected leaves using WidowX EE poses."""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from visualization_msgs.msg import Marker, MarkerArray
from wodowx_interfaces.msg import LeafDetectionArray
from wodowx_interfaces.srv import MeasureLeaves

from wodowx_control.target_planning import (
    LeafTarget,
    choose_collision_free_targets,
    make_leaf_target,
    quaternion_to_rotation,
)
from wodowx_control.geometry import xyz_from_point_cloud_reading


class InterbotixArmBackend:
    """Thin adapter around the supported Interbotix Python ROS interface."""

    def __init__(
        self,
        robot_model: str,
        robot_name: str,
        moving_time_s: float,
        accel_time_s: float,
    ) -> None:
        from interbotix_common_modules.common_robot.robot import (
            robot_startup,
        )
        from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

        self._robot = InterbotixManipulatorXS(
            robot_model=robot_model,
            group_name="arm",
            gripper_name="gripper",
            robot_name=robot_name,
            moving_time=moving_time_s,
            accel_time=accel_time_s,
        )
        # Follow the initialization sequence used by Interbotix's bartender demo:
        # construct the manipulator, then start its ROS executor.
        robot_startup()
        self._shutdown = False

    def current_position(self) -> np.ndarray:
        """Return the measured EE position in the robot base frame."""
        pose = np.asarray(self._robot.arm.get_ee_pose(), dtype=np.float64)
        return pose[:3, 3].copy()

    def move_pose(self, pose: np.ndarray) -> None:
        _joint_solution, success = self._robot.arm.set_ee_pose_matrix(
            T_sd=pose,
            execute=True,
            moving_time=self._robot.arm.moving_time,
            accel_time=self._robot.arm.accel_time,
            blocking=True,
        )
        if not success:
            raise RuntimeError("Interbotix rejected the requested EE pose (IK/limits).")

    def pose_reachable(self, pose: np.ndarray) -> bool:
        """Check IK and joint limits without sending a motion command."""
        _joint_solution, success = self._robot.arm.set_ee_pose_matrix(
            T_sd=pose,
            execute=False,
            moving_time=self._robot.arm.moving_time,
            accel_time=self._robot.arm.accel_time,
            blocking=False,
        )
        return bool(success)

    def move_cartesian_to_pose(self, pose: np.ndarray) -> None:
        """Move in a straight line to a same-orientation pose in base coordinates."""
        current_pose = np.asarray(self._robot.arm.T_sb, dtype=np.float64)
        if not np.allclose(current_pose[:3, :3], pose[:3, :3], atol=1e-3):
            raise ValueError(
                "Cartesian leaf approach requires the current and target "
                "end-effector orientations to match."
            )
        yaw = float(np.arctan2(current_pose[1, 0], current_pose[0, 0]))
        yaw_rotation = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        displacement = yaw_rotation.T @ (
            pose[:3, 3] - current_pose[:3, 3]
        )
        moving_time = float(self._robot.arm.moving_time)
        waypoint_time = min(0.2, moving_time / 2.0)
        success = self._robot.arm.set_ee_cartesian_trajectory(
            x=float(displacement[0]),
            y=float(displacement[1]),
            z=float(displacement[2]),
            moving_time=moving_time,
            wp_moving_time=waypoint_time,
            wp_accel_time=waypoint_time / 2.0,
            wp_period=0.05,
        )
        if not success:
            raise RuntimeError(
                "Interbotix could not plan the straight EE Cartesian approach."
            )

    def open_gripper(self) -> None:
        self._robot.gripper.release()

    def close_gripper(self) -> None:
        self._robot.gripper.grasp()

    def shutdown(self) -> None:
        if not self._shutdown:
            from interbotix_common_modules.common_robot.robot import robot_shutdown

            robot_shutdown()
            self._shutdown = True


class LeafControlNode(Node):
    """Consume a leaf array, remove unsafe nominal paths, and execute measurements."""

    def __init__(self) -> None:
        super().__init__("leaf_control")
        self.declare_parameter("detections_topic", "/perception/leaves")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("transform_timeout_s", 2.0)
        self.declare_parameter("target_count", 3)
        self.declare_parameter("approach_distance_m", 0.08)
        self.declare_parameter("tool_clearance_radius_m", 0.035)
        self.declare_parameter("gripper_hold_s", 3.0)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("robot_model", "wx250s")
        self.declare_parameter("robot_name", "wx250s")
        self.declare_parameter("moving_time_s", 2.0)
        self.declare_parameter("accel_time_s", 0.3)
        self.declare_parameter("debug_start_position_m", [0.25, 0.0, 0.25])
        self.declare_parameter(
            "target_shortfall_topic", "/control/target_shortfall"
        )

        self._base_frame = str(self.get_parameter("base_frame").value)
        target_count = int(self.get_parameter("target_count").value)
        approach_distance = float(
            self.get_parameter("approach_distance_m").value
        )
        clearance_radius = float(
            self.get_parameter("tool_clearance_radius_m").value
        )
        hold_time = float(self.get_parameter("gripper_hold_s").value)
        transform_timeout = float(
            self.get_parameter("transform_timeout_s").value
        )
        moving_time = float(self.get_parameter("moving_time_s").value)
        accel_time = float(self.get_parameter("accel_time_s").value)
        if target_count < 0:
            raise ValueError("target_count cannot be negative.")
        if approach_distance <= 0.0:
            raise ValueError("approach_distance_m must be positive.")
        if clearance_radius < 0.0 or hold_time < 0.0:
            raise ValueError("Clearance radius and gripper hold time cannot be negative.")
        if transform_timeout <= 0.0:
            raise ValueError("transform_timeout_s must be positive.")
        if moving_time <= 0.0 or accel_time < 0.0 or accel_time > moving_time / 2.0:
            raise ValueError(
                "Motion timing must satisfy moving_time_s > 0 and "
                "0 <= accel_time_s <= moving_time_s / 2."
            )
        if len(self.get_parameter("debug_start_position_m").value) != 3:
            raise ValueError("debug_start_position_m must contain three coordinates.")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._latest_detections: LeafDetectionArray | None = None
        self._backend: InterbotixArmBackend | None = None
        self._measurement_lock = threading.Lock()
        self._last_completed_sequence = 0
        self._target_shortfall_pub = self.create_publisher(
            Bool, str(self.get_parameter("target_shortfall_topic").value), 10
        )
        self._debug_targets_pub = self.create_publisher(
            MarkerArray, "/control/debug_targets", 1
        )
        detection_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._detections_sub = self.create_subscription(
            LeafDetectionArray,
            str(self.get_parameter("detections_topic").value),
            self._on_detections,
            detection_qos,
        )
        self._measure_service = self.create_service(
            MeasureLeaves, "/control/measure", self._measure_callback
        )
        self.get_logger().info(
            f"Control ready; dry_run={self.get_parameter('dry_run').value}, "
            f"base_frame={self._base_frame}"
        )

    def _on_detections(self, message: LeafDetectionArray) -> None:
        self._latest_detections = message

    def _measure_callback(
        self, request: MeasureLeaves.Request, response: MeasureLeaves.Response
    ) -> MeasureLeaves.Response:
        if not self._measurement_lock.acquire(blocking=False):
            response.success = False
            response.message = "A leaf measurement is already in progress."
            return response

        try:
            detections = self._latest_detections
            if detections is None or detections.sequence != request.sequence:
                response.success = False
                response.message = (
                    f"Requested perception sequence {request.sequence} is not available."
                )
                return response
            if request.sequence <= self._last_completed_sequence:
                response.success = False
                response.message = (
                    f"Perception sequence {request.sequence} was already processed."
                )
                return response

            dry_run = bool(self.get_parameter("dry_run").value)
            backend = None if dry_run else self._get_backend()
            if backend is not None:
                start_position = backend.current_position()
                self.get_logger().info(
                    "Starting target sequence from current EE position in "
                    f"{self._base_frame}: {start_position.round(3).tolist()} m"
                )
            else:
                start_position = np.asarray(
                    self.get_parameter("debug_start_position_m").value,
                    dtype=np.float64,
                )

            candidates = self._build_candidates(detections)
            if backend is not None:
                candidates = self._keep_ik_reachable_candidates(
                    backend, candidates
                )
            requested = int(self.get_parameter("target_count").value)
            selected = choose_collision_free_targets(
                candidates,
                requested,
                start_position,
                float(self.get_parameter("tool_clearance_radius_m").value),
            )
            self._publish_debug_targets(
                detections.header.stamp, start_position, selected
            )
            shortfall = len(selected) < requested
            self._target_shortfall_pub.publish(Bool(data=shortfall))
            if shortfall:
                self.get_logger().warning(
                    f"Only {len(selected)} collision-free targets are available; "
                    f"{requested} requested. Continuing with available leaves."
                )

            measured = 0
            if dry_run:
                for target in selected:
                    self.get_logger().info(
                        f"DRY RUN leaf {target.leaf_id}: "
                        f"approach={target.approach_position.tolist()}, "
                        f"contact={target.contact_position.tolist()}"
                    )
            else:
                if backend is None:
                    raise RuntimeError("Physical control backend was not initialized.")
                measured = self._execute_measurements(backend, selected)

            self._last_completed_sequence = request.sequence
            response.success = True
            response.target_shortfall = shortfall
            response.planned_targets = len(selected)
            response.measured_targets = measured
            response.message = (
                f"sequence={request.sequence}, planned={len(selected)}, "
                f"measured={measured}, target_shortfall={shortfall}"
            )
            self.get_logger().info(response.message)
            return response
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
            ImportError,
            ValueError,
            RuntimeError,
        ) as exc:
            response.success = False
            response.message = f"Measurement planning/execution failed: {exc}"
            self.get_logger().error(response.message)
            return response
        finally:
            self._measurement_lock.release()

    def _build_candidates(
        self, detections: LeafDetectionArray
    ) -> list[LeafTarget]:
        candidates: list[LeafTarget] = []
        timeout = Duration(
            seconds=float(self.get_parameter("transform_timeout_s").value)
        )
        stamp = Time.from_msg(detections.header.stamp)
        approach_distance = float(
            self.get_parameter("approach_distance_m").value
        )

        for leaf in detections.detections:
            if leaf.label.lower() != "leaf":
                continue
            cloud_frame = leaf.cloud.header.frame_id or detections.header.frame_id
            transform = self._tf_buffer.lookup_transform(
                self._base_frame, cloud_frame, stamp, timeout=timeout
            )
            translation = transform.transform.translation
            rotation_msg = transform.transform.rotation
            rotation = quaternion_to_rotation(
                (rotation_msg.x, rotation_msg.y, rotation_msg.z, rotation_msg.w)
            )
            offset = np.array(
                [translation.x, translation.y, translation.z], dtype=np.float64
            )
            points_camera = xyz_from_point_cloud_reading(
                point_cloud2.read_points(
                    leaf.cloud,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            )
            if points_camera.ndim != 2 or points_camera.shape[1] != 3:
                raise ValueError(f"Leaf {leaf.id} has an invalid or empty point cloud.")
            points_base = points_camera @ rotation.T + offset
            center_camera = np.array(
                [leaf.center.x, leaf.center.y, leaf.center.z], dtype=np.float64
            )
            normal_camera = np.array(
                [leaf.normal.x, leaf.normal.y, leaf.normal.z], dtype=np.float64
            )
            center_base = rotation @ center_camera + offset
            normal_base = rotation @ normal_camera
            target = make_leaf_target(
                leaf.id,
                leaf.confidence,
                points_base,
                center_base,
                normal_base,
                approach_distance,
            )
            self.get_logger().info(
                f"Leaf {leaf.id} center in {self._base_frame}: "
                f"x={center_base[0]:.3f}, y={center_base[1]:.3f}, "
                f"z={center_base[2]:.3f} m "
                f"(confidence={leaf.confidence:.3f}, source_frame={cloud_frame})"
            )
            self.get_logger().info(
                f"Leaf {leaf.id} EE targets in {self._base_frame}: "
                f"approach={target.approach_position.round(3).tolist()} m, "
                f"contact={target.contact_position.round(3).tolist()} m, "
                f"normal={normal_base.round(3).tolist()}"
            )
            candidates.append(target)
        return candidates

    def _publish_debug_targets(
        self,
        stamp,
        start_position: np.ndarray,
        targets: list[LeafTarget],
    ) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._base_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for target in targets:
            marker_id = target.leaf_id * 3
            contact = Marker()
            contact.header.frame_id = self._base_frame
            contact.header.stamp = stamp
            contact.ns = "selected_leaf_contacts"
            contact.id = marker_id
            contact.type = Marker.SPHERE
            contact.action = Marker.ADD
            contact.pose.position = self._point(target.contact_position)
            contact.pose.orientation.w = 1.0
            contact.scale.x = contact.scale.y = contact.scale.z = 0.016
            contact.color.r, contact.color.g, contact.color.a = 1.0, 0.35, 1.0

            approach = Marker()
            approach.header = contact.header
            approach.ns = "selected_leaf_approaches"
            approach.id = marker_id + 1
            approach.type = Marker.SPHERE
            approach.action = Marker.ADD
            approach.pose.position = self._point(target.approach_position)
            approach.pose.orientation.w = 1.0
            approach.scale.x = approach.scale.y = approach.scale.z = 0.012
            approach.color.g, approach.color.b, approach.color.a = 1.0, 1.0, 1.0

            path = Marker()
            path.header = contact.header
            path.ns = "selected_leaf_ee_paths"
            path.id = marker_id + 2
            path.type = Marker.LINE_STRIP
            path.action = Marker.ADD
            path.scale.x = 0.004
            path.color.r, path.color.g, path.color.b, path.color.a = (
                0.2,
                0.55,
                1.0,
                1.0,
            )
            path.points = [
                self._point(start_position),
                self._point(target.approach_position),
                self._point(target.contact_position),
                self._point(target.approach_position),
                self._point(start_position),
            ]
            markers.markers.extend((contact, approach, path))

        self._debug_targets_pub.publish(markers)

    def _keep_ik_reachable_candidates(
        self,
        backend: InterbotixArmBackend,
        candidates: list[LeafTarget],
    ) -> list[LeafTarget]:
        reachable: list[LeafTarget] = []
        for target in candidates:
            approach_pose = target.pose_matrix(target.approach_position)
            contact_pose = target.pose_matrix(target.contact_position)
            approach_ok = backend.pose_reachable(approach_pose)
            contact_ok = backend.pose_reachable(contact_pose)
            if approach_ok and contact_ok:
                reachable.append(target)
            else:
                self.get_logger().warning(
                    f"Leaf {target.leaf_id} rejected by IK: "
                    f"approach_reachable={approach_ok} at "
                    f"{target.approach_position.round(3).tolist()} m, "
                    f"contact_reachable={contact_ok} at "
                    f"{target.contact_position.round(3).tolist()} m. "
                    "Positions use base_frame and poses include the leaf-grasp "
                    "orientation."
                )
        return reachable

    @staticmethod
    def _point(position: np.ndarray) -> Point:
        return Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))

    def _execute_measurements(
        self, backend: InterbotixArmBackend, targets: list[LeafTarget]
    ) -> int:
        hold_time = float(self.get_parameter("gripper_hold_s").value)
        measured = 0
        backend.open_gripper()
        for target in targets:
            contact_reached = False
            try:
                backend.move_pose(
                    target.pose_matrix(target.approach_position)
                )
                backend.move_cartesian_to_pose(
                    target.pose_matrix(target.contact_position)
                )
                contact_reached = True
                backend.close_gripper()
                time.sleep(hold_time)
                backend.open_gripper(                )
                backend.move_cartesian_to_pose(
                    target.pose_matrix(target.approach_position)
                )
                measured += 1
            except RuntimeError:
                if contact_reached:
                    backend.open_gripper()
                    backend.move_pose(
                        target.pose_matrix(target.approach_position)
                    )
                raise
        return measured

    def _get_backend(self) -> InterbotixArmBackend:
        if self._backend is None:
            self._backend = InterbotixArmBackend(
                robot_model=str(self.get_parameter("robot_model").value),
                robot_name=str(self.get_parameter("robot_name").value),
                moving_time_s=float(self.get_parameter("moving_time_s").value),
                accel_time_s=float(self.get_parameter("accel_time_s").value),
            )
        return self._backend

    def destroy_node(self) -> bool:
        if self._backend is not None:
            self._backend.shutdown()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LeafControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
