"""Leaf target construction and conservative end-effector path filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LeafTarget:
    """A base-frame contact target and its approach pose."""

    leaf_id: int
    confidence: float
    points: np.ndarray
    contact_position: np.ndarray
    approach_position: np.ndarray
    orientation: np.ndarray

    def pose_matrix(self, position: np.ndarray) -> np.ndarray:
        """Create a homogeneous EE pose using the target orientation."""
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.orientation
        pose[:3, 3] = position
        return pose


def quaternion_to_rotation(
    quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Convert a normalized ROS-order quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-12:
        raise ValueError("Transform contains a zero-length quaternion.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_leaf_target(
    leaf_id: int,
    confidence: float,
    points_base: np.ndarray,
    center_base: np.ndarray,
    normal_base: np.ndarray,
    approach_distance_m: float,
) -> LeafTarget:
    """Choose the leaf point nearest the robot base and construct its EE frame.

    WidowX250s gripper jaws close along local +Y, so +Y is aligned with the
    leaf normal. The tool +Z axis is directed from the base toward the selected
    leaf edge, allowing the fingers to approach in the leaf plane.
    """
    if points_base.ndim != 2 or points_base.shape[1] != 3 or points_base.size == 0:
        raise ValueError("A target requires a non-empty Nx3 point cloud.")
    if approach_distance_m <= 0.0:
        raise ValueError("Approach distance must be positive.")

    normal = _unit_vector(normal_base)
    tangent = _dominant_tangent(points_base, center_base, normal)
    # "Nearest the base" is interpreted as minimum Euclidean distance from
    # the configured robot base origin, not from the camera or leaf centroid.
    contact = points_base[int(np.argmin(np.linalg.norm(points_base, axis=1)))]
    contact_in_plane = contact - np.dot(contact, normal) * normal
    if np.linalg.norm(contact_in_plane) < 1e-8:
        approach_axis = tangent
    else:
        approach_axis = _unit_vector(contact_in_plane)
    tool_z = approach_axis
    tool_y = normal
    tool_x = _unit_vector(np.cross(tool_y, tool_z))
    orientation = np.column_stack((tool_x, tool_y, tool_z))
    approach = contact - approach_axis * approach_distance_m
    return LeafTarget(
        leaf_id=leaf_id,
        confidence=float(confidence),
        points=points_base,
        contact_position=contact.astype(np.float64),
        approach_position=approach.astype(np.float64),
        orientation=orientation,
    )


def choose_collision_free_targets(
    candidates: list[LeafTarget],
    desired_count: int,
    start_position: np.ndarray,
    clearance_radius_m: float,
) -> list[LeafTarget]:
    """Greedily select targets along the planned, continuous EE sequence.

    The first candidate is checked from the arm's current position. Each later
    candidate is checked from the previous candidate's approach position,
    because execution retreats there after measurement instead of returning
    home. This is a point-cloud corridor prefilter, not full robot-link or
    MoveIt collision checking.
    """
    if desired_count < 0:
        raise ValueError("Desired target count cannot be negative.")
    if clearance_radius_m < 0.0:
        raise ValueError("Clearance radius cannot be negative.")
    if desired_count == 0:
        return []

    start = np.asarray(start_position, dtype=np.float64)
    if start.shape != (3,):
        raise ValueError("Start position must contain three coordinates.")

    remaining = list(candidates)
    selected: list[LeafTarget] = []
    while remaining and len(selected) < desired_count:
        ordered = sorted(
            remaining,
            key=lambda target: (
                float(np.linalg.norm(target.approach_position - start)),
                -target.confidence,
                target.leaf_id,
            ),
        )
        chosen: LeafTarget | None = None
        for candidate in ordered:
            obstacles = [
                other.points
                for other in candidates
                if other.leaf_id != candidate.leaf_id and other.points.size
            ]
            if _path_is_clear(
                start,
                candidate.approach_position,
                candidate.contact_position,
                obstacles,
                clearance_radius_m,
                candidate.points,
            ):
                chosen = candidate
                break
        if chosen is None:
            break
        selected.append(chosen)
        remaining = [
            candidate
            for candidate in remaining
            if candidate.leaf_id != chosen.leaf_id
        ]
        start = chosen.approach_position
    return selected


def _path_is_clear(
    start: np.ndarray,
    approach: np.ndarray,
    contact: np.ndarray,
    obstacle_clouds: list[np.ndarray],
    clearance_radius_m: float,
    target_cloud: np.ndarray,
) -> bool:
    approach_obstacles = [*obstacle_clouds]
    if target_cloud.size:
        approach_obstacles.append(target_cloud)
    if not _segment_clears_cloud(
        start, approach, approach_obstacles, clearance_radius_m
    ):
        return False
    return _segment_clears_cloud(
        approach, contact, obstacle_clouds, clearance_radius_m
    )


def _segment_clears_cloud(
    start: np.ndarray,
    end: np.ndarray,
    obstacle_points: np.ndarray | list[np.ndarray],
    clearance_radius_m: float,
) -> bool:
    if isinstance(obstacle_points, list):
        obstacle_points = [
            points for points in obstacle_points if points.size
        ]
        if not obstacle_points:
            return True
        obstacle_points = np.concatenate(obstacle_points, axis=0)
    if obstacle_points.size == 0:
        return True
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared < 1e-12:
        closest = np.broadcast_to(start, obstacle_points.shape)
    else:
        fractions = np.clip(
            ((obstacle_points - start) @ direction) / length_squared, 0.0, 1.0
        )
        closest = start + fractions[:, None] * direction
    distances = np.linalg.norm(obstacle_points - closest, axis=1)
    return bool(np.all(distances > clearance_radius_m))


def _dominant_tangent(
    points: np.ndarray, center: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    centered = points - center
    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, 2]
    tangent -= np.dot(tangent, normal) * normal
    return _unit_vector(tangent)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm
