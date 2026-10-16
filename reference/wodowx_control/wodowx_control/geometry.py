"""Numerical geometry helpers shared by perception and control."""

from __future__ import annotations

import cv2
import numpy as np


def depth_mask_to_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    min_depth_m: float,
    max_depth_m: float,
    stride: int = 1,
    distortion_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project valid mask pixels into XYZ points in the camera optical frame."""
    if depth_m.ndim != 2 or mask.ndim != 2:
        raise ValueError("Depth and segmentation mask must both be 2-D arrays.")
    if depth_m.shape != mask.shape:
        raise ValueError(
            f"Depth/mask size mismatch: {depth_m.shape} versus {mask.shape}."
        )
    if stride < 1:
        raise ValueError("Point stride must be at least 1.")

    fx, fy, cx, cy = intrinsics
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive.")

    sampled_depth = depth_m[::stride, ::stride]
    sampled_mask = mask[::stride, ::stride].astype(bool)
    rows, cols = np.nonzero(
        sampled_mask
        & np.isfinite(sampled_depth)
        & (sampled_depth >= min_depth_m)
        & (sampled_depth < max_depth_m)
    )
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = sampled_depth[rows, cols].astype(np.float32)
    u = cols.astype(np.float32) * stride
    v = rows.astype(np.float32) * stride
    if distortion_coefficients is not None and np.any(distortion_coefficients):
        pixels = np.column_stack((u, v)).reshape(-1, 1, 2)
        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        normalized = cv2.undistortPoints(
            pixels,
            camera_matrix,
            np.asarray(distortion_coefficients, dtype=np.float64),
        ).reshape(-1, 2)
        x = normalized[:, 0] * z
        y = normalized[:, 1] * z
    else:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def decode_depth_buffer(
    data: bytes,
    width: int,
    height: int,
    step: int,
    encoding: str,
    is_bigendian: bool,
) -> np.ndarray:
    """Decode a ROS Image depth buffer while respecting row stride and byte order."""
    if encoding in ("16UC1", "mono16"):
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
    elif encoding == "32FC1":
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
    else:
        raise ValueError(
            f"Unsupported depth encoding {encoding!r}; expected 16UC1 or 32FC1."
        )

    row_bytes = width * dtype.itemsize
    if width <= 0 or height <= 0 or step < row_bytes or len(data) < height * step:
        raise ValueError("Depth Image message has an invalid step or data length.")
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    values = np.ascontiguousarray(rows[:, :row_bytes]).view(dtype)
    return values.reshape(height, width)


def xyz_from_point_cloud_reading(reading: object) -> np.ndarray:
    """Convert Humble's structured PointCloud2 view or tuple iterator to Nx3."""
    if isinstance(reading, np.ndarray) and reading.dtype.names is not None:
        names = reading.dtype.names
        if not all(name in names for name in ("x", "y", "z")):
            raise ValueError("PointCloud2 data must contain x, y, and z fields.")
        return np.column_stack(
            (reading["x"], reading["y"], reading["z"])
        ).astype(np.float64, copy=False)
    points = np.asarray(list(reading), dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("PointCloud2 data must contain XYZ points.")
    return points


def fit_leaf_plane(
    points: np.ndarray,
    up_axis: np.ndarray | None = None,
    toward_camera_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a plane with PCA and orient its normal toward camera/up when possible.

    Returns the centroid, unit normal, and dominant in-plane direction.
    The normal sign is otherwise ambiguous; the optical-frame defaults prefer
    both upward (-Y) and toward-camera (-Z) in ROS camera coordinates.
    """
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("Plane fitting requires at least three XYZ points.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Plane fitting received non-finite points.")

    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] < 1e-10:
        raise ValueError("Point cloud is degenerate and does not define a leaf plane.")

    normal = eigenvectors[:, 0]
    tangent = eigenvectors[:, 2]
    up = _unit_vector(
        np.asarray(up_axis if up_axis is not None else [0.0, -1.0, 0.0])
    )
    toward_camera = _unit_vector(
        np.asarray(
            toward_camera_axis
            if toward_camera_axis is not None
            else [0.0, 0.0, -1.0]
        )
    )
    if np.dot(normal, up) + np.dot(normal, toward_camera) < 0.0:
        normal = -normal

    return center.astype(np.float32), _unit_vector(normal), _unit_vector(tangent)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (vector / norm).astype(np.float32)
