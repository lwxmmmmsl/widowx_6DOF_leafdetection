import numpy as np
import pytest

from wodowx_control.geometry import (
    decode_depth_buffer,
    depth_mask_to_points,
    fit_leaf_plane,
    xyz_from_point_cloud_reading,
)


def test_depth_backprojection_filters_invalid_and_far_pixels():
    depth = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=np.uint8)

    points = depth_mask_to_points(
        depth,
        mask,
        intrinsics=(2.0, 2.0, 0.0, 0.0),
        min_depth_m=0.1,
        max_depth_m=1.0,
    )

    np.testing.assert_allclose(points, [[0.25, 0.0, 0.5]])


def test_depth_backprojection_rejects_unaligned_inputs():
    with pytest.raises(ValueError, match="size mismatch"):
        depth_mask_to_points(
            np.ones((2, 3)),
            np.ones((3, 2)),
            intrinsics=(1.0, 1.0, 0.0, 0.0),
            min_depth_m=0.1,
            max_depth_m=2.0,
        )


def test_depth_image_decode_respects_endianness_and_row_padding():
    little_endian = np.array([[1000, 1500]], dtype="<u2").tobytes() + b"\0\0"
    decoded_little = decode_depth_buffer(
        little_endian,
        width=2,
        height=1,
        step=6,
        encoding="16UC1",
        is_bigendian=False,
    )
    big_endian = np.array([[1000, 1500]], dtype=">u2").tobytes() + b"\0\0"
    decoded_big = decode_depth_buffer(
        big_endian, 2, 1, 6, "16UC1", True
    )

    np.testing.assert_array_equal(decoded_little, [[1000, 1500]])
    np.testing.assert_array_equal(decoded_big, [[1000, 1500]])


def test_depth_image_decode_rejects_unsupported_encoding():
    with pytest.raises(ValueError, match="Unsupported depth encoding"):
        decode_depth_buffer(b"\0\0", 1, 1, 2, "mono8", False)


def test_point_cloud_reader_supports_humble_structured_arrays():
    structured = np.array(
        [(0.1, 0.2, 0.3), (1.0, 2.0, 3.0)],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
    )

    points = xyz_from_point_cloud_reading(structured)

    np.testing.assert_allclose(points, [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])


def test_plane_normal_prefers_camera_facing_direction():
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 5), np.linspace(-0.1, 0.1, 5))
    points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))

    center, normal, tangent = fit_leaf_plane(points)

    np.testing.assert_allclose(center, [0.0, 0.0, 1.0], atol=1e-6)
    assert normal[2] < -0.99
    assert np.isclose(np.linalg.norm(tangent), 1.0)
    assert abs(float(np.dot(normal, tangent))) < 1e-6
