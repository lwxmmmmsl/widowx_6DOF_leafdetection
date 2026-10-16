import numpy as np

from wodowx_control.target_planning import (
    LeafTarget,
    choose_collision_free_targets,
    make_leaf_target,
    quaternion_to_rotation,
)


def test_quaternion_to_rotation_maps_identity():
    np.testing.assert_allclose(quaternion_to_rotation([0.0, 0.0, 0.0, 1.0]), np.eye(3))


def test_target_uses_point_nearest_robot_base_and_right_handed_orientation():
    points = np.array([[0.4, 0.1, 0.3], [0.2, 0.2, 0.3], [0.3, 0.3, 0.3]])
    target = make_leaf_target(
        leaf_id=4,
        confidence=0.9,
        points_base=points,
        center_base=np.mean(points, axis=0),
        normal_base=np.array([0.0, 0.0, 1.0]),
        approach_distance_m=0.1,
    )

    np.testing.assert_allclose(target.contact_position, points[1])
    np.testing.assert_allclose(target.orientation[:, 1], [0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.norm(target.approach_position - points[1]), 0.1)
    assert abs(
        float(
            np.dot(
                target.approach_position - target.contact_position,
                target.orientation[:, 1],
            )
        )
    ) < 1e-8
    assert np.dot(
        target.contact_position - target.approach_position,
        target.orientation[:, 2],
    ) > 0.0
    np.testing.assert_allclose(target.orientation.T @ target.orientation, np.eye(3))
    assert np.isclose(np.linalg.det(target.orientation), 1.0)


def test_selection_skips_target_blocked_by_another_leaf():
    blocked = LeafTarget(
        leaf_id=1,
        confidence=0.99,
        points=np.array([[1.0, 0.0, 0.0]]),
        contact_position=np.array([1.0, 0.0, 0.0]),
        approach_position=np.array([1.0, 0.0, 0.1]),
        orientation=np.eye(3),
    )
    clear = LeafTarget(
        leaf_id=2,
        confidence=0.8,
        points=np.array([[0.5, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        contact_position=np.array([2.0, 1.0, 0.0]),
        approach_position=np.array([2.0, 1.0, 0.1]),
        orientation=np.eye(3),
    )

    selected = choose_collision_free_targets(
        [blocked, clear],
        desired_count=1,
        start_position=np.zeros(3),
        clearance_radius_m=0.05,
    )

    assert [target.leaf_id for target in selected] == [2]


def test_selection_rejects_approach_path_through_its_target_leaf():
    target = LeafTarget(
        leaf_id=3,
        confidence=0.9,
        points=np.array([[0.5, 0.0, 0.05], [1.0, 0.0, 0.0]]),
        contact_position=np.array([1.0, 0.0, 0.0]),
        approach_position=np.array([1.0, 0.0, 0.1]),
        orientation=np.eye(3),
    )

    selected = choose_collision_free_targets(
        [target],
        desired_count=1,
        start_position=np.zeros(3),
        clearance_radius_m=0.05,
    )

    assert selected == []


def test_selection_orders_each_target_from_previous_approach_position():
    def target(leaf_id, approach):
        approach_position = np.asarray(approach, dtype=np.float64)
        contact_position = approach_position + np.array([0.0, 0.0, 1.0])
        return LeafTarget(
            leaf_id=leaf_id,
            confidence=0.9,
            points=np.array([contact_position]),
            contact_position=contact_position,
            approach_position=approach_position,
            orientation=np.eye(3),
        )

    first = target(1, [0.8, 0.0, 0.0])
    second_from_initial = target(2, [-0.9, 0.0, 0.0])
    second_from_first = target(3, [1.0, 1.0, 0.0])

    selected = choose_collision_free_targets(
        [first, second_from_initial, second_from_first],
        desired_count=3,
        start_position=np.zeros(3),
        clearance_radius_m=0.02,
    )

    assert [candidate.leaf_id for candidate in selected] == [1, 3, 2]
