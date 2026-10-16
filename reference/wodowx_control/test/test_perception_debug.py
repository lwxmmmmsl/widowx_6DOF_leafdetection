import numpy as np
from std_msgs.msg import Header

from wodowx_control.perception_node import LeafPerceptionNode


def test_debug_image_message_preserves_bgr_pixels_and_header():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    header = Header()
    header.frame_id = "rgb_camera_link"

    message = LeafPerceptionNode._make_bgr_image_message(image, header)

    assert message.header.frame_id == "rgb_camera_link"
    assert message.encoding == "bgr8"
    assert (message.height, message.width, message.step) == (4, 5, 15)
    np.testing.assert_array_equal(
        np.frombuffer(message.data, dtype=np.uint8).reshape(4, 5, 3),
        image,
    )


def test_debug_prediction_draws_mask_and_box():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[6:12, 6:12] = True

    LeafPerceptionNode._draw_prediction(
        image=image,
        mask=mask,
        box=np.array([5, 5, 13, 13]),
        label="Leaf",
        instance_id=2,
        confidence=0.37,
        color=(0, 255, 0),
        detail="target class",
    )

    assert image[8, 8, 1] > image[8, 8, 2]
    np.testing.assert_array_equal(image[5, 5], [0, 255, 0])
