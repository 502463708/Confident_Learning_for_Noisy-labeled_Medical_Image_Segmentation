import cv2
import numpy as np


def dilate_mask(src_label_np, dilation_radius):
    """
    This function implements the dilation for mask
    :param src_label_np:
    :param dilation_radius:
    :return:
    """
    assert dilation_radius > 0

    dilation_diameter = 2 * dilation_radius + 1
    kernel = np.zeros((dilation_diameter, dilation_diameter), np.uint8)

    for row_idx in range(dilation_diameter):
        for column_idx in range(dilation_diameter):
            if np.linalg.norm(np.array([row_idx, column_idx]) - np.array(
                    [dilation_radius, dilation_radius])) <= dilation_radius:
                kernel[row_idx, column_idx] = 1

    dst_label_np = cv2.dilate(src_label_np, kernel, iterations=1)

    assert dst_label_np.shape == src_label_np.shape

    return dst_label_np


def add_noisy(src_label_np, beta):
    assert len(src_label_np.shape) == 2
    assert 0 <= beta <= 1

    dst_label_np = dilate_mask(src_label_np, 2)

    return dst_label_np
