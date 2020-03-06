import cv2
import numpy as np
import random


def dilate_mask(src_label_np, min_radius, max_radius):
    """
    This function implements the dilation for mask
    :param src_label_np:
    :param radius:
    :return:
    """
    assert max_radius >= min_radius > 0

    radius = random.randint(min_radius, max_radius)

    dilation_diameter = int(2 * radius + 1)
    kernel = np.zeros((dilation_diameter, dilation_diameter), np.uint8)

    for row_idx in range(dilation_diameter):
        for column_idx in range(dilation_diameter):
            if np.linalg.norm(np.array([row_idx, column_idx]) - np.array(
                    [radius, radius])) <= radius:
                kernel[row_idx, column_idx] = 1

    dst_label_np = cv2.dilate(src_label_np, kernel, iterations=1)

    assert dst_label_np.shape == src_label_np.shape

    return dst_label_np


def erode_mask(src_label_np, min_radius, max_radius):
    """
    This function implements the dilation for mask
    :param src_label_np:
    :param radius:
    :return:
    """
    assert max_radius >= min_radius > 0

    radius = random.randint(min_radius, max_radius)

    erode_diameter = int(2 * radius + 1)
    kernel = np.zeros((erode_diameter, erode_diameter), np.uint8)

    for row_idx in range(erode_diameter):
        for column_idx in range(erode_diameter):
            if np.linalg.norm(np.array([row_idx, column_idx]) - np.array(
                    [radius, radius])) <= radius:
                kernel[row_idx, column_idx] = 1

    dst_label_np = cv2.erode(src_label_np, kernel, iterations=1)

    assert dst_label_np.shape == src_label_np.shape

    return dst_label_np


def foreground_move(src_label_np, max_distance):
    assert max_distance >= 0

    height, width = src_label_np.shape

    padded_label_np = np.pad(src_label_np, ((max_distance, max_distance), (max_distance, max_distance)), 'constant', constant_values=0)

    r = np.random.randint(0, 2 * max_distance)
    c = np.random.randint(0, 2 * max_distance)

    dst_label_np = padded_label_np[r:r + height, c:c + width]

    assert src_label_np.shape == dst_label_np.shape

    return dst_label_np


def add_noise(src_label_np, beta1, beta2):
    assert len(src_label_np.shape) == 2
    assert 0 <= beta1 <= beta2

    random_num = random.random()
    noise_type = 'dilate'

    if random_num < 0.33:
        dst_label_np = dilate_mask(src_label_np, beta1, beta2)
    elif random_num < 0.66:
        dst_label_np = erode_mask(src_label_np, beta1 - 5, beta2 - 5)
        noise_type = 'erode'
    else:
        dst_label_np = foreground_move(src_label_np, beta2)
        noise_type = 'foreground move'

    return dst_label_np, noise_type
