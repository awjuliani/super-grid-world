from typing import Dict
import numpy as np
import cv2 as cv


def onehot(value: int, max_value: int):
    """
    Creates a onehot encoding of an integer number.
    """
    vec = np.zeros(max_value, dtype=np.int32)
    value = np.clip(value, 0, max_value - 1)
    vec[value] = 1
    return vec


def twohot(value, max_value):
    """
    Creates a two-hot encoding of a given pair of integers.
    """
    vec_1 = np.zeros(max_value, dtype=np.float32)
    vec_2 = np.zeros(max_value, dtype=np.float32)
    vec_1[value[0]] = 1
    vec_2[value[1]] = 1
    return np.concatenate([vec_1, vec_2])


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = [int(w / 2), int(h / 2)]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def softmax(x, axis=-1):
    """
    Computes the softmax function on a given vector.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis)


def resize_obs(img, resolution, torch_obs):
    """Unified image resizing logic."""
    if torch_obs:
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_NEAREST)
        return np.moveaxis(img, 2, 0) / 255.0
    return cv.resize(img, (resolution, resolution), interpolation=cv.INTER_NEAREST)
