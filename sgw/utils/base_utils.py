import numpy as np
import cv2 as cv


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = [int(w / 2), int(h / 2)]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def resize_obs(img, resolution, torch_obs):
    """Unified image resizing logic."""
    if torch_obs:
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_NEAREST)
        return np.moveaxis(img, 2, 0) / 255.0
    return cv.resize(img, (resolution, resolution), interpolation=cv.INTER_NEAREST)
