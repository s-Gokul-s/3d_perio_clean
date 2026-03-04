"""
preprocessing.py
================
Strong lighting normalization for periocular authentication.

The key insight: force EVERY crop to look the same brightness
regardless of room lighting. Use a high target mean (140+) so
even very dark frames get aggressively brightened.
"""

import cv2
import numpy as np


def _force_brightness(bgr, target_mean=140.0):
    """
    Aggressively normalizes brightness to target_mean.
    Works by computing the actual mean and applying gamma to reach target.
    target_mean=140 means every crop will look like it was taken
    in good indoor lighting, regardless of actual conditions.
    """
    mean = np.mean(bgr)
    if mean < 2.0:
        return bgr

    gamma = np.log(target_mean / 255.0) / np.log(max(mean, 1.0) / 255.0)
    gamma = float(np.clip(gamma, 0.2, 4.0))

    lut = np.array([
        min(255, int((i / 255.0) ** gamma * 255))
        for i in range(256)
    ], dtype=np.uint8)

    return cv2.LUT(bgr, lut)


def _clahe_lab(bgr):
    """CLAHE on LAB L channel — preserves colour, normalises contrast."""
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_eq    = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def enhance_crop(crop: np.ndarray) -> np.ndarray:
    """
    Two-pass brightness normalization:
      Pass 1: Gamma to target brightness (handles dim/dark rooms)
      Pass 2: LAB CLAHE for local contrast (handles uneven lighting)

    NO grayscale conversion — model was trained on RGB.
    """
    if crop is None or crop.size == 0:
        return crop

    # Pass 1: force brightness to target
    out = _force_brightness(crop, target_mean=140.0)

    # Pass 2: local contrast normalization
    out = _clahe_lab(out)

    return out