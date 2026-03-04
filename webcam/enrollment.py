import cv2
import numpy as np
from preprocessing import enhance_crop


def _residual_brightness(crop, delta):
    return np.clip(crop.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def _add_noise(crop, std=8):
    noise = np.random.normal(0, std, crop.shape).astype(np.float32)
    return np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _color_cast(crop, b=0, g=0, r=0):
    out = crop.astype(np.float32).copy()
    out[:, :, 0] = np.clip(out[:, :, 0] + b, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] + g, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] + r, 0, 255)
    return out.astype(np.uint8)


def _small_blur(crop):
    return cv2.GaussianBlur(crop, (3, 3), 0)


def generate_lighting_variations(crop):
    """
    Preprocess ONCE here, then generate variants.
    Variants are passed to get_signature with already_enhanced=True
    so enhance_crop is NOT called again inside feature_extraction.
    This prevents double processing.
    """
    base = enhance_crop(crop)  # single preprocessing pass

    return [
        base.copy(),
        _residual_brightness(base, +20),
        _residual_brightness(base, -20),
        _residual_brightness(base, +40),
        _residual_brightness(base, -40),
        _add_noise(base, std=6),
        _add_noise(base, std=12),
        _small_blur(base),
        _color_cast(base, b=15, r=-10),
        _color_cast(base, r=15, b=-10),
    ]


def handle_enrollment(crop, enroll_name, db, samples_captured,
                      has_specs, target, get_signature):
    for variant in generate_lighting_variations(crop):
        # already_enhanced=True — variant was preprocessed in generate_lighting_variations
        # so feature_extraction must NOT call enhance_crop again
        sig = get_signature(variant, already_enhanced=True)
        db[enroll_name].append(sig)

    samples_captured += 1
    info_msg = f"ENROLLING: MOVE HEAD SLOWLY ({samples_captured}/{target})"

    if samples_captured >= target:
        return True, samples_captured, info_msg
    return False, samples_captured, info_msg