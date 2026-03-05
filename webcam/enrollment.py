import cv2
import numpy as np

def generate_lighting_variations(crop):
    variations = []

    # 1️⃣ Original
    variations.append(crop)

    # 2️⃣ Slight Brightness Increase
    bright = cv2.convertScaleAbs(crop, alpha=1.15, beta=15)
    variations.append(bright)

    # 3️⃣ Slight Darkening
    dark = cv2.convertScaleAbs(crop, alpha=0.85, beta=-20)
    variations.append(dark)

    # 4️⃣ Mild Contrast Increase
    contrast = cv2.convertScaleAbs(crop, alpha=1.25, beta=0)
    variations.append(contrast)

    return variations


def handle_enrollment(crop, enroll_name, db, samples_captured,
                      has_specs, target, get_signature):

    # Generate lighting variations
    variations = generate_lighting_variations(crop)

    # Store embeddings for all variations
    for variant in variations:
        sig = get_signature(variant)
        db[enroll_name].append(sig)

    samples_captured += 1

    # Preserve your phase logic
    if not has_specs or samples_captured <= 20:
        info_msg = f"PHASE 1: MOVE HEAD SLOWLY ({samples_captured}/20)"
    else:
        info_msg = f"PHASE 2: WEAR SPECS & MOVE ({samples_captured}/40)"

    if samples_captured >= target:
        return True, samples_captured, info_msg

    return False, samples_captured, info_msg
