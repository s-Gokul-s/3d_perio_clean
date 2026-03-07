import cv2
import numpy as np

def generate_lighting_variations(crop):
    """
    Simulates real-world lighting environments (Dim, Harsh, Side-lit) 
    to create a robust biometric enrollment profile.
    """
    variations = []

    # 1️⃣ Baseline: Original captured frame
    variations.append(crop)

    # 2️⃣ Gamma Correction: Simulates Sensor Behavior in different light
    # Gamma < 1.0 simulates dim/shadowed environments
    # Gamma > 1.0 simulates harsh/over-exposed environments
    for g in [0.6, 1.6]:
        invGamma = 1.0 / g
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(crop, table)
        variations.append(gamma_corrected)

    # 3️⃣ Directional Shading: Simulates Window or Side-lighting
    # This creates a gradient shadow to mimic light coming from one side
    h, w = crop.shape[:2]
    gradient = np.linspace(0.5, 1.0, w) # Gradient from 50% darkness to 100% light
    mask = np.tile(gradient, (h, 1))
    
    # Apply shadow to the left
    shadow_left = (crop.astype(np.float32) * mask[:,:,None]).astype(np.uint8)
    variations.append(shadow_left)
    
    # Apply shadow to the right
    shadow_right = (crop.astype(np.float32) * mask[:,::-1,None]).astype(np.uint8)
    variations.append(shadow_right)

    # 4️⃣ Contrast Stretch: Simulates High-Intensity Artificial Light
    contrast = cv2.convertScaleAbs(crop, alpha=1.3, beta=-10)
    variations.append(contrast)

    return variations


def handle_enrollment(crop, enroll_name, db, samples_captured,
                      has_specs, target, get_signature):
    """
    Handles the multi-phase enrollment process and stores augmented signatures.
    """
    # Generate realistic lighting variations for each captured sample
    variations = generate_lighting_variations(crop)

    # Extract and store signatures for the original and all synthetic variations
    for variant in variations:
        sig = get_signature(variant)
        db[enroll_name].append(sig)

    samples_captured += 1

    # Progress Messaging
    if not has_specs or samples_captured <= 20:
        info_msg = f"PHASE 1: MOVE HEAD SLOWLY ({samples_captured}/20)"
    else:
        info_msg = f"PHASE 2: WEAR SPECS & MOVE ({samples_captured}/40)"

    # Completion Check
    if samples_captured >= target:
        return True, samples_captured, info_msg

    return False, samples_captured, info_msg