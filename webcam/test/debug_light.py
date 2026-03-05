"""
debug_lighting.py
=================
Run this to see exactly what your preprocessing outputs look like
and what similarity scores you get across lighting conditions.

This will help understand why morning/daylight light fails.

HOW TO RUN:
    cd D:\MCA\restructuredperio(13_02_2026)
    python debug_lighting.py

WHAT IT SHOWS:
    - Live camera feed
    - The preprocessed crop as the model sees it
    - Mean brightness of the crop BEFORE and AFTER preprocessing
    - Current similarity score
    - Press S to save a snapshot of current crop for inspection
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
sys.path.insert(0, 'webcam')

from model_setup import load_model, get_transform
from feature_extraction import get_signature
from database import load_database
from preprocessing import enhance_crop
from config import MATCH_THRESHOLD, DEVICE
from scipy.spatial.distance import cosine

model     = load_model()
transform = get_transform()
db        = load_database()

if not db:
    print("No database found. Enroll first.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Running debug. Press S to save snapshot. Press Q to quit.")
print("Watch the BEFORE/AFTER brightness values.\n")

snap_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    debug_info = []

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]

        p1   = np.array([lm.landmark[468].x * w, lm.landmark[468].y * h])
        p2   = np.array([lm.landmark[473].x * w, lm.landmark[473].y * h])
        dist = np.linalg.norm(p1 - p2)
        pad  = int(dist * 0.35)

        eye_idx = [33, 133, 159, 145, 153, 154]
        pts = np.array([(int(lm.landmark[i].x * w),
                         int(lm.landmark[i].y * h)) for i in eye_idx])
        ex, ey, ew, eh = cv2.boundingRect(pts)

        crop = frame[max(0, ey-pad):min(h, ey+eh+pad),
                     max(0, ex-pad):min(w, ex+ew+pad)]

        if crop.size > 0:
            # Measure BEFORE preprocessing
            brightness_before = float(np.mean(crop))

            # Preprocess
            enhanced = enhance_crop(crop)

            # Measure AFTER preprocessing
            brightness_after = float(np.mean(enhanced))

            # Get similarity
            sig = get_signature(crop, model, transform)
            sims = []
            for user, templates in db.items():
                user_sims = [1 - cosine(sig, t) for t in templates]
                top5      = sorted(user_sims, reverse=True)[:10]
                sims.append((user, float(np.mean(top5))))
            best_user, best_sim = max(sims, key=lambda x: x[1])

            debug_info = [
                f"Brightness BEFORE enhance: {brightness_before:.1f}",
                f"Brightness AFTER  enhance: {brightness_after:.1f}",
                f"Best match: {best_user}  sim={best_sim:.4f}",
                f"Decision: {'GRANTED' if best_sim > MATCH_THRESHOLD else 'DENIED'}",
            ]

            # Show enhanced crop large
            crop_display = cv2.resize(enhanced, (300, 200))
            cv2.imshow("Enhanced Crop (what model sees)", crop_display)

            # Show original crop
            crop_orig = cv2.resize(crop, (300, 200))
            cv2.imshow("Original Crop (raw)", crop_orig)

            # Print to console
            print("\r" + "  |  ".join(debug_info), end="", flush=True)

    # Overlay on main frame
    cv2.rectangle(frame, (0, 0), (w, 30*len(debug_info)+10), (0,0,0), -1)
    for i, line in enumerate(debug_info):
        color = (0,255,0) if "GRANTED" in line else (0,0,255) if "DENIED" in line else (255,255,255)
        cv2.putText(frame, line, (10, 25+i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.imshow("Debug View", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and crop is not None:
        snap_count += 1
        cv2.imwrite(f"debug_snap_{snap_count}_raw.jpg", crop)
        cv2.imwrite(f"debug_snap_{snap_count}_enhanced.jpg", enhanced)
        print(f"\nSaved snapshot {snap_count}")
        print(f"  Raw brightness:      {brightness_before:.1f}")
        print(f"  Enhanced brightness: {brightness_after:.1f}")
        print(f"  Similarity score:    {best_sim:.4f}")

cap.release()
cv2.destroyAllWindows()