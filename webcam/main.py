import cv2
import numpy as np
import mediapipe as mp

from model_setup import load_model, get_transform
from feature_extraction import get_signature
from database import load_database, save_database
from authentication import authenticate
from config import *
from enrollment import handle_enrollment
from preprocessing import enhance_crop
from liveness import LivenessDetector   # ✅ NEW PASSIVE RGB LIVENESS


# ==============================
# INITIALIZATION
# ==============================
model = load_model()
transform = get_transform()
db = load_database()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


cv2.namedWindow("Biometric Access Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Biometric Access Control", 640, 480)

# ✅ Initialize passive liveness detector
liveness_detector = LivenessDetector()

mode = "AUTH"
enroll_step = "IDLE"
enroll_name = ""
has_specs = False
samples_captured = 0

score_history = []


# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_color = (255, 255, 0)
    display_msg = f"SYSTEM: {mode}"
    info_msg = "PRESS [E] TO ENROLL"
    current_score = 0.0

    # ------------------------------------------------
    # IF NO FACE → RESET LIVENESS (IMPORTANT)
    # ------------------------------------------------
    if not results.multi_face_landmarks:
        liveness_detector.reset()
        display_msg = "NO FACE DETECTED"
        status_color = (0, 0, 255)

    else:
        lm = results.multi_face_landmarks[0]

        # -----------------------------
        # ROI EXTRACTION
        # -----------------------------
        p1 = np.array([lm.landmark[468].x * w, lm.landmark[468].y * h])
        p2 = np.array([lm.landmark[473].x * w, lm.landmark[473].y * h])
        dist = np.linalg.norm(p1 - p2)
        pad = int(dist * 0.35)

        eye_idx = [33, 133, 159, 145, 153, 154]
        pts = np.array([
            (int(lm.landmark[i].x * w),
             int(lm.landmark[i].y * h))
            for i in eye_idx
        ])

        ex, ey, ew, eh = cv2.boundingRect(pts)

        crop = frame[
            max(0, ey - pad):min(h, ey + eh + pad),
            max(0, ex - pad):min(w, ex + ew + pad)
        ]

        if crop.size > 0:

            roi_display = enhance_crop(crop)
            cv2.imshow("Periocular ROI", roi_display)

            # =====================================
            # AUTHENTICATION MODE
            # =====================================
            if mode == "AUTH":

                # 1️⃣ Passive Liveness Check
                live = liveness_detector.check(frame, lm)

                if live is None:
                    display_msg = "VERIFYING LIVENESS..."
                    status_color = (0, 255, 255)

                elif live is False:
                    display_msg = "ACCESS DENIED - SPOOF DETECTED"
                    status_color = (0, 0, 255)
                    current_score = 0.0

                else:
                    # 2️⃣ Identity Verification
                    sig = get_signature(crop, model, transform)

                    if db:
                        success, user, current_score = authenticate(
                            sig, db, score_history
                        )

                        if success:
                            display_msg = f"ACCESS GRANTED: {user}"
                            status_color = (0, 255, 0)

                        else:
                            display_msg = "ACCESS DENIED - UNKNOWN USER"
                            status_color = (0, 0, 255)

            # =====================================
            # ENROLLMENT MODE
            # =====================================
            elif mode == "ENROLL":

                if enroll_step == "WAITING_FOR_SPACE":
                    info_msg = "LOOK AT CAMERA & PRESS [SPACE] TO START"
                    status_color = (0, 255, 255)

                elif enroll_step == "CAPTURING":

                    target = 80 if has_specs else 20

                    finished, samples_captured, info_msg = handle_enrollment(
                        crop,
                        enroll_name,
                        db,
                        samples_captured,
                        has_specs,
                        target,
                        lambda c: get_signature(c, model, transform)
                    )

                    if finished:
                        save_database(db)
                        liveness_detector.reset()
                        mode = "AUTH"
                        enroll_step = "IDLE"
                        print(f"User {enroll_name} saved successfully.")

        # Draw ROI box
        cv2.rectangle(
            frame,
            (ex - pad, ey - pad),
            (ex + ew + pad, ey + eh + pad),
            status_color,
            2
        )

    # -----------------------------
    # UI OVERLAY
    # -----------------------------
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

    score_display_color = (0, 255, 0) if current_score > MATCH_THRESHOLD else (0, 0, 255)

    cv2.putText(
        frame,
        f"SIMILARITY SCORE: {current_score:.4f}",
        (w - 400, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        score_display_color,
        2
    )

    cv2.putText(
        frame,
        display_msg,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    cv2.putText(
        frame,
        info_msg,
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.imshow("Biometric Access Control", frame)

    # -----------------------------
    # KEYBOARD CONTROLS
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('e') and mode == "AUTH":
        enroll_name = input("Enter User Name: ")
        specs_input = input("Does this user wear glasses? (y/n): ").lower()
        has_specs = True if specs_input == 'y' else False

        db[enroll_name] = []
        samples_captured = 0

        liveness_detector.reset()

        mode = "ENROLL"
        enroll_step = "WAITING_FOR_SPACE"
        print("Switching to Camera... Click on the Camera window then press SPACE.")

    elif key == 32 and enroll_step == "WAITING_FOR_SPACE":
        enroll_step = "CAPTURING"

cap.release()
cv2.destroyAllWindows()
 