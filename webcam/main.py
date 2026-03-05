import cv2
import numpy as np
import mediapipe as mp
import time

from model_setup import load_model, get_transform
from feature_extraction import get_signature
from database import load_database, save_database
from authentication import authenticate
from config import *
from enrollment import handle_enrollment
from preprocessing import enhance_crop
from liveness import LivenessDetector

# ==============================
# INITIALIZATION
# ==============================
model     = load_model()
transform = get_transform()
db        = load_database()

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Biometric Access Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Biometric Access Control", 640, 480)

liveness_detector = LivenessDetector()

mode             = "AUTH"
enroll_step      = "IDLE"
enroll_name      = ""
has_specs        = False
samples_captured = 0
score_history    = []

pause_start = None

# ==============================
# MAIN LOOP
# ==============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_color  = (255,255,0)
    display_msg   = f"SYSTEM: {mode}"
    info_msg      = "PRESS [E] TO ENROLL"
    current_score = 0.0

    if not results.multi_face_landmarks:

        liveness_detector.reset()
        display_msg  = "NO FACE DETECTED"
        status_color = (0,0,255)

    else:

        lm = results.multi_face_landmarks[0]

        # ==============================
        # ROI EXTRACTION
        # ==============================
        p1   = np.array([lm.landmark[468].x * w, lm.landmark[468].y * h])
        p2   = np.array([lm.landmark[473].x * w, lm.landmark[473].y * h])
        dist = np.linalg.norm(p1 - p2)
        pad  = int(dist * 0.35)

        eye_idx = [33,133,159,145,153,154]
        pts = np.array([
            (int(lm.landmark[i].x * w), int(lm.landmark[i].y * h))
            for i in eye_idx
        ])

        ex, ey, ew, eh = cv2.boundingRect(pts)

        crop = frame[
            max(0, ey-pad):min(h, ey+eh+pad),
            max(0, ex-pad):min(w, ex+ew+pad)
        ]

        if crop.size > 0:

            roi_display = enhance_crop(crop)
            cv2.imshow("Periocular ROI", roi_display)

            # ==============================
            # AUTHENTICATION
            # ==============================
            if mode == "AUTH":

                live = liveness_detector.check(frame, lm)

                if live is None:

                    display_msg  = "VERIFYING LIVENESS..."
                    status_color = (0,255,255)

                elif live is False:

                    display_msg  = "ACCESS DENIED - SPOOF DETECTED"
                    status_color = (0,0,255)
                    current_score = 0.0

                else:

                    sig = get_signature(crop, model, transform)

                    if db:

                        success, user, current_score = authenticate(
                            sig, db, score_history
                        )

                        if success:

                            display_msg  = f"ACCESS GRANTED: {user}"
                            status_color = (0,255,0)

                        else:

                            display_msg  = "ACCESS DENIED - UNKNOWN USER"
                            status_color = (0,0,255)

            # ==============================
            # ENROLLMENT
            # ==============================
            elif mode == "ENROLL":

                if enroll_step == "WAITING_FOR_SPACE":

                    info_msg     = "LOOK AT CAMERA & PRESS [SPACE] TO START"
                    status_color = (0,255,255)

                # ------------------------------
                # PHASE 1 (NO SPECS)
                # ------------------------------
                elif enroll_step == "PHASE1":

                    finished, samples_captured, info_msg = handle_enrollment(
                        crop,
                        enroll_name,
                        db,
                        samples_captured,
                        False,
                        30,
                        lambda c, already_enhanced=False: get_signature(
                            c, model, transform, already_enhanced
                        )
                    )

                    if finished:

                        if has_specs:
                            enroll_step = "PAUSE"
                            pause_start = time.time()
                        else:
                            save_database(db)
                            liveness_detector.reset()
                            mode = "AUTH"
                            enroll_step = "IDLE"
                            print(f"[OK] {enroll_name} enrolled.")

                # ------------------------------
                # PAUSE FOR SPECS
                # ------------------------------
                elif enroll_step == "PAUSE":

                    display_msg = "PLEASE WEAR GLASSES"
                    info_msg = "Enrollment resumes in 5 seconds"

                    if time.time() - pause_start > 5:
                        samples_captured = 0
                        enroll_step = "PHASE2"

                # ------------------------------
                # PHASE 2 (WITH SPECS)
                # ------------------------------
                elif enroll_step == "PHASE2":

                    finished, samples_captured, info_msg = handle_enrollment(
                        crop,
                        enroll_name,
                        db,
                        samples_captured,
                        True,
                        30,
                        lambda c, already_enhanced=False: get_signature(
                            c, model, transform, already_enhanced
                        )
                    )

                    if finished:

                        save_database(db)
                        liveness_detector.reset()
                        mode = "AUTH"
                        enroll_step = "IDLE"

                        print(f"[OK] {enroll_name} enrolled with specs.")

        cv2.rectangle(
            frame,
            (ex-pad, ey-pad),
            (ex+ew+pad, ey+eh+pad),
            status_color,
            2
        )

    # ==============================
    # UI
    # ==============================
    cv2.rectangle(frame,(0,0),(w,60),(0,0,0),-1)

    score_color = (0,255,0) if current_score > MATCH_THRESHOLD else (0,0,255)

    cv2.putText(frame,f"SIMILARITY SCORE: {current_score:.4f}",
                (w-400,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                score_color,
                2)

    cv2.putText(frame,display_msg,
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2)

    cv2.putText(frame,info_msg,
                (20,h-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                1)

    cv2.imshow("Biometric Access Control",frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('e') and mode == "AUTH":

        enroll_name = input("Enter User Name: ")
        specs_input = input("Does this user wear glasses? (y/n): ").lower()

        has_specs = specs_input == "y"

        db[enroll_name] = []
        samples_captured = 0

        liveness_detector.reset()

        mode = "ENROLL"
        enroll_step = "WAITING_FOR_SPACE"

        print("Click camera window then press SPACE")

    elif key == 32 and enroll_step == "WAITING_FOR_SPACE":

        enroll_step = "PHASE1"

cap.release()
cv2.destroyAllWindows()