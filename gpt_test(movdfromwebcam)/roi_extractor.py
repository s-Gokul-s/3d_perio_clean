import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [
    33, 133, 160, 159, 158, 157, 173,
    246, 161, 163, 144, 145, 153, 154, 155
]

def extract_periocular(frame):
    h, w, _ = frame.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0]
        xs, ys = [], []

        for idx in LEFT_EYE:
            lm = landmarks.landmark[idx]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        eye_w = x_max - x_min
        eye_h = y_max - y_min
        pad = int(0.4 * max(eye_w, eye_h))  # proportional padding

        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        roi = frame[y_min:y_max, x_min:x_max]
        return roi
