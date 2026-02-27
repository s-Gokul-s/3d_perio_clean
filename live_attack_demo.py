"""
live_attack_demo.py
===================
Live camera demo that shows adversarial attack in real time.

Press A to toggle attack ON/OFF and watch the similarity score drop.

HOW TO RUN:
    cd D:\MCA\restructuredperio(13_02_2026)
    python live_attack_demo.py

CONTROLS:
    A     — toggle FGSM attack ON/OFF
    +/-   — increase/decrease epsilon
    Q     — quit

WHAT TO OBSERVE:
    Attack OFF: similarity score ~0.75-0.90 → ACCESS GRANTED
    Attack ON:  similarity score drops below 0.65 → ACCESS DENIED
    (with clean model)

    After switching to adversarial model in config.py:
    Attack ON:  similarity score stays above 0.65 → ACCESS GRANTED (defended)
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "webcam"))

from webcam.model_setup import load_model, get_transform
from webcam.feature_extraction import get_signature
from webcam.database import load_database
from webcam.authentication import authenticate
from webcam.preprocessing import enhance_crop
from webcam.config import MATCH_THRESHOLD, DEVICE
from PIL import Image

# ─────────────────────────────────────────────────────────────
from transformers import ViTForImageClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTForImageClassification.from_pretrained(
    "training/models/adv_model/periocular_vit_adv_best"
).to(device)
model.eval()

print("Running Adversarial Robust Model")
transform = get_transform()
db        = load_database()

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ─────────────────────────────────────────────────────────────
attack_on = False
epsilon   = 0.005
score_history = []

def fgsm_attack(model, tensor, epsilon):
    """Apply FGSM perturbation to input tensor."""
    img = tensor.clone().detach().to(DEVICE)
    img.requires_grad_(True)
    logits = model(img).logits
    label  = torch.argmax(logits, dim=1)
    loss   = torch.nn.CrossEntropyLoss()(logits, label)
    model.zero_grad()
    loss.backward()
    adv = img + epsilon * img.grad.sign()
    return torch.clamp(adv, -1.0, 1.0).detach()


def get_signature_from_tensor(tensor, model):
    """Get embedding directly from tensor (skips enhance_crop)."""
    with torch.no_grad():
        out = model.vit(tensor)
    return out.last_hidden_state[0, 0].cpu().numpy()


print("Live Attack Demo started.")
print("Press A to toggle attack | +/- to change epsilon | Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    display_msg  = "NO FACE DETECTED"
    status_color = (0, 0, 255)
    current_sim  = 0.0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]

        # ROI extraction — same as main.py
        p1 = np.array([lm.landmark[468].x * w, lm.landmark[468].y * h])
        p2 = np.array([lm.landmark[473].x * w, lm.landmark[473].y * h])
        dist = np.linalg.norm(p1 - p2)
        pad  = int(dist * 0.35)

        eye_idx = [33, 133, 159, 145, 153, 154]
        pts = np.array([(int(lm.landmark[i].x * w),
                         int(lm.landmark[i].y * h)) for i in eye_idx])
        ex, ey, ew, eh = cv2.boundingRect(pts)

        crop = frame[max(0, ey-pad):min(h, ey+eh+pad),
                     max(0, ex-pad):min(w, ex+ew+pad)]

        if crop.size > 0:
            # Convert crop to tensor the same way feature_extraction.py does
            enhanced = enhance_crop(crop)
            pil_img  = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            tensor   = transform(pil_img).unsqueeze(0).to(DEVICE)

            if attack_on:
                # Apply FGSM attack to the input tensor
                model.eval()
                tensor = fgsm_attack(model, tensor, epsilon)

            # Get embedding and authenticate
            sig = get_signature_from_tensor(tensor, model)

            if db:
                from scipy.spatial.distance import cosine
                best_sim  = 0
                best_user = "UNKNOWN"
                for user, templates in db.items():
                    sims  = [1 - cosine(sig, t) for t in templates]
                    max_s = max(sims)
                    if max_s > best_sim:
                        best_sim  = max_s
                        best_user = user

                score_history.append(best_sim)
                if len(score_history) > 4:
                    score_history.pop(0)
                current_sim = np.mean(score_history)

                if current_sim >= MATCH_THRESHOLD:
                    display_msg  = f"ACCESS GRANTED: {best_user}"
                    status_color = (0, 255, 0)
                else:
                    display_msg  = "ACCESS DENIED"
                    status_color = (0, 0, 255)
            else:
                display_msg  = "NO DATABASE — enroll first"
                status_color = (0, 165, 255)

            # Draw ROI box — RED if attacking, GREEN/normal otherwise
            box_color = (0, 0, 255) if attack_on else status_color
            cv2.rectangle(frame,
                          (ex-pad, ey-pad), (ex+ew+pad, ey+eh+pad),
                          box_color, 2)

    # ── UI ────────────────────────────────────────────────────
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    cv2.putText(frame, display_msg,
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

    sim_color = (0, 255, 0) if current_sim >= MATCH_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"Similarity: {current_sim:.4f}",
                (w - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, sim_color, 2)

    # Attack status banner
    if attack_on:
        cv2.rectangle(frame, (0, 70), (w, 115), (0, 0, 180), -1)
        cv2.putText(frame,
                    f"⚠  FGSM ATTACK ACTIVE   epsilon={epsilon:.4f}   "
                    f"Press A to disable",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 70), (w, 115), (0, 100, 0), -1)
        cv2.putText(frame,
                    f"Attack OFF   epsilon={epsilon:.4f}   "
                    f"Press A to enable attack",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Bottom instructions
    cv2.putText(frame, "A=toggle attack | +/- change epsilon | Q=quit",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Live Adversarial Attack Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        attack_on = not attack_on
        score_history.clear()
        print(f"Attack {'ON' if attack_on else 'OFF'}  epsilon={epsilon:.4f}")
    elif key == ord('+') or key == ord('='):
        epsilon = min(round(epsilon + 0.005, 3), 0.1)
        print(f"Epsilon increased to {epsilon:.4f}")
    elif key == ord('-'):
        epsilon = max(round(epsilon - 0.005, 3), 0.001)
        print(f"Epsilon decreased to {epsilon:.4f}")

cap.release()
cv2.destroyAllWindows()