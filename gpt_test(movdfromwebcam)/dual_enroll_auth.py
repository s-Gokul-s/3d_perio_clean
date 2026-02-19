import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import pickle
import time
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from scipy.spatial.distance import cosine

# =============================
# SETTINGS
# =============================
MODEL_PATH = "training/models/periocular_vit_best"
DATABASE_FILE = "biometric_database.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MATCH_THRESHOLD = 0.65 
ADAPTIVE_THRESHOLD = 0.82 
SMOOTHING_WINDOW = 6
BLUR_THRESHOLD = 60.0

# =============================
# INITIALIZATION
# =============================
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

db = {}
if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, "rb") as f:
        db = pickle.load(f)

def enhance_crop(crop):
    # 1. Convert to Grayscale to remove color-casting from different light bulbs
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply aggressive CLAHE to standardize contrast
    # This makes the dark areas and light areas look the same in any room
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    standardized = clahe.apply(gray)
    
    # 3. Denoising to remove "grain" caused by low-light sensors
    denoised = cv2.fastNlMeansDenoising(standardized, None, 10, 7, 21)
    
    # ViT expects 3 channels, so we convert grayscale back to BGR 
    # but the information is now color-neutral
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def get_signature(crop):
    enhanced = enhance_crop(crop)
    img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # Normalizing tensors helps the model focus on structure rather than brightness
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Using the feature extractor specifically
        out = model.vit(tensor)
    
    # Flatten and return for cosine similarity
    return out.last_hidden_state[0, 0].cpu().numpy()

# =============================
# MAIN SYSTEM
# =============================
cap = cv2.VideoCapture(1) # Iriun Webcam Index

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mode = "AUTH"
enroll_step = "IDLE" 
enroll_name = ""
has_specs = False
samples_captured = 0
score_history = []

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_color = (255, 255, 0)
    display_msg = f"SYSTEM: {mode}"
    info_msg = "PRESS [E] TO ENROLL"
    current_score = 0.0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        p1 = np.array([lm.landmark[468].x*w, lm.landmark[468].y*h])
        p2 = np.array([lm.landmark[473].x*w, lm.landmark[473].y*h])
        dist = np.linalg.norm(p1-p2)
        pad = int(dist * 0.35) 

        eye_idx = [33, 133, 159, 145, 153, 154]
        pts = np.array([(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in eye_idx])
        ex, ey, ew, eh = cv2.boundingRect(pts)
        crop = frame[max(0,ey-pad):min(h,ey+eh+pad), max(0,ex-pad):min(w,ex+ew+pad)]

        if crop.size > 0:
            if mode == "AUTH":
                sig = get_signature(crop)
                if db:
                    best_sim = 0
                    best_user = "None"
                    for user, templates in db.items():
                        sims = [1 - cosine(sig, t) for t in templates]
                        max_s = np.mean(sorted(sims, reverse=True)[:3])
                        if max_s > best_sim:
                            best_sim = max_s
                            best_user = user
                    
                    score_history.append(best_sim)
                    if len(score_history) > SMOOTHING_WINDOW: score_history.pop(0)
                    current_score = np.mean(score_history)

                    if current_score > MATCH_THRESHOLD:
                        display_msg = f"ACCESS GRANTED: {best_user}"
                        status_color = (0, 255, 0)
                    else:
                        display_msg = "UNKNOWN USER"
                        status_color = (0, 0, 255)

            elif mode == "ENROLL":
                if enroll_step == "WAITING_FOR_SPACE":
                    info_msg = "LOOK AT CAMERA & PRESS [SPACE] TO START"
                    status_color = (0, 255, 255)
                
                elif enroll_step == "CAPTURING":
                    sig = get_signature(crop)
                    db[enroll_name].append(sig)
                    samples_captured += 1
                    
                    if not has_specs or samples_captured <= 20:
                        info_msg = f"PHASE 1: MOVE HEAD SLOWLY ({samples_captured}/20)"
                    else:
                        info_msg = f"PHASE 2: WEAR SPECS & MOVE ({samples_captured}/40)"

                    target = 40 if has_specs else 20
                    if samples_captured >= target:
                        with open(DATABASE_FILE, "wb") as f:
                            pickle.dump(db, f)
                        mode = "AUTH"
                        enroll_step = "IDLE"
                        print(f"User {enroll_name} saved successfully.")

        cv2.rectangle(frame, (ex-pad, ey-pad), (ex+ew+pad, ey+eh+pad), status_color, 2)

    # --- UI UPDATED FOR LIVE SCORE ---
    cv2.rectangle(frame, (0, 0), (w, 60), (0,0,0), -1)
    
    # Display the score with dynamic coloring (Green if above threshold, Red if below)
    score_display_color = (0, 255, 0) if current_score > MATCH_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"SIMILARITY SCORE: {current_score:.4f}", (w - 400, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_display_color, 2)

    cv2.putText(frame, display_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, info_msg, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Biometric Access Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    
    elif key == ord('e') and mode == "AUTH":
        enroll_name = input("Enter User Name: ")
        specs_input = input("Does this user wear glasses? (y/n): ").lower()
        has_specs = True if specs_input == 'y' else False
        
        db[enroll_name] = []
        samples_captured = 0
        mode = "ENROLL"
        enroll_step = "WAITING_FOR_SPACE"
        print("Switching to Camera... Click on the Camera window then press SPACE.")

    elif key == 32 and enroll_step == "WAITING_FOR_SPACE": 
        enroll_step = "CAPTURING"

cap.release()
cv2.destroyAllWindows()