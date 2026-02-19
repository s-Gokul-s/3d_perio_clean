import cv2
import torch
import numpy as np
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
import torch.nn.functional as F
from roi_extractor import extract_periocular

MODEL_PATH = "training/models/periocular_vit_best"
TEMPLATE_DIR = "webcam/templates"
THRESHOLD = 0.65
NUM_FRAMES = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

user_id = input("Enter User ID: ").strip()
template = torch.load(f"{TEMPLATE_DIR}/{user_id}.pt").to(device)

cap = cv2.VideoCapture(0)
scores = []

print("\nAuthenticating...\n")

while len(scores) < NUM_FRAMES:
    ret, frame = cap.read()
    if not ret:
        continue

    roi = extract_periocular(frame)
    if roi is None:
        cv2.imshow("Authentication", frame)
        cv2.waitKey(1)
        continue

    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).last_hidden_state[:, 0, :]
        sim = F.cosine_similarity(emb, template).item()
        scores.append(sim)

    cv2.putText(frame, f"Similarity: {sim:.2f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Authentication", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

final_score = float(np.median(scores))

print(f"Final Similarity Score: {final_score:.2f}")

if final_score >= THRESHOLD:
    print("✅ ACCESS GRANTED")
else:
    print("❌ ACCESS DENIED")
