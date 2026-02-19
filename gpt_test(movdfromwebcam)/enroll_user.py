import cv2
import torch
import os
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
from roi_extractor import extract_periocular

MODEL_PATH = "training/models/periocular_vit_best"
SAVE_DIR = "webcam/templates"
NUM_FRAMES = 16  # 8 with specs + 8 without specs

os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

user_id = input("Enter User ID: ").strip()
cap = cv2.VideoCapture(0)

embeddings = []
print("\nEnrollment started")
print("→ First HALF: WITH specs")
print("→ Second HALF: WITHOUT specs\n")

while len(embeddings) < NUM_FRAMES:
    ret, frame = cap.read()
    if not ret:
        continue

    roi = extract_periocular(frame)
    if roi is None:
        cv2.imshow("Enrollment", frame)
        cv2.waitKey(1)
        continue

    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu())

    msg = f"Captured {len(embeddings)}/{NUM_FRAMES}"
    cv2.putText(frame, msg, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Enrollment", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

template = torch.mean(torch.cat(embeddings), dim=0)
torch.save(template, f"{SAVE_DIR}/{user_id}.pt")

print(f"\n✅ Enrollment successful for user: {user_id}")
