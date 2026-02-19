import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# =============================
# SETTINGS
# =============================
MODEL_PATH = "training/models/periocular_vit_best"
IMAGE_DIR = "dataset/images"
TEST_IMAGE = "dataset/images/C1_S1_I1.jpg" # Change to any image in your folder
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# HELPER: REBUILD LABEL MAP
# =============================
def get_label_map(image_dir):
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    persons = sorted(list(set(f.split("_")[0] for f in files)))
    id_to_person = {i: p for i, p in enumerate(persons)}
    return id_to_person

# =============================
# AUTHENTICATION LOGIC
# =============================
def authenticate(image_path):
    # 1. Rebuild the map
    id_to_person = get_label_map(IMAGE_DIR)
    
    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # 3. Prepare Image (Must match the Val transforms exactly!)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension

    # 4. Predict
    with torch.no_grad():
        outputs = model(img_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    person_id = id_to_person[predicted_idx.item()]
    conf_score = confidence.item() * 100

    # 5. Result
    print("-" * 30)
    print(f"Target Image: {os.path.basename(image_path)}")
    print(f"Detected Identity: {person_id}")
    print(f"Confidence Score: {conf_score:.2f}%")
    
    # MCA Logic: Adding a threshold for security
    if conf_score > 85:
        print("Status: ✅ AUTHENTICATION SUCCESS")
    else:
        print("Status: ❌ ACCESS DENIED (Low Confidence)")
    print("-" * 30)

if __name__ == "__main__":
    if os.path.exists(TEST_IMAGE):
        authenticate(TEST_IMAGE)
    else:
        print(f"Error: {TEST_IMAGE} not found. Please update the TEST_IMAGE path.")