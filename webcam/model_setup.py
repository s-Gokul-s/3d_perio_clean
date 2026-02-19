from transformers import ViTForImageClassification
from torchvision import transforms
from config import MODEL_PATH, DEVICE

def load_model():
    model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
