import cv2
from PIL import Image
import torch
from config import DEVICE
from preprocessing import enhance_crop

def get_signature(crop, model, transform):
    enhanced = enhance_crop(crop)
    img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model.vit(tensor)

    return out.last_hidden_state[0, 0].cpu().numpy()
