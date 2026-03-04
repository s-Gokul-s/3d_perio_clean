import cv2
from PIL import Image
import numpy as np
import torch
from config import DEVICE
from preprocessing import enhance_crop


def get_signature(crop, model, transform, already_enhanced=False):
    """
    Get embedding from a periocular crop.
    
    already_enhanced=False  → applies enhance_crop (normal auth flow)
    already_enhanced=True   → skips enhance_crop (enrollment variants
                               that were already preprocessed)
    """
    if already_enhanced:
        enhanced = crop
    else:
        enhanced = enhance_crop(crop)

    img    = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model.vit(tensor)

    return out.last_hidden_state[0, 0].cpu().numpy()