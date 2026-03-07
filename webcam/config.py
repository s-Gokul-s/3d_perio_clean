import torch

MODEL_PATH = "training/models/adv_model/periocular_vit_adv_best"
DATABASE_FILE = "biometric_database.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MATCH_THRESHOLD = 0.68
ADAPTIVE_THRESHOLD = 0.82
SMOOTHING_WINDOW = 8
BLUR_THRESHOLD = 60.0
