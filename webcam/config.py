import torch

MODEL_PATH = "training/models/adv_model/periocular_vit_adv_best"
DATABASE_FILE = "biometric_database.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MATCH_THRESHOLD = 0.62
ADAPTIVE_THRESHOLD = 0.82
SMOOTHING_WINDOW = 6
BLUR_THRESHOLD = 60.0
