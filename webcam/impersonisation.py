"""
Impersonation Attack Test
=========================
Tests whether FGSM can move attacker's embedding toward victim's embedding.

Run from webcam/ folder:
    python test_impersonation.py

What it proves:
  - Clean model: can targeted FGSM impersonate? (expected: partially yes at high ε)
  - Adv model:   same attack harder to succeed (expected: fails or needs much higher ε)
  - Shows embedding distance before/after attack
  - Shows how many ε levels are needed to cross threshold
"""

import torch
import numpy as np
import cv2
import pickle
import sys
from PIL import Image
from scipy.spatial.distance import cosine

from model_setup import load_model, get_transform
from config import MODEL_PATH, DATABASE_FILE, MATCH_THRESHOLD

# ── Load model and database ────────────────────────────────────────
print("Loading model and database...")
model     = load_model()
transform = get_transform()
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

with open(DATABASE_FILE, "rb") as f:
    db = pickle.load(f)

if len(db) < 2:
    print("ERROR: Need at least 2 enrolled users to test impersonation.")
    print(f"Currently enrolled: {list(db.keys())}")
    sys.exit(1)

users = list(db.keys())
print(f"\nEnrolled users: {users}")
print(f"Match threshold: {MATCH_THRESHOLD}")

# ── Helper: get embedding from image array ─────────────────────────
def get_embedding(img_bgr):
    pil  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t    = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.vit(t)
    return out.last_hidden_state[0, 0].cpu().numpy()

def get_embedding_tensor(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return transform(pil).unsqueeze(0).to(device)

# ── Targeted FGSM: move embedding TOWARD victim ────────────────────
def targeted_fgsm(tensor, victim_embedding, eps, steps=1):
    """
    Targeted attack: minimise distance between attacker embedding
    and victim embedding. Uses gradient descent on embedding space.
    """
    victim_t = torch.tensor(victim_embedding, dtype=torch.float32).to(device)
    t = tensor.clone().detach().requires_grad_(True)

    for _ in range(steps):
        out      = model.vit(t)
        emb      = out.last_hidden_state[0, 0]
        # Loss: cosine distance to victim (minimise = move toward victim)
        loss     = 1.0 - torch.nn.functional.cosine_similarity(
                        emb.unsqueeze(0), victim_t.unsqueeze(0))
        loss.backward()
        # Step TOWARD victim (negative gradient = descend toward target)
        t = torch.clamp(t - eps * t.grad.sign(), -1, 1).detach()
        t.requires_grad_(True)

    return t.detach()

def get_emb_from_tensor(tensor):
    with torch.no_grad():
        out = model.vit(tensor)
    return out.last_hidden_state[0, 0].cpu().numpy()

# ── Capture one frame from webcam ─────────────────────────────────
def capture_face(label):
    print(f"\n[CAMERA] Show {label} face. Press SPACE to capture, Q to quit.")
    cap = cv2.VideoCapture(0)
    img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Show: {label} | SPACE=capture Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            img = frame.copy()
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return img

# ══════════════════════════════════════════════════════════════════
# TEST SETUP
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("IMPERSONATION ATTACK TEST")
print("="*60)
print("\nWho is the ATTACKER? (person trying to impersonate)")
for i, u in enumerate(users):
    print(f"  {i}: {u}")

try:
    atk_idx = int(input("Enter attacker index: "))
    vic_idx = int(input("Enter victim index (person to impersonate): "))
    attacker_name = users[atk_idx]
    victim_name   = users[vic_idx]
except (ValueError, IndexError):
    print("Invalid input. Using first two users.")
    attacker_name, victim_name = users[0], users[1]

print(f"\nATTACKER: {attacker_name}")
print(f"VICTIM:   {victim_name}")

# Victim's mean embedding from database
victim_emb = np.mean(db[victim_name], axis=0)

# ── Capture attacker's live face ───────────────────────────────────
atk_img = capture_face(f"ATTACKER ({attacker_name})")
if atk_img is None:
    print("No image captured. Exiting.")
    sys.exit(1)

# ── Baseline embeddings ────────────────────────────────────────────
atk_emb_clean = get_embedding(atk_img)
atk_tensor    = get_embedding_tensor(atk_img)

sim_clean_to_victim   = 1 - cosine(atk_emb_clean, victim_emb)
sim_clean_to_self     = 1 - cosine(atk_emb_clean,
                                    np.mean(db[attacker_name], axis=0))

print("\n" + "="*60)
print("BASELINE (no attack)")
print("="*60)
print(f"Attacker→Victim similarity:   {sim_clean_to_victim:.4f}  "
      f"{'WOULD GRANT (!)' if sim_clean_to_victim >= MATCH_THRESHOLD else 'DENIED ✓'}")
print(f"Attacker→Self similarity:     {sim_clean_to_self:.4f}")
print(f"Gap to threshold:             {MATCH_THRESHOLD - sim_clean_to_victim:.4f}")

# ── Test across ε values ───────────────────────────────────────────
eps_values = [0.005, 0.010, 0.020, 0.050, 0.100]

print("\n" + "="*60)
print("TARGETED FGSM IMPERSONATION ATTACK")
print("="*60)
print(f"{'ε':>8} | {'Atk→Victim SIM':>16} | {'Atk→Self SIM':>14} | {'Result':>20} | {'Δ toward victim':>16}")
print("-"*80)

results = []
for eps in eps_values:
    adv_tensor = targeted_fgsm(atk_tensor, victim_emb, eps, steps=1)
    adv_emb    = get_emb_from_tensor(adv_tensor)

    sim_to_victim = 1 - cosine(adv_emb, victim_emb)
    sim_to_self   = 1 - cosine(adv_emb, np.mean(db[attacker_name], axis=0))
    delta         = sim_to_victim - sim_clean_to_victim
    granted       = sim_to_victim >= MATCH_THRESHOLD

    results.append({
        "eps": eps,
        "sim_victim": sim_to_victim,
        "sim_self":   sim_to_self,
        "granted":    granted,
        "delta":      delta
    })

    status = "⚠ IMPERSONATED" if granted else "DENIED ✓"
    print(f"{eps:>8.3f} | {sim_to_victim:>16.4f} | {sim_to_self:>14.4f} | "
          f"{status:>20} | {delta:>+16.4f}")

# ── Multi-step attack (stronger) ──────────────────────────────────
print("\n" + "="*60)
print("MULTI-STEP TARGETED FGSM (10 steps, ε=0.010)")
print("="*60)
adv_tensor_ms = targeted_fgsm(atk_tensor, victim_emb, eps=0.010, steps=10)
adv_emb_ms    = get_emb_from_tensor(adv_tensor_ms)
sim_ms        = 1 - cosine(adv_emb_ms, victim_emb)
print(f"Similarity to victim after 10 steps: {sim_ms:.4f}  "
      f"{'⚠ IMPERSONATED' if sim_ms >= MATCH_THRESHOLD else 'DENIED ✓'}")

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
any_granted = any(r["granted"] for r in results)
min_eps_granted = next((r["eps"] for r in results if r["granted"]), None)
max_delta = max(r["delta"] for r in results)

print(f"Impersonation succeeded:        {'YES ⚠' if any_granted else 'NO ✓'}")
if min_eps_granted:
    print(f"Minimum ε needed to impersonate: {min_eps_granted}")
else:
    print(f"Max similarity achieved:         {max(r['sim_victim'] for r in results):.4f}  "
          f"(threshold={MATCH_THRESHOLD})")
print(f"Max embedding shift toward victim: {max_delta:+.4f}")
print(f"\nConclusion: Attacker needed ε={'N/A — failed' if not any_granted else min_eps_granted} "
      f"to cross threshold.")
print("At ε > 0.02 the image is visibly distorted — detectable by human inspection.")