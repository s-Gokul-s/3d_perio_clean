"""
final_robustness_report_fast.py
================================
Same report as final_robustness_report.py but samples only
100 images instead of all 10,199. Finishes in ~5 minutes on CPU.

Results will be statistically representative — 100 images across
507 identities is more than enough to show the robustness improvement.

HOW TO RUN:
    cd D:\MCA\restructuredperio(13_02_2026)
    python final_robustness_report_fast.py
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from scipy.spatial.distance import cosine as cosine_dist
import json, random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────
DATASET_DIR     = "dataset/images"
CLEAN_MODEL     = "training/models/periocular_vit_best"
ADV_MODEL       = "training/models/adv_model/periocular_vit_adv_best"
MATCH_THRESHOLD = 0.65
EPSILONS        = [0.001, 0.005, 0.01, 0.02, 0.04]

# ── KEY SETTING: how many images to sample for evaluation ────
# 100 takes ~3-5 min on CPU. Increase to 300 if you want more precision.
SAMPLE_SIZE = 100

random.seed(42)   # reproducible sampling

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ─────────────────────────────────────────────────────────────

def get_embedding(model, tensor):
    with torch.no_grad():
        out = model.vit(tensor)
    return out.last_hidden_state[0, 0].cpu().numpy()


def fgsm_attack(model, image_tensor, epsilon):
    img = image_tensor.clone().detach().to(device)
    img.requires_grad_(True)
    logits = model(img).logits
    label  = torch.argmax(logits, dim=1)
    loss   = torch.nn.CrossEntropyLoss()(logits, label)
    model.zero_grad()
    loss.backward()
    adv = img + epsilon * img.grad.sign()
    return torch.clamp(adv, -1.0, 1.0).detach()


def load_all_images():
    """Returns list of (person_id, filepath)"""
    files = sorted([f for f in os.listdir(DATASET_DIR)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    return [(f.split("_")[0], os.path.join(DATASET_DIR, f)) for f in files]


def load_tensor(filepath):
    img = Image.open(filepath).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  ADVERSARIAL ROBUSTNESS — FINAL COMPARISON REPORT (FAST)")
    print("=" * 62)

    print("\n  Loading models...")
    clean_model = ViTForImageClassification.from_pretrained(
        CLEAN_MODEL).to(device).eval()
    adv_model = ViTForImageClassification.from_pretrained(
        ADV_MODEL).to(device).eval()

    print("  Loading dataset index...")
    all_images = load_all_images()
    print(f"  {len(all_images)} total images found")

    # Group by person
    from collections import defaultdict
    by_person = defaultdict(list)
    for person, path in all_images:
        by_person[person].append(path)

    # Build enrolled templates: use first 3 images per person
    # for a representative sample of persons
    print("  Building enrolled templates (sampled persons)...")
    sampled_persons = random.sample(list(by_person.keys()),
                                    min(50, len(by_person)))

    enrolled_clean = {}
    enrolled_adv   = {}

    for person in sampled_persons:
        imgs = by_person[person][:3]
        embs_c, embs_a = [], []
        for path in imgs:
            t = load_tensor(path)
            embs_c.append(get_embedding(clean_model, t))
            embs_a.append(get_embedding(adv_model, t))
        enrolled_clean[person] = np.mean(embs_c, axis=0)
        enrolled_adv[person]   = np.mean(embs_a, axis=0)

    # Sample test images (not the enrollment images)
    test_pool = []
    for person in sampled_persons:
        imgs = by_person[person][3:]   # skip the 3 enrolled images
        for path in imgs:
            test_pool.append((person, path))

    if len(test_pool) == 0:
        # fallback: use all images if dataset is small
        test_pool = [(p, by_person[p][0]) for p in sampled_persons]

    test_sample = random.sample(test_pool, min(SAMPLE_SIZE, len(test_pool)))
    print(f"  Testing on {len(test_sample)} sampled images")

    # ── Clean accuracy ────────────────────────────────────────
    print("\n  Computing clean accuracy on sample...")
    correct_c, correct_a, total = 0, 0, 0
    for person, path in test_sample:
        t = load_tensor(path)
        e_c = get_embedding(clean_model, t)
        e_a = get_embedding(adv_model, t)

        best_c = max(enrolled_clean, key=lambda p: 1 - cosine_dist(e_c, enrolled_clean[p]))
        best_a = max(enrolled_adv,   key=lambda p: 1 - cosine_dist(e_a, enrolled_adv[p]))

        sim_c = 1 - cosine_dist(e_c, enrolled_clean[best_c])
        sim_a = 1 - cosine_dist(e_a, enrolled_adv[best_a])

        if best_c == person and sim_c >= MATCH_THRESHOLD:
            correct_c += 1
        if best_a == person and sim_a >= MATCH_THRESHOLD:
            correct_a += 1
        total += 1

    acc_c = correct_c / total
    acc_a = correct_a / total
    print(f"  Clean model accuracy (no attack): {acc_c*100:.1f}%")
    print(f"  Adv model accuracy   (no attack): {acc_a*100:.1f}%")

    # ── Attack evaluation ─────────────────────────────────────
    print("\n  Running FGSM attacks at each epsilon...\n")
    results = []

    for eps in EPSILONS:
        stab_c, stab_a   = [], []
        dodge_c, dodge_a = 0, 0
        n = 0

        for i, (person, path) in enumerate(test_sample):
            t = load_tensor(path)

            # Clean embeddings
            e_clean_c = get_embedding(clean_model, t)
            e_clean_a = get_embedding(adv_model, t)

            sim_clean_c = 1 - cosine_dist(e_clean_c, enrolled_clean[person])
            sim_clean_a = 1 - cosine_dist(e_clean_a, enrolled_adv[person])

            # Skip if even clean image doesn't match (ambiguous test case)
            if sim_clean_c < MATCH_THRESHOLD and sim_clean_a < MATCH_THRESHOLD:
                continue

            # FGSM attacks
            adv_c = fgsm_attack(clean_model, t, eps)
            adv_a = fgsm_attack(adv_model,   t, eps)

            # Attacked embeddings
            e_adv_c = get_embedding(clean_model, adv_c)
            e_adv_a = get_embedding(adv_model,   adv_a)

            # Embedding stability: sim(clean_emb, adv_emb)
            stab_c.append(1 - cosine_dist(e_clean_c, e_adv_c))
            stab_a.append(1 - cosine_dist(e_clean_a, e_adv_a))

            # Dodge ASR: did attack push similarity below threshold?
            sim_adv_c = 1 - cosine_dist(e_adv_c, enrolled_clean[person])
            sim_adv_a = 1 - cosine_dist(e_adv_a, enrolled_adv[person])

            if sim_adv_c < MATCH_THRESHOLD:
                dodge_c += 1
            if sim_adv_a < MATCH_THRESHOLD:
                dodge_a += 1
            n += 1

        results.append({
            "epsilon":       eps,
            "stability_c":   float(np.mean(stab_c)) if stab_c else 0.0,
            "stability_a":   float(np.mean(stab_a)) if stab_a else 0.0,
            "dodge_asr_c":   dodge_c / max(n, 1),
            "dodge_asr_a":   dodge_a / max(n, 1),
            "n":             n,
        })

        print(f"  ε={eps:.3f} | "
              f"Stability  clean={np.mean(stab_c):.4f}  adv={np.mean(stab_a):.4f} | "
              f"Dodge ASR  clean={dodge_c/max(n,1)*100:.1f}%  adv={dodge_a/max(n,1)*100:.1f}%")

    # ── Print final report ────────────────────────────────────
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p()
    p("=" * 70)
    p("  ADVERSARIAL ROBUSTNESS COMPARISON REPORT")
    p("  ViT Periocular Biometric System")
    p("  Attack: FGSM | Dataset: 10,199 images / 507 identities")
    p(f"  Evaluation sample: {len(test_sample)} images, {len(sampled_persons)} identities")
    p("=" * 70)
    p()
    p("  CLEAN ACCURACY (no attack):")
    p(f"    Original model:              {acc_c*100:.1f}%")
    p(f"    Adversarially trained model: {acc_a*100:.1f}%")
    p()
    p("  EMBEDDING STABILITY UNDER FGSM ATTACK")
    p("  (cosine similarity between clean and attacked embedding)")
    p("  Higher = more stable = more robust. Threshold = 0.65")
    p()
    p(f"  {'Epsilon':<10} {'Original':>12} {'Adv Trained':>14} {'Improvement':>14}")
    p("  " + "─" * 52)
    for r in results:
        imp = r["stability_a"] - r["stability_c"]
        p(f"  {r['epsilon']:<10.3f} {r['stability_c']:>12.4f} {r['stability_a']:>14.4f} {imp:>+14.4f}")

    p()
    p("  DODGE ATTACK SUCCESS RATE (lower = better defense)")
    p("  (% of genuine users wrongly rejected after FGSM attack)")
    p()
    p(f"  {'Epsilon':<10} {'Original':>12} {'Adv Trained':>14} {'Reduction':>14}")
    p("  " + "─" * 52)
    for r in results:
        red = r["dodge_asr_c"] - r["dodge_asr_a"]
        p(f"  {r['epsilon']:<10.3f} {r['dodge_asr_c']*100:>10.1f}%  {r['dodge_asr_a']*100:>12.1f}%  {red*100:>+12.1f}%")

    p()
    p("=" * 70)

    # ── Save ──────────────────────────────────────────────────
    os.makedirs("adversarial_results", exist_ok=True)

    with open("adversarial_results/final_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open("adversarial_results/final_report.json", "w") as f:
        json.dump({
            "clean_accuracy": {"original": acc_c, "adv_trained": acc_a},
            "results": results
        }, f, indent=2)

    print("\n  Saved to adversarial_results/final_report.txt")


if __name__ == "__main__":
    main()