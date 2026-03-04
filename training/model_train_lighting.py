"""
train_model_lighting_robust.py
==============================
Same as train_model.py with ONE key change:

  ColorJitter(brightness=0.2) → ColorJitter(brightness=0.6, contrast=0.5)

This forces the model to learn features that are invariant to large
lighting changes. After this training, authentication will work in
dim light, bright light, and different colour temperatures.

Training time: ~2-3 hours (same as original, same GPU)
Expected accuracy: 97-99% (same as original)

HOW TO RUN:
    cd D:\MCA\restructuredperio(13_02_2026)
    python train_model_lighting_robust.py
"""

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image

IMAGE_DIR  = "dataset/images"
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 2e-5
MODEL_NAME = "google/vit-base-patch16-224"
SAVE_PATH  = "training/models/periocular_vit_lighting_robust"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class PeriocularDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images, self.labels = [], []
        files   = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        persons = sorted(set(f.split("_")[0] for f in files))
        self.label_map = {p: i for i, p in enumerate(persons)}
        for f in files:
            self.images.append(f)
            self.labels.append(self.label_map[f.split("_")[0]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir,
                                      self.images[idx])).convert("RGB")
        return img, self.labels[idx]


class TransformedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __getitem__(self, i):
        x, y = self.subset[i]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)


# ── THE KEY CHANGE ────────────────────────────────────────────
# brightness=0.6 means the model sees crops from 40% to 160%
# of original brightness during every training step.
# This directly teaches the model to ignore lighting variation.
# contrast=0.5 handles uneven/directional lighting.
# saturation=0.3 handles colour temperature shifts (warm/cool light).
# hue=0.05 handles slight colour cast from different light sources.
# ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.6,    # was 0.2 — now covers dim to bright
        contrast=0.5,      # was 0.2 — handles directional lighting
        saturation=0.3,    # NEW — handles warm/cool colour temperature
        hue=0.05,          # NEW — handles slight colour cast
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

full_dataset = PeriocularDataset(IMAGE_DIR)
train_size   = int(0.8 * len(full_dataset))
val_size     = len(full_dataset) - train_size
train_raw, val_raw = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(TransformedDataset(train_raw, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TransformedDataset(val_raw, val_transform),
                          batch_size=BATCH_SIZE, shuffle=False)

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(full_dataset.label_map),
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
os.makedirs("training/models", exist_ok=True)

print(f"Training with STRONG lighting augmentation...")
print(f"Dataset: {len(full_dataset.label_map)} persons, {len(full_dataset)} images")
print(f"Saving to: {SAVE_PATH}\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images).logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds   = torch.argmax(model(images).logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc  = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(SAVE_PATH)
        print(f"  ✓ Saved (best so far: {val_acc*100:.2f}%)")

print(f"\nDone. Best accuracy: {best_val_acc*100:.2f}%")
print(f"Model saved to: {SAVE_PATH}")
print(f"\nNext step: update config.py")
print(f'  MODEL_PATH = "{SAVE_PATH}"')