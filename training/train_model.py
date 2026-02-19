import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image

# =============================
# CONFIGURATION
# =============================
IMAGE_DIR = "dataset/images"
BATCH_SIZE = 16  # Increased for stability if VRAM allows, otherwise keep 8
EPOCHS = 10      # Increased to allow for slower, better learning
LR = 2e-5        # Lower LR is much better for fine-tuning Transformers
MODEL_NAME = "google/vit-base-patch16-224"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================
# DATASET CLASS (Updated to handle separate transforms)
# =============================
class PeriocularDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []

        files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        persons = sorted(list(set(f.split("_")[0] for f in files)))
        self.label_map = {p: i for i, p in enumerate(persons)}

        for f in files:
            self.images.append(f)
            self.labels.append(self.label_map[f.split("_")[0]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# =============================
# TRANSFORMS (The "Defense" against Overfitting)
# =============================
# We apply augmentation to TRAIN but not to VAL
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Eyes are symmetrical
    transforms.RandomRotation(15),     # Account for head tilts
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Lighting variations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Standardization
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create Dataset and Split
full_dataset = PeriocularDataset(IMAGE_DIR)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_raw, val_raw = random_split(full_dataset, [train_size, val_size])

# Apply specific transforms to each split
class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x) # PIL to Tensor happens here
        return x, y
    def __len__(self):
        return len(self.subset)

train_loader = DataLoader(TransformedDataset(train_raw, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TransformedDataset(val_raw, val_transform), batch_size=BATCH_SIZE, shuffle=False)

# =============================
# MODEL & OPTIMIZER
# =============================
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(full_dataset.label_map),
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# =============================
# TRAINING LOOP
# =============================
best_val_acc = 0.0
os.makedirs("training/models", exist_ok=True)

print(f"\nTraining starting... Targets: {len(full_dataset.label_map)} persons")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # SAVE BEST MODEL ONLY
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("training/models/periocular_vit_best")
        print(f"⭐ New Best Model Saved! Accuracy: {val_acc*100:.2f}%")

print(f"\n✅ Training Finished. Best Accuracy: {best_val_acc*100:.2f}%")