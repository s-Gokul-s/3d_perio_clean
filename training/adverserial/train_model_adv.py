import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
from tqdm import tqdm

# =============================
# CONFIGURATION
# =============================
IMAGE_DIR = "dataset/images"
BATCH_SIZE = 13
EPOCHS = 3   # Only fine-tuning
LR = 2e-5
EPSILON = 0.005

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================
# DATASET
# =============================
class PeriocularDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = []
        self.labels = []

        files = sorted([f for f in os.listdir(image_dir)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
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
        return img, self.labels[idx]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# =============================
# DATA LOADERS
# =============================
full_dataset = PeriocularDataset(IMAGE_DIR)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_raw, val_raw = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    TransformedDataset(train_raw, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TransformedDataset(val_raw, val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =============================
# LOAD CLEAN MODEL (IMPORTANT)
# =============================
model = ViTForImageClassification.from_pretrained(
    "training/models/periocular_vit_best"
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# =============================
# TRAINING
# =============================
best_val_acc = 0.0
os.makedirs("training/models/adv_model", exist_ok=True)

print(f"\nAdversarial Fine-Tuning Starting...")

total_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        images.requires_grad = True

        # ---- 1️⃣ Get gradient for FGSM ----
        outputs = model(images).logits
        loss_for_grad = criterion(outputs, labels)

        grad = torch.autograd.grad(
            loss_for_grad,
            images,
            retain_graph=False,
            create_graph=False
        )[0]

        # ---- 2️⃣ Generate adversarial images ----
        adv_images = images + EPSILON * grad.sign()
        adv_images = torch.clamp(adv_images, -1, 1).detach()

        # ---- 3️⃣ Compute clean loss ----
        clean_outputs = model(images).logits
        loss_clean = criterion(clean_outputs, labels)

        # ---- 4️⃣ Compute adversarial loss ----
        adv_outputs = model(adv_images).logits
        loss_adv = criterion(adv_outputs, labels)

        # ---- 5️⃣ Combine ----
        total_batch_loss = loss_clean + loss_adv

        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        progress_bar.set_postfix(loss=total_batch_loss.item())

    # =============================
    # VALIDATION
    # =============================
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    epoch_time = time.time() - epoch_start

    print(f"\nEpoch {epoch+1} Completed")
    print(f"Val Accuracy: {val_acc*100:.2f}%")
    print(f"Epoch Time: {epoch_time/60:.2f} minutes\n")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("training/models/adv_model/periocular_vit_adv_best")
        print(f"⭐ New Best Adversarial Model Saved!\n")

total_time = time.time() - total_start_time

print(f"\n✅ Adversarial Fine-Tuning Finished")
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"Total Training Time: {total_time/60:.2f} minutes")