import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import timm
from huggingface_hub import login
import os
import traceback
from tqdm import tqdm

# ── Fix critique Windows/CUDA : erreurs synchrones et lisibles ────────────────
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Authentification HuggingFace
login(token=os.environ.get("HF_TOKEN"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation : {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Chargement du dataset PatchCamelyon...")
dataset = load_dataset("1aurent/PatchCamelyon")
print(f"Train      : {len(dataset['train'])} images")
print(f"Validation : {len(dataset['validation'])} images")
print(f"Test       : {len(dataset['test'])} images")

# ── Transformations ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

def preprocess(batch):
    batch["image"] = [transform(img.convert("RGB")) for img in batch["image"]]
    return batch

dataset = dataset.with_transform(preprocess)

# ── Collate function : remplace torch.stack ───────────────────────────────────
def collate_fn(batch):
    # batch = liste de dicts {"image": tensor, "label": int}
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return images, labels

# ── Modèle UNI ────────────────────────────────────────────────────────────────
print("Chargement du modèle UNI...")
backbone = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True
)

for param in backbone.parameters():
    param.requires_grad = False

class HistoClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

model = HistoClassifier(backbone).to(device)
print("Modèle prêt !")

# ── DataLoaders ───────────────────────────────────────────────────────────────
# num_workers=0 est OBLIGATOIRE sur Windows pour éviter les crashes silencieux
train_loader = DataLoader(
    dataset["train"],
    batch_size=32,
    shuffle=True,
    num_workers=0,        # ← fix Windows
    collate_fn=collate_fn,
    pin_memory=True,      # accélère le transfert CPU→GPU
)
val_loader = DataLoader(
    dataset["validation"],
    batch_size=32,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True,
)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
NUM_EPOCHS = 3

# ── Boucle d'entraînement ─────────────────────────────────────────────────────
try:
    for epoch in range(NUM_EPOCHS):
        # — Train —
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        avg_loss  = total_loss / len(train_loader)

        # — Validation —
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} — "
              f"Loss: {avg_loss:.4f} — "
              f"Train Acc: {train_acc:.2f}% — "
              f"Val Acc: {val_acc:.2f}%")

    # Sauvegarde uniquement si l'entraînement s'est terminé normalement
    torch.save(model.state_dict(), "histo_classifier.pth")
    print("✅ Modèle sauvegardé dans histo_classifier.pth")

except Exception as e:
    print(f"\n❌ Erreur pendant l'entraînement : {e}")
    traceback.print_exc()