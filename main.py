from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from huggingface_hub import login
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# Authentification HuggingFace
login(token=os.environ.get("HF_TOKEN"))

# Créer l'application FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation : {device}")

# ── Tête de classification (identique à train.py) ─────────────────────────────
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

# ── Charger le modèle ─────────────────────────────────────────────────────────
print("Chargement du modèle UNI...")
backbone = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True
)

model = HistoClassifier(backbone)
model.load_state_dict(torch.load("histo_classifier.pth", map_location=device))
model.to(device)
model.eval()
print("Modèle prêt !")

# Labels
LABELS = {0: "Pas de tumeur", 1: "Tumeur détectée"}

# ── Transformations ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "API Histopathologie - UNI (Nature Medicine) + PatchCamelyon",
        "modele": "MahmoodLab/UNI",
        "tache": "Classification tumeur / pas tumeur",
        "accuracy_validation": "92.05%"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Vérifier le type de fichier
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Envoyez une image JPG ou PNG."
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Impossible de lire l'image. Fichier corrompu ?"
        )

    # Préparer l'image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prédiction
    with torch.inference_mode():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=-1)
        predicted_class = probabilities.argmax(dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "prediction": LABELS[predicted_class],
        "confidence": f"{round(confidence * 100, 2)}%",
        "filename": file.filename,
        "modele": "UNI - Nature Medicine"
    }