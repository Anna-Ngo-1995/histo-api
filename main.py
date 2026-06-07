from fastapi import FastAPI, File, UploadFile
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
import timm
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
import os
login(token=os.environ.get("HF_TOKEN"))
# Créer l'application FastAPI
app = FastAPI()

## Charger le modèle une seule fois au démarrage

# print("Chargement du modèle...")
# model_name = "google/vit-base-patch16-224"
# processor = AutoImageProcessor.from_pretrained(model_name)
# model = AutoModelForImageClassification.from_pretrained(model_name)
# print("Modèle prêt !")

# Charger le modèle UNI
print("Chargement du modèle UNI...")

model = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True
)
model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

print("Modèle UNI prêt !")
# Route de test pour vérifier que l'API fonctionne
@app.get("/")
def home():
    return {"message": "API Histopathologie opérationnelle"}

# Route principale : reçoit une image et retourne une prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code = 400,
            detail = "Format non supporté. Envoyez une image JPG ou PNG"
        )
    # Lire l'image envoyée
    contents = await file.read()
    # Vérifier que l'imgae est lisible
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail ="Impossible de lire l'image. FIchier corrompu?"
        )

    # # Faire la prédiction
    # inputs = processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # predicted_class = outputs.logits.argmax(-1).item()
    # label = model.config.id2label[predicted_class]

    # Préparer l'image pour UNI
    img_tensor = transform(image).unsqueeze(0)

    # Extraire les features avec UNI
    with torch.inference_mode():
        features = model(img_tensor)


    # probabilities = F.softmax(outputs.logits, dim=-1)
    # confidence = probabilities[0][predicted_class].item()
    # confidence_pct = round(confidence * 100, 2)
    return {
        "message": "Features extraites avec succès",
        "modele": "UNI - Nature Medicine (MahmoodLab)",
        "taille_features": features.shape[1],
        "filename": file.filename
        # "prediction": label,
        # "confidence": f"{confidence_pct}%",
        # "filename": file.filename
    }