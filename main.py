from fastapi import FastAPI, File, UploadFile
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
# Créer l'application FastAPI
app = FastAPI()

# Charger le modèle une seule fois au démarrage
print("Chargement du modèle...")
model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
print("Modèle prêt !")

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

    # Faire la prédiction
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()
    label = model.config.id2label[predicted_class]

    return {
        "prediction": label,
        "filename": file.filename
    }