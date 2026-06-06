from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
from io import BytesIO
# io est un module Python intégré qui gère les flux de données en mémoire.
# BytesIO crée un fichier virtuel en mémoire — c'est-à-dire qu'il simule un fichier sans rien écrire sur le disque.
# Pourquoi on en a besoin ici ?
# Quand l'API reçoit une image envoyée par un utilisateur, elle arrive sous forme de bytes (une suite de nombres bruts) — pas comme un fichier .jpg sur le disque. Or Pillow (Image.open()) attend un fichier.
# BytesIO fait le pont entre les deux :
# pythoncontents = await file.read()      # bytes bruts reçus par l'API
# image = Image.open(io.BytesIO(contents))  # on fait croire à Pillow que c'est un fichier

# Charger un modèle pré-entraîné
model_name = "google/vit-base-patch16-224"

print("Chargement du modèle...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
print("Modèle chargé !")

# # Télécharger une image test
# url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
# response = requests.get(url)
# image = Image.open(BytesIO(response.content)).convert("RGB")
# print("Image chargée !")
# Charger l'image locale
image = Image.open("test.jpg").convert("RGB")
print("Image chargée !")
# Faire une prédiction
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
label = model.config.id2label[predicted_class]

print(f"Prédiction : {label}")