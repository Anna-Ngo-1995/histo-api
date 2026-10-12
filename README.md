# API de classification d'images histopathologiques

API REST développée dans le cadre d'un projet d'IA appliquée à l'imagerie médicale.

## Description

Ce projet expose un modèle de deep learning (Vision Transformer - ViT) via une API REST FastAPI.
L'API reçoit une image et retourne une prédiction de classification.

**Contexte** : application à l'analyse d'images histopathologiques pour la recherche en oncologie.

## Technologies

- Python 3.11
- FastAPI
- PyTorch
- HuggingFace Transformers (ViT)
- Uvicorn

## Installation

```bash
pip install -r requirements.txt
```

## Lancer l'API

```bash
uvicorn main:app --reload
```

L'API est accessible sur `http://127.0.0.1:8000`

# Interface web

Une interface web permet de tester l'API visuellement (drag & drop d'image).

Une fois l'API lancée, ouvrir dans le navigateur :
http://127.0.0.1:8000/ui

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Vérification que l'API fonctionne |
| POST | `/predict` | Envoie une image, reçoit une prédiction |

## Documentation interactive
## Lancer avec Docker

L'image ne contient ni les poids du modèle ni de token : ils sont fournis au lancement.

**Prérequis**

- Un token Hugging Face (lecture) avec accès au modèle protégé [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI).
- Le fichier `histo_classifier.pth` (environ 1,2 Go, non versionné car trop lourd pour GitHub), généré par `train.py`.

**Construire l'image**

```bash
docker build -t histo-api .
```

**Lancer le container**

```bash
read -s -p "Token HF : " HF_TOKEN; export HF_TOKEN
docker run --rm -p 8000:8000 \
  -e HF_TOKEN \
  -v "$(pwd)/histo_classifier.pth:/app/histo_classifier.pth:ro" \
  -v hf_cache:/root/.cache/huggingface \
  histo-api
```

Au premier démarrage, les poids d'UNI (environ 1,2 Go) sont téléchargés et conservés dans le volume `hf_cache`.

L'API est ensuite disponible sur `http://localhost:8000` : page de test sur `/ui`, documentation interactive sur `/docs`.

Testé dans GitHub Codespaces.
Accessible sur `http://127.0.0.1:8000/docs`