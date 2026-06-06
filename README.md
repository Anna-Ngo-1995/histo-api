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

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Vérification que l'API fonctionne |
| POST | `/predict` | Envoie une image, reçoit une prédiction |

## Documentation interactive

Accessible sur `http://127.0.0.1:8000/docs`