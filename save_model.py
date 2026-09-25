from transformers import pipeline
import os

MODEL_NAME = "facebook/bart-large-mnli"

SAVE_PATH = "../models/zero_shot_model"

# Télécharge le modèle (sera automatiquement mis en cache)
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME
)

# Sauvegarde du modèle
classifier.model.save_pretrained(SAVE_PATH)
classifier.tokenizer.save_pretrained(SAVE_PATH)

print("Model downloaded and saved successfully.")