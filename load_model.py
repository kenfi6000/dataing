from transformers import pipeline

MODEL_PATH = "../models/zero_shot_model"

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

print("Local model loaded successfully.")