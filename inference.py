from transformers import pipeline

MODEL_PATH = "../models/zero_shot_model"

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

text = """
ChatGPT can help developers write Python applications using machine learning.
"""

candidate_labels = [
    "Technology",
    "Politics",
    "Sports",
    "Finance",
    "Healthcare"
]

result = classifier(
    text,
    candidate_labels
)

print("\nPrediction\n")

for label, score in zip(result["labels"], result["scores"]):
    print(f"{label}: {score:.5f}")