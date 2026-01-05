"""
Inference script for Arabic text embedding models.

This script demonstrates how to use trained embedding models to encode sentences
and compute semantic similarity scores.
"""

from sentence_transformers import SentenceTransformer

# Load model - replace with your model path
model_path = "../models/your_model"  # Update this path
model = SentenceTransformer(model_path)

# Example sentences in Arabic
sentences = [
    "شخص على حصان يقفز فوق طائرة معطلة",
    "شخص في الهواء الطلق، على حصان.",
    "شخص في مطعم، يطلب عجة."
]

# Encode sentences
print("Encoding sentences...")
embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")
# Expected output: [3, 768]

# Compute similarity scores
print("\nComputing similarity scores...")
similarities = model.similarity(embeddings, embeddings)
print(f"Similarities shape: {similarities.shape}")
# Expected output: [3, 3]

print("\nSimilarity matrix:")
print(similarities)

# Example: Find most similar sentence pairs
print("\nMost similar pairs:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"Sentence {i} <-> Sentence {j}: {similarities[i][j]:.4f}")

