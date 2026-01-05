# Inference

This directory contains scripts for using trained Arabic text embedding models.

## encode_sentences.py

Script for encoding sentences and computing semantic similarity.

### Usage

```bash
python encode_sentences.py
```

Make sure to update the `model_path` variable in the script to point to your trained model.

### Example

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("path/to/model")

# Encode sentences
sentences = [
    "شخص على حصان يقفز فوق طائرة معطلة",
    "شخص في الهواء الطلق، على حصان.",
    "شخص في مطعم، يطلب عجة."
]
embeddings = model.encode(sentences)
print(embeddings.shape)  # [3, 768]

# Compute similarity matrix
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)  # [3, 3]
```

### Output

- **Embeddings**: NumPy array of shape `(num_sentences, embedding_dim)`
- **Similarities**: Similarity matrix of shape `(num_sentences, num_sentences)`

