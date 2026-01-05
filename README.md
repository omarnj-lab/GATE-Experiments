# Arabic Text Embedding

This repository contains the code, data, and evaluation scripts for training Arabic text embedding models using Matryoshka representation learning and hybrid loss training.

## Overview

This work presents state-of-the-art Arabic text embedding models trained using:
- **Matryoshka Loss**: Enables nested embedding learning at multiple resolutions
- **Multiple Negatives Ranking Loss**: Enhances discrimination between similar and dissimilar text pairs
- **Hybrid Training**: Combines both loss functions for optimal performance

## Repository Structure

```
.
├── data/
│   ├── train/              # Training datasets
│   │   ├── train_nli.csv   # Arabic NLI triplet training data
│   │   └── train_sts.csv   # Arabic STS training data
│   ├── test/               # Test datasets
│   │   ├── test_nli.csv    # Arabic NLI triplet test data
│   │   └── test_sts.csv    # Arabic STS test data
│   ├── samples/            # Sample data files (JSON format)
│   │   ├── nli_train_samples.json
│   │   ├── nli_test_samples.json
│   │   ├── sts_train_samples.json
│   │   └── sts_test_samples.json
│   ├── README_NLI.md       # NLI dataset documentation
│   └── README_STS.md       # STS dataset documentation
├── training/
│   └── matryoshka_nli.py   # Main training script
├── evaluation/             # Evaluation scripts
│   ├── evaluate_sts.py     # STS benchmark evaluation
│   └── evaluate_mteb.py   # MTEB evaluation (STS17, STS22.v2)
├── inference/              # Inference scripts
│   └── encode_sentences.py # Sentence encoding and similarity
├── models/                 # Model checkpoints (to be added)
└── README.md               # This file
```

## Datasets

### Arabic NLI Triplet Dataset

The Arabic version of SNLI and MultiNLI datasets (triplet subset). Contains anchor-positive-negative triplets for training semantic similarity models.

- **Format**: CSV with columns: `anchor`, `positive`, `negative`
- **Training samples**: ~557,852 triplets (large file not included in repository due to size limits)
- **Test samples**: ~6,611 triplets

See `data/README_NLI.md` for more details.

### Arabic STS Dataset

The Arabic version of the Semantic Textual Similarity Benchmark. Used for evaluating semantic similarity models.

- **Format**: CSV with columns: `sentence1`, `sentence2`, `score`
- **Training samples**: ~5,751 pairs
- **Test samples**: ~1,381 pairs
- **Score range**: 0.0 to 1.0

See `data/README_STS.md` for more details.

## Training

### Requirements

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install sentence-transformers datasets torch mteb scikit-learn
```

### Usage

Train a model using the default base model:

```bash
cd training
python matryoshka_nli.py
```

Or specify a custom base model:

```bash
python matryoshka_nli.py path/to/base/model
```

### Training Configuration

The training script uses the following default settings:
- **Base model**: Specify via command line argument or modify default in script
- **Batch size**: 128
- **Epochs**: 3
- **Matryoshka dimensions**: [768, 512, 256, 128, 64]
- **Mixed precision**: FP16 enabled

### Training Process

1. Loads Arabic NLI triplet data from `data/train/train_nli.csv`
2. Trains using MatryoshkaLoss with MultipleNegativesRankingLoss
3. Evaluates on STS validation set during training
4. Final evaluation on STS test set
5. Saves model to `output/` directory

## Models

The framework can be used to train Arabic text embedding models that achieve state-of-the-art performance on Arabic semantic textual similarity benchmarks.

## Evaluation

### STS Benchmark Evaluation

Evaluate a trained model on the Arabic STS test set:

```bash
cd evaluation
python evaluate_sts.py ../models/my_model ../data/test/test_sts.csv 768
```

### MTEB Evaluation

Evaluate models on STS17 and STS22.v2 benchmarks using MTEB:

```bash
cd evaluation
python evaluate_mteb.py ../models/my_model ../results
```

This will:
- Evaluate on STS17 and STS22.v2 benchmarks
- Compute average scores across both tasks
- Save results to JSON file

The script supports Matryoshka dimensions and will evaluate at different embedding sizes if specified.

## Inference

### Encoding Sentences

Use trained models to encode sentences and compute similarities:

```bash
cd inference
python encode_sentences.py
```

Or use in your code:

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

# Compute similarity
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)  # [3, 3]
```

## Data Samples

Sample data files in JSON format are available in `data/samples/` for quick inspection:
- `nli_train_samples.json`: 100 NLI training samples
- `nli_test_samples.json`: 50 NLI test samples
- `sts_train_samples.json`: 100 STS training samples
- `sts_test_samples.json`: 50 STS test samples

To generate new samples:

```bash
python create_samples.py
```

## License

This work is licensed under the Apache 2.0 License.

