"""
Evaluation script for Arabic Semantic Textual Similarity (STS) tasks.

This script evaluates trained embedding models on the Arabic STS benchmark datasets.
"""

import logging
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from datasets import Dataset

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M-%S", level=logging.INFO)

def load_sts_dataset(csv_path):
    """Load STS dataset from CSV file."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "score": float(row["score"])
            })
    return Dataset.from_list(data)

def evaluate_model(model_path, test_data_path, output_dim=768):
    """
    Evaluate a model on STS test data.
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to the test STS CSV file
        output_dim: Dimension to use for evaluation (for Matryoshka models)
    """
    logging.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    logging.info(f"Loading test data from {test_data_path}")
    test_dataset = load_sts_dataset(test_data_path)
    
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        scores=test_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-test",
        truncate_dim=output_dim if output_dim < 768 else None,
    )
    
    logging.info(f"Evaluating model at dimension {output_dim}")
    score = evaluator(model)
    logging.info(f"STS Test Score (Spearman correlation): {score:.4f}")
    
    return score

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_sts.py <model_path> [test_data_path] [output_dim]")
        print("Example: python evaluate_sts.py ../models/my_model ../data/test/test_sts.csv 768")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data_path = sys.argv[2] if len(sys.argv) > 2 else "../data/test/test_sts.csv"
    output_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 768
    
    evaluate_model(model_path, test_data_path, output_dim)

