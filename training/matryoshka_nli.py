"""
Training script for Arabic text embedding models using Matryoshka Loss.

This script trains a transformer model on Arabic NLI triplet data with MatryoshkaLoss 
using MultipleNegativesRankingLoss. The model is trained at output dimensions [768, 512, 256, 128, 64].
Entailments are positive pairs and contradictions are added as hard negatives.
During training, the model is evaluated on the STS benchmark dataset at different output dimensions.

Usage:
python matryoshka_nli.py

OR
python matryoshka_nli.py pretrained_transformer_model_name
"""

import logging
import sys
import csv
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M-%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "path/to/base/model"
batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
num_train_epochs = 3
matryoshka_dims = [768, 512, 256, 128, 64]

# Data paths
data_dir = Path(__file__).parent.parent / "data"
train_nli_path = data_dir / "train" / "train_nli.csv"
test_nli_path = data_dir / "test" / "test_nli.csv"
train_sts_path = data_dir / "train" / "train_sts.csv"
test_sts_path = data_dir / "test" / "test_sts.csv"

# Save path of the model
output_dir = f"../output/matryoshka_nli_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

def load_csv_as_dataset(csv_path, task_type="nli"):
    """Load CSV file and convert to Dataset."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if task_type == "nli":
                # Convert to triplet format: (anchor, positive, negative)
                data.append({
                    "anchor": row["anchor"],
                    "positive": row["positive"],
                    "negative": row["negative"]
                })
            elif task_type == "sts":
                # Convert to STS format: (sentence1, sentence2, score)
                data.append({
                    "sentence1": row["sentence1"],
                    "sentence2": row["sentence2"],
                    "score": float(row["score"])
                })
    return Dataset.from_list(data)

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

# 2. Load the Arabic NLI dataset from local CSV files
logging.info(f"Loading training data from {train_nli_path}")
train_dataset = load_csv_as_dataset(train_nli_path, task_type="nli")
logging.info(f"Loading evaluation data from {test_nli_path}")
eval_dataset = load_csv_as_dataset(test_nli_path, task_type="nli")
logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Eval dataset size: {len(eval_dataset)}")

# If you wish, you can limit the number of training samples
# train_dataset = train_dataset.select(range(5000))

# 3. Define our training loss
inner_train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
logging.info(f"Loading STS validation data from {train_sts_path}")
stsb_eval_dataset = load_csv_as_dataset(train_sts_path, task_type="sts")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=stsb_eval_dataset["sentence1"],
            sentences2=stsb_eval_dataset["sentence2"],
            scores=stsb_eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-dev-{dim}",
            truncate_dim=dim,
        )
    )
dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="matryoshka-nli",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the STS Benchmark test dataset
logging.info(f"Loading STS test data from {test_sts_path}")
test_dataset = load_csv_as_dataset(test_sts_path, task_type="sts")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-test-{dim}",
            truncate_dim=dim,
        )
    )
test_evaluator = SequentialEvaluator(evaluators)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)
logging.info(f"Model saved to {final_output_dir}")
