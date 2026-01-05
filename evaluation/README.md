# Evaluation

This directory contains evaluation scripts for Arabic text embedding models.

## evaluate_mteb_simple.py

A simplified version of the MTEB evaluation script that closely matches the example code pattern.

### Usage

```bash
python evaluate_mteb_simple.py
```

Edit the `model_names` list in the script to specify which models to evaluate.

## evaluate_sts.py

Evaluates models on the Arabic Semantic Textual Similarity (STS) benchmark.

### Usage

```bash
python evaluate_sts.py <model_path> [test_data_path] [output_dim]
```

### Example

```bash
python evaluate_sts.py ../models/my_model ../data/test/test_sts.csv 768
```

### Output

Prints the Spearman correlation score on the STS test set.

## evaluate_mteb.py

Evaluates models on MTEB benchmarks (STS17 and STS22.v2) and computes average scores.

### Usage

```bash
python evaluate_mteb.py <model_path> [output_dir]
```

### Example

```bash
python evaluate_mteb.py ../models/my_model ../results
```

### Features

- Evaluates on STS17 and STS22.v2 benchmarks
- Supports Matryoshka dimensions (can evaluate at multiple embedding sizes)
- Computes average scores across tasks
- Saves results to JSON file

### Output

- Prints evaluation results for each task and dimension
- Computes and displays average scores
- Saves detailed results to JSON file in the output directory

### Configuration

You can modify the script to:
- Change tasks: Edit the `tasks` list (default: `["STS17", "STS22.v2"]`)
- Change dimensions: Edit the `matryoshka_dims` list (default: `[768]`)
- Change language: Edit the `task_langs` list (default: `["ar"]`)

