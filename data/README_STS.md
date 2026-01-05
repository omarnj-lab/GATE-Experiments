# Arabic Semantic Textual Similarity (STS) Dataset

## Dataset Summary

The Arabic version of the Semantic Textual Similarity Benchmark (STS) dataset. This dataset is used for evaluating semantic textual similarity models on Arabic text.

## Dataset Structure

**Columns:** `sentence1`, `sentence2`, `score`

**Column types:** str, str, float

**Format:** CSV files with header row

**Score range:** 0.0 to 1.0, where:
- 0.0 indicates completely unrelated sentences
- 1.0 indicates semantically equivalent sentences

### Examples

```json
{
  "sentence1": "طائرة ستقلع",
  "sentence2": "طائرة جوية ستقلع",
  "score": 1.0
}
```

```json
{
  "sentence1": "رجل يعزف على ناي كبير",
  "sentence2": "رجل يعزف على الناي.",
  "score": 0.76
}
```

## Files

- `train_sts.csv`: Training set containing sentence pairs with similarity scores
- `test_sts.csv`: Test set containing sentence pairs with similarity scores

## Usage

The dataset is provided in CSV format with the following structure:
- First row contains column headers: `sentence1,sentence2,score`
- Each subsequent row contains a pair of Arabic sentences and their similarity score

## Evaluation

This dataset is commonly used for evaluating Arabic semantic textual similarity models. The evaluation metric is typically Spearman's rank correlation coefficient between predicted similarities and ground truth scores.

