# Arabic NLI Triplet Dataset

## Dataset Summary

The Arabic Version of SNLI and MultiNLI datasets (Triplet Subset). Originally used for Natural Language Inference (NLI), this dataset may be used for training/finetuning an embedding model for semantic textual similarity.

## Dataset Structure

### Triplet Subset

**Columns:** `anchor`, `positive`, `negative`

**Column types:** str, str, str

**Format:** CSV files with header row

### Examples

```json
{
  "anchor": "شخص على حصان يقفز فوق طائرة معطلة",
  "positive": "شخص في الهواء الطلق، على حصان.",
  "negative": "شخص في مطعم، يطلب عجة."
}
```

## Files

- `train_nli.csv`: Training set containing anchor-positive-negative triplets
- `test_nli.csv`: Test set containing anchor-positive-negative triplets

## Usage

The dataset is provided in CSV format with the following structure:
- First row contains column headers: `anchor,positive,negative`
- Each subsequent row contains a triplet of Arabic sentences

## Disclaimer

Please note that the translated sentences are generated using neural machine translation and may not always convey the intended meaning accurately.

