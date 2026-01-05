#!/usr/bin/env python3
"""
Script to create JSON samples from CSV datasets for demonstration purposes.
"""

import csv
import json
import os
from pathlib import Path

def csv_to_json_samples(csv_path, json_path, num_samples=50):
    """Convert CSV to JSON samples."""
    samples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            samples.append(row)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(samples)} samples in {json_path}")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Create samples directory if it doesn't exist
    samples_dir = script_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Create NLI samples
    print("Creating NLI samples...")
    csv_to_json_samples(
        script_dir / "train" / "train_nli.csv", 
        samples_dir / "nli_train_samples.json", 
        num_samples=100
    )
    csv_to_json_samples(
        script_dir / "test" / "test_nli.csv", 
        samples_dir / "nli_test_samples.json", 
        num_samples=50
    )
    
    # Create STS samples
    print("Creating STS samples...")
    csv_to_json_samples(
        script_dir / "train" / "train_sts.csv", 
        samples_dir / "sts_train_samples.json", 
        num_samples=100
    )
    csv_to_json_samples(
        script_dir / "test" / "test_sts.csv", 
        samples_dir / "sts_test_samples.json", 
        num_samples=50
    )
    
    print("Sample creation complete!")

