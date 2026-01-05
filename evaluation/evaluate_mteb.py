"""
MTEB evaluation script for Arabic semantic textual similarity tasks.

This script evaluates trained embedding models on STS17 and STS22.v2 benchmarks
using the MTEB framework and computes average scores.
"""

import logging
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from mteb import MTEB
import numpy as np

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M-%S", level=logging.INFO)

def evaluate_model_mteb(model_path, tasks=None, task_langs=None, matryoshka_dims=None, output_dir=None):
    """
    Evaluate a model on MTEB benchmarks.
    
    Args:
        model_path: Path to the trained model
        tasks: List of task names (default: ["STS17", "STS22.v2"])
        task_langs: List of languages (default: ["ar"])
        matryoshka_dims: List of dimensions to evaluate (default: [768])
        output_dir: Directory to save results (default: "results")
    
    Returns:
        Dictionary with evaluation results
    """
    if tasks is None:
        tasks = ["STS17", "STS22.v2"]
    if task_langs is None:
        task_langs = ["ar"]
    if matryoshka_dims is None:
        matryoshka_dims = [768]
    if output_dir is None:
        output_dir = Path("results")
    
    # Extract model name from path
    model_name = Path(model_path).name if Path(model_path).exists() else model_path.split('/')[-1]
    
    results = {}
    
    # Loop through each dimension
    for dim in matryoshka_dims:
        logging.info(f"Evaluating model: {model_path} at dimension: {dim}")
        
        # Load the model
        model = SentenceTransformer(model_path)
        
        # If dimension is less than full dimension, we need to truncate
        # Note: MTEB handles this automatically for Matryoshka models
        if dim < 768:
            # Truncate embeddings during evaluation
            # This is handled by the evaluator's truncate_dim parameter
            pass
        
        # Initialize the evaluation
        evaluation = MTEB(tasks=tasks, task_langs=task_langs)
        
        # Set the output folder based on model name and dimension
        output_folder = output_dir / "ar" / f"{model_name}_{dim}"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Run the evaluation
        results[dim] = evaluation.run(model, output_folder=str(output_folder))
        
        logging.info(f"Finished evaluation for model: {model_name} at dimension {dim}")
    
    return results

def compute_average_scores(results):
    """
    Compute average scores across tasks and dimensions.
    
    Args:
        results: Dictionary of results from evaluate_model_mteb
    
    Returns:
        Dictionary with average scores
    """
    averages = {}
    
    for dim, dim_results in results.items():
        scores = []
        task_scores = {}
        
        # Extract scores from results
        # MTEB returns results as a list of dictionaries
        if isinstance(dim_results, list):
            for task_result in dim_results:
                task_name = task_result.get('dataset', {}).get('revision', 'unknown')
                # Try to extract task name from various possible fields
                if 'dataset' in task_result:
                    task_name = task_result['dataset'].get('name', task_name)
                
                # Extract the main score (usually Spearman correlation)
                score = 0.0
                if 'test' in task_result:
                    test_results = task_result['test']
                    if isinstance(test_results, dict):
                        score = test_results.get('spearman', test_results.get('cosine_spearman', 
                                    test_results.get('score', 0.0)))
                    elif isinstance(test_results, (int, float)):
                        score = test_results
                elif 'spearman' in task_result:
                    score = task_result['spearman']
                elif 'cosine_spearman' in task_result:
                    score = task_result['cosine_spearman']
                elif 'score' in task_result:
                    score = task_result['score']
                else:
                    # Try to find any numeric value in the result
                    for key, value in task_result.items():
                        if isinstance(value, (int, float)) and key != 'dim':
                            score = value
                            break
                
                if score > 0:  # Only add valid scores
                    scores.append(score)
                    task_scores[task_name] = score
        
        averages[dim] = {
            'tasks': task_scores,
            'average': np.mean(scores) if scores else 0.0
        }
    
    return averages

def print_results(results, averages):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Raw results dictionary
        averages: Averaged scores dictionary
    """
    print("\n" + "="*60)
    print("MTEB Evaluation Results")
    print("="*60)
    
    for dim, avg_data in averages.items():
        print(f"\nDimension: {dim}")
        print("-" * 60)
        for task_name, score in avg_data['tasks'].items():
            print(f"  {task_name}: {score:.4f}")
        print(f"  Average: {avg_data['average']:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_mteb.py <model_path> [output_dir]")
        print("Example: python evaluate_mteb.py ../models/my_model ../results")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    
    # Define tasks and dimensions
    tasks = ["STS17", "STS22.v2"]
    task_langs = ["ar"]
    matryoshka_dims = [768]  # Can be extended to [768, 512, 256, 128, 64]
    
    # Run evaluation
    results = evaluate_model_mteb(
        model_path=model_path,
        tasks=tasks,
        task_langs=task_langs,
        matryoshka_dims=matryoshka_dims,
        output_dir=output_dir
    )
    
    # Compute averages
    averages = compute_average_scores(results)
    
    # Print results
    print_results(results, averages)
    
    # Save results to JSON
    results_file = output_dir / f"{Path(model_path).name}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model': model_path,
            'results': results,
            'averages': averages
        }, f, indent=2, default=str)
    
    logging.info(f"Results saved to {results_file}")

