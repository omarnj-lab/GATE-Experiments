"""
Simple MTEB evaluation script for Arabic STS benchmarks.

This is a simplified version that closely matches the provided example code.
Evaluates models on STS17 and STS22.v2 and computes average scores.
"""

from sentence_transformers import SentenceTransformer
from mteb import MTEB

# Define the list of models to evaluate
model_names = [
    "path/to/your/model",  # Replace with your model path
]

# Define the list of tasks
tasks = ["STS17", "STS22.v2"]
task_langs = ["ar"]  # Arabic language

# Define the matryoshka dimensions
matryoshka_dims = [768]  # Can be extended to [768, 512, 256, 128, 64]

# Loop through each model and evaluate at different dimensions
for model_name in model_names:
    print(f"Evaluating model: {model_name}")

    # Loop through each dimension
    for dim in matryoshka_dims:
        print(f"Evaluating at dimension: {dim}")

        # Load the model
        model = SentenceTransformer(model_name)

        # Initialize the evaluation
        evaluation = MTEB(tasks=tasks, task_langs=task_langs)

        # Set the output folder based on model name and dimension
        model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
        output_folder = f"results/ar/{model_short_name}_{dim}"

        # Run the evaluation and save the results in a specific folder
        results = evaluation.run(model, output_folder=output_folder)

        # Extract and compute average scores
        scores = []
        for task_result in results:
            # Extract score from result
            score = 0.0
            if isinstance(task_result, dict):
                if 'test' in task_result:
                    test_data = task_result['test']
                    if isinstance(test_data, dict):
                        score = test_data.get('spearman', test_data.get('cosine_spearman', 0))
                    elif isinstance(test_data, (int, float)):
                        score = test_data
                elif 'spearman' in task_result:
                    score = task_result['spearman']
                elif 'score' in task_result:
                    score = task_result['score']
            
            if score > 0:
                scores.append(score)
                task_name = task_result.get('dataset', {}).get('name', 'unknown')
                print(f"  {task_name}: {score:.4f}")

        # Compute and print average
        if scores:
            average = sum(scores) / len(scores)
            print(f"  Average: {average:.4f}")

        print(f"Finished evaluation for model: {model_name} at dimension {dim}\n")

