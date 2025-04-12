#!/bin/bash
# Multi-Model Comparison Script for Telugu Benchmarking
# This script runs the same benchmarks on multiple models for comparison

# Configuration
OUTPUT_DIR="comparison_results"
DEVICE="cuda:0"
TASKS="indicqa_te,indicsentiment_te,flores_te_en"

# Create output directory
mkdir -p $OUTPUT_DIR

# Model list - Add your models here
declare -a MODELS=(
    "bharathkumar1922001/gemma-3-12b-pt-telugu-sft"  # Your Telugu fine-tuned model
    "google/gemma-3-12b"                             # Base model for comparison
    "ai4bharat/indic-gemma-7b"                       # Existing Indic model
)

# Run benchmarks for each model
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename $MODEL)
    echo "Evaluating model: $MODEL_NAME"
    
    # Use vLLM for faster inference when possible
    if [[ "$MODEL" == *"gemma"* ]]; then
        # Regular Hugging Face evaluation for models that may not be compatible with vLLM
        lm_eval --model hf \
            --model_args pretrained=$MODEL,trust_remote_code=True \
            --tasks $TASKS \
            --device $DEVICE \
            --batch_size auto \
            --output_path $OUTPUT_DIR/${MODEL_NAME}_results.json \
            --log_samples
    else
        # Use vLLM for compatible models (faster inference)
        lm_eval --model vllm \
            --model_args pretrained=$MODEL,trust_remote_code=True,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
            --tasks $TASKS \
            --batch_size auto \
            --output_path $OUTPUT_DIR/${MODEL_NAME}_results.json \
            --log_samples
    fi
done

# Generate comparison report
python -c "
import json
import os
import pandas as pd

# Load results
results = {}
for filename in os.listdir('$OUTPUT_DIR'):
    if filename.endswith('_results.json'):
        model_name = filename.replace('_results.json', '')
        with open(os.path.join('$OUTPUT_DIR', filename), 'r') as f:
            results[model_name] = json.load(f)

# Extract metrics for each task
comparison = {}
for model, result in results.items():
    for task, metrics in result['results'].items():
        if task not in comparison:
            comparison[task] = {}
        comparison[task][model] = metrics

# Create comparison table
for task, models in comparison.items():
    print(f'\n=== {task} ===')
    task_df = pd.DataFrame(models).T
    print(task_df)
    # Save to CSV
    task_df.to_csv(os.path.join('$OUTPUT_DIR', f'{task}_comparison.csv'))

print(f'\nComparison tables saved to {os.path.abspath('$OUTPUT_DIR')}')
"

echo "Benchmark comparison completed"