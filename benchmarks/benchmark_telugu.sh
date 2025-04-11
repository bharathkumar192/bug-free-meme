#!/bin/bash
# Complete Setup and Benchmarking Script for Telugu Language Model Evaluation
# This script handles everything from installation to running benchmarks

set -e  # Exit on any error

# Define the lm-evaluation-harness directory
LM_EVAL_DIR="lm-evaluation-harness"

# Step 1: Clone LM Evaluation Harness
echo "Step 1: Cloning LM Evaluation Harness repository..."
if [ ! -d "$LM_EVAL_DIR" ]; then
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
fi
cd "$LM_EVAL_DIR"

# Step 2: Create directory for custom tasks
echo "Step 2: Setting up custom task directory..."
mkdir -p lm_eval/tasks/custom
touch lm_eval/tasks/custom/__init__.py
echo "# Custom tasks for Telugu benchmark" > lm_eval/tasks/custom/__init__.py

# Step 3: Create YAML task definitions
echo "Step 3: Creating custom task YAML definitions..."

# Create IndicSentiment Telugu YAML
cat > lm_eval/tasks/custom/indic_sentiment_te.yaml << 'EOL'
# IndicSentiment Telugu Evaluation
task: indic_sentiment_te
group: telugu_benchmarks
description: Telugu sentiment analysis from IndicSentiment dataset
dataset_path: ai4bharat/IndicSentiment
dataset_name: te
output_type: multiple_choice
doc_to_text: |
  Classify the sentiment of the following Telugu review as Positive, Negative, or Neutral:

  {INDIC REVIEW}
doc_to_target: "{LABEL}"
doc_to_choice: ["Positive", "Negative", "Neutral"]
metric_list:
  - accuracy
  - f1_macro
  - f1_micro
  - precision
  - recall
  - mcc
split: test
EOL

# Create MMLU Telugu YAML
cat > lm_eval/tasks/custom/mmlu_te.yaml << 'EOL'
# MMLU Telugu Evaluation
task: mmlu_te
group: telugu_benchmarks
description: Telugu version of the MMLU benchmark
dataset_path: sarvamai/mmlu-indic
dataset_filter: "language == 'te'"
output_type: multiple_choice
doc_to_text: |
  {question}

  A. {choices[0]}
  B. {choices[1]}
  C. {choices[2]}
  D. {choices[3]}

  Answer:
doc_to_target: "{answer}"
doc_to_choice: ["A", "B", "C", "D"]
metric_list:
  - accuracy
  - group_accuracy
  - mc_correct_ratio
split: test
EOL

# Step 4: Install the package with dependencies
echo "Step 4: Installing/Re-installing dependencies..."
pip install -e ".[multilingual]" --no-cache-dir

# Step 5: List available tasks to verify registration
echo "Step 5: Verifying task registration..."
python -m lm_eval --tasks list --verbosity DEBUG

# Step 6: Run benchmarks
echo "Step 6: Running benchmarks with accelerate..."
OUTPUT_DIR="../telugu_benchmark_results"
mkdir -p "$OUTPUT_DIR"

MODEL_PATH="bharathkumar1922001/Gemma3-12b-Indic"
BATCH_SIZE=32
CUSTOM_TASKS_PATH=$(realpath lm_eval/tasks/custom)

# Run IndicSentiment benchmark with accelerate
echo "Running IndicSentiment benchmark..."
accelerate launch --num_processes=2 -m lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH",trust_remote_code=True \
    --tasks indic_sentiment_te \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR/indic_sentiment_results.json" \
    --log_samples \
    --include_path "$CUSTOM_TASKS_PATH" \
    --verbosity DEBUG

# Run MMLU benchmark with accelerate
echo "Running MMLU benchmark..."
accelerate launch --num_processes=2 -m lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH",trust_remote_code=True \
    --tasks mmlu_te \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR/mmlu_results.json" \
    --log_samples \
    --include_path "$CUSTOM_TASKS_PATH" \
    --verbosity DEBUG

echo "All benchmarks completed. Results saved to $OUTPUT_DIR"

# Step 7: Instructions for finding results (adjusted path)
echo ""
echo "After benchmarking, find your results in the ../telugu_benchmark_results directory."
echo "You can copy them back to your local machine using:"
echo "scp -r user@your-server:path/to/bug-free-meme/telugu_benchmark_results ."