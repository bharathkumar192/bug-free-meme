#!/bin/bash
# Complete Setup and Benchmarking Script for Telugu Language Model Evaluation
# This script ONLY runs the IndicSentiment benchmark using YAML task definitions
# and custom Python functions for prompt formatting.

set -e  # Exit on any error

# Login to Hugging Face (Optional - uncomment if needed)
echo "Logging in to Hugging Face..."
HUGGING_FACE_TOKEN="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" # Replace if necessary
echo "$HUGGING_FACE_TOKEN" | huggingface-cli login --token "$HUGGING_FACE_TOKEN"

# Define the lm-evaluation-harness directory
LM_EVAL_DIR="lm-evaluation-harness"

# Step 1: Clone LM Evaluation Harness (or update if it exists)
echo "Step 1: Managing LM Evaluation Harness repository..."
if [ ! -d "$LM_EVAL_DIR" ]; then
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd "$LM_EVAL_DIR"
else
    cd "$LM_EVAL_DIR"
    echo "Repository exists, checking for updates..."
    # Optional: Consider 'git pull' if you always want the latest version
    # git pull
fi

# Step 2: Create directory for custom tasks and ensure __init__.py is empty
echo "Step 2: Setting up custom task directory..."
mkdir -p lm_eval/tasks/custom
# Ensure the init file is empty
> lm_eval/tasks/custom/__init__.py

# Step 3a: Create custom Python utility functions (Needed for IndicSentiment)
echo "Step 3a: Creating custom utility functions (custom_utils.py)..."
cat > lm_eval/tasks/custom/custom_utils.py << 'EOL'
# Custom utility functions for lm-eval-harness tasks

def indic_sentiment_doc_to_text(doc):
    """Formats the prompt for IndicSentiment, handling the 'INDIC REVIEW' key."""
    # Access the key directly using dictionary lookup
    review_text = doc.get('INDIC REVIEW', '') # Use .get for safety
    return f"Classify the sentiment of the following Telugu review as Positive, Negative, or Neutral:\n\n{review_text}"

def indic_sentiment_doc_to_target(doc):
    """Maps the string label from IndicSentiment to an integer index."""
    label = doc.get('LABEL', '') # Use .get for safety
    if label == 'Positive':
        return 0
    elif label == 'Negative':
        return 1
    elif label == 'Neutral':
        return 2
    else:
        # Return an invalid index if label is unexpected
        # The harness's process_results should handle this
        return -1
EOL

# Step 3b: Create custom YAML task definition for IndicSentiment
echo "Step 3b: Creating custom YAML task definition for IndicSentiment..."

# Create indic_sentiment_te.yaml using !function syntax
cat > lm_eval/tasks/custom/indic_sentiment_te.yaml << 'EOL'
# Task definition for Telugu Sentiment Analysis
task: indic_sentiment_te
dataset_path: ai4bharat/IndicSentiment
dataset_name: translation-te
dataset_kwargs:
  trust_remote_code: True
output_type: multiple_choice
test_split: test
num_fewshot: 0
# Use custom Python functions defined in custom_utils.py
# Assumes the harness can find functions relative to the --include_path
doc_to_text: !function custom_utils.indic_sentiment_doc_to_text
doc_to_target: !function custom_utils.indic_sentiment_doc_to_target
doc_to_choice: ["Positive", "Negative", "Neutral"]
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
EOL

# --- MMLU YAML definition is removed/commented out ---
# echo "Skipping MMLU YAML creation."

echo "Custom YAML and utility files created for IndicSentiment."

# Step 4: Install the package with dependencies
echo "Step 4: Installing/Re-installing dependencies..."
pip install --upgrade pip --quiet
pip install -e ".[multilingual]" --no-cache-dir --quiet

# Step 5: List available tasks to verify registration (Optional but Recommended)
echo "Step 5: Verifying task registration..."
python -m lm_eval --tasks list --verbosity INFO | grep -E 'indic_sentiment_te' || echo "IndicSentiment task not found in list, check YAML/utils files, paths, and installation."

# Step 6: Run benchmarks
echo "Step 6: Running benchmarks with accelerate..."
OUTPUT_DIR="../telugu_benchmark_results" # Results saved outside the git repo
mkdir -p "$OUTPUT_DIR"

MODEL_PATH="bharathkumar1922001/Gemma3-12b-Indic"
BATCH_SIZE=auto # Starting with 8, adjust if needed.
CUSTOM_TASKS_PATH=$(realpath lm_eval/tasks/custom) # Get absolute path

# Add the main library path AND the custom path to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(realpath .):$CUSTOM_TASKS_PATH"

# Run IndicSentiment benchmark with accelerate launch
echo "Running IndicSentiment benchmark..."
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH",trust_remote_code=True \
    --tasks indic_sentiment_te \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR/indic_sentiment_results.json" \
    --log_samples \
    --include_path "$CUSTOM_TASKS_PATH" \
    --verbosity DEBUG


echo "IndicSentiment benchmark completed. Results saved to $OUTPUT_DIR"

# Step 7: Instructions for finding results
echo ""
echo "After benchmarking, find your results in the $OUTPUT_DIR directory."
echo "You can copy them back using (adjust path if needed):"
echo "scp -r user@your-server:$(realpath "$OUTPUT_DIR") ."