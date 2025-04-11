#!/bin/bash
# Complete Setup and Benchmarking Script for Telugu Language Model Evaluation
# This script handles everything from installation to running benchmarks using YAML task definitions

set -e  # Exit on any error

# Login to Hugging Face (Optional - uncomment if needed)
echo "Logging in to Hugging Face..."
HUGGING_FACE_TOKEN="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"
echo "$HUGGING_FACE_TOKEN" | huggingface-cli login --token "$HUGGING_FACE_TOKEN"

# Define the lm-evaluation-harness directory
LM_EVAL_DIR="lm-evaluation-harness"

Step 1: Clone LM Evaluation Harness (or update if it exists)
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
# Ensure the init file is empty, needed for package discovery but no imports required for YAML
> lm_eval/tasks/custom/__init__.py

# Step 3: Create custom YAML task definitions
echo "Step 3: Creating custom YAML task definitions..."

# Create CORRECTED indic_sentiment_te.yaml
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
# CORRECTED: Use 'INDIC REVIEW' directly, not ['INDIC REVIEW']
doc_to_text: "Classify the sentiment of the following Telugu review as Positive, Negative, or Neutral:\n\n{{ INDIC REVIEW }}"
# CORRECTED: Use 'LABEL' directly, not ['LABEL']
doc_to_target: "{% if LABEL == 'Positive' %}0{% elif LABEL == 'Negative' %}1{% elif LABEL == 'Neutral' %}2{% else %}-1{% endif %}"
doc_to_choice: ["Positive", "Negative", "Neutral"]
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
EOL

# Create CORRECTED mmlu_te.yaml
cat > lm_eval/tasks/custom/mmlu_te.yaml << 'EOL'
# Task definition for Telugu MMLU subset
task: mmlu_te
dataset_path: sarvamai/mmlu-indic
dataset_kwargs:
  trust_remote_code: True
output_type: multiple_choice
test_split: test
num_fewshot: 0

# Optional Filter: Uncomment and adjust if needed (ensure column name 'language' exists)
# filter_list:
#  - name: "filter_language_telugu"
#    filter:
#      - function: "regex"
#        inputs: "language" # Verify this is the correct column name in the dataset
#        regex_pattern: "^te$"

# Modified doc_to_text to handle varying numbers of choices gracefully
doc_to_text: "{{ question }}\n{% if choices is defined and choices|length > 0 %}\nA. {{ choices[0] }}{% endif %}{% if choices is defined and choices|length > 1 %}\nB. {{ choices[1] }}{% endif %}{% if choices is defined and choices|length > 2 %}\nC. {{ choices[2] }}{% endif %}{% if choices is defined and choices|length > 3 %}\nD. {{ choices[3] }}{% endif %}\n\nAnswer:"
# CORRECTED: Map answer letter (A,B,C,D) to index (0,1,2,3)
doc_to_target: "{% if answer == 'A' %}0{% elif answer == 'B' %}1{% elif answer == 'C' %}2{% elif answer == 'D' %}3{% else %}-1{% endif %}"
doc_to_choice: ["A", "B", "C", "D"]
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
EOL

echo "Custom YAML task files created."

# Step 4: Install the package with dependencies
echo "Step 4: Installing/Re-installing dependencies..."
# Ensure pip is up-to-date (Recommended)
pip install --upgrade pip
cd lm-evaluation-harness
# Install lm-eval-harness editable, including multilingual extras (CRITICAL)
pip install -e ".[multilingual]" --no-cache-dir

# Step 5: List available tasks to verify registration (Recommended)
echo "Step 5: Verifying task registration..."
# Check if the custom tasks appear in the list
# python -m lm_eval --tasks list --verbosity INFO | grep -E 'indic_sentiment_te|mmlu_te' || echo "Custom tasks not found in list, check YAML files, paths, and installation."

# Step 6: Run benchmarks
echo "Step 6: Running benchmarks with accelerate..."
OUTPUT_DIR="../telugu_benchmark_results" # Results saved outside the git repo
mkdir -p "$OUTPUT_DIR"

# MODEL_PATH="krutrim-ai-labs/Krutrim-2-instruct"
MODEL_PATH="bharathkumar1922001/Gemma3-12b-Indic"
BATCH_SIZE=8 # Use auto batch size detection. Adjust (e.g., 4, 8, 16) if OOM occurs
CUSTOM_TASKS_PATH=$(realpath lm_eval/tasks/custom) # Get absolute path

# Add the main library path to PYTHONPATH in case of import issues (optional, usually not needed if installed with -e)
export PYTHONPATH="$PYTHONPATH:$(realpath .)"

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
    --verbosity DEBUG # Use DEBUG for detailed logs, change to INFO for less output

# Run MMLU benchmark with accelerate launch
# echo "Running MMLU benchmark..."
# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained="$MODEL_PATH",trust_remote_code=True \
#     --tasks mmlu_te \
#     --batch_size "$BATCH_SIZE" \
#     --output_path "$OUTPUT_DIR/mmlu_results.json" \
#     --log_samples \
#     --include_path "$CUSTOM_TASKS_PATH" \
#     --verbosity DEBUG # Use DEBUG for detailed logs, change to INFO for less output

echo "All benchmarks completed. Results saved to $OUTPUT_DIR"

# Step 7: Instructions for finding results (adjusted path)
echo ""
echo "After benchmarking, find your results in the ../telugu_benchmark_results directory relative to the lm-evaluation-harness folder."
echo "From the parent directory where you ran the script (assuming it's outside lm-evaluation-harness), the path is 'telugu_benchmark_results'."
echo "You can copy them back to your local machine using (adjust path as needed):"
# Adjust the source path if you run the script from a different location relative to the results dir
echo "scp -r user@your-server:$(realpath "$OUTPUT_DIR") ."