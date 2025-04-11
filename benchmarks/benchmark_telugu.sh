#!/bin/bash
# Complete Setup and Benchmarking Script for Telugu Language Model Evaluation
# This script handles everything from installation to running benchmarks using YAML task definitions

set -e  # Exit on any error

# Login to Hugging Face
echo "Logging in to Hugging Face..."
# Uncomment and add your token if login is required and not cached
# HUGGING_FACE_TOKEN="hf_YOUR_TOKEN_HERE" 
# echo "$HUGGING_FACE_TOKEN" | huggingface-cli login --token "$HUGGING_FACE_TOKEN"

# Define the lm-evaluation-harness directory
LM_EVAL_DIR="lm-evaluation-harness"

# Step 1: Clone LM Evaluation Harness (or update if it exists)
echo "Step 1: Managing LM Evaluation Harness repository..."
if [ ! -d "$LM_EVAL_DIR" ]; then
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd "$LM_EVAL_DIR"
else
    cd "$LM_EVAL_DIR"
    echo "Repository exists, pulling latest changes..."
    # Optional: git pull # Uncomment if you want to always update to the latest version
fi

# Step 2: Create directory for custom tasks and ensure __init__.py is empty
echo "Step 2: Setting up custom task directory..."
mkdir -p lm_eval/tasks/custom
# Ensure the init file is empty, needed for package discovery but no imports required for YAML
> lm_eval/tasks/custom/__init__.py 

# Step 3: Create custom YAML task definitions
echo "Step 3: Creating custom YAML task definitions..."

# Create indic_sentiment_te.yaml (Mapping target label to index)
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
doc_to_text: "Classify the sentiment of the following Telugu review as Positive, Negative, or Neutral:\n\n{{ ['INDIC REVIEW'] }}"
# Map the string label to its integer index based on doc_to_choice
doc_to_target: "{% if ['LABEL'] == 'Positive' %}0{% elif ['LABEL'] == 'Negative' %}1{% elif ['LABEL'] == 'Neutral' %}2{% else %}-1{% endif %}"
doc_to_choice: ["Positive", "Negative", "Neutral"]
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
EOL

# Create mmlu_te.yaml (MMLU likely already provides integer/letter targets, check if needed)
# Assuming 'answer' column in sarvamai/mmlu-indic already contains 'A', 'B', 'C', or 'D' which might
# be handled correctly by default processing or might need similar index mapping if it expects integers 0-3.
# Let's leave it as is for now unless mmlu_te fails similarly.
cat > lm_eval/tasks/custom/mmlu_te.yaml << 'EOL'
# Task definition for Telugu MMLU subset
task: mmlu_te
dataset_path: sarvamai/mmlu-indic
dataset_kwargs:
  trust_remote_code: True
output_type: multiple_choice
test_split: test
num_fewshot: 0

# Optional Filter: Uncomment and adjust if needed
# filter_list:
#  - name: "filter_language_telugu"
#    filter:
#      - function: "regex"
#        inputs: "language" # Replace with actual column name
#        regex_pattern: "^te$"

doc_to_text: "{{ question }}\n\nA. {{ choices[0] }}\nB. {{ choices[1] }}\nC. {{ choices[2] }}\nD. {{ choices[3] }}\n\nAnswer:"
doc_to_target: "{{ answer }}" # Check if 'answer' needs mapping to 0, 1, 2, 3 if process_results expects int
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
# Ensure pip is up-to-date
# pip install --upgrade pip
# Install lm-eval-harness editable, including multilingual extras
# pip install -e ".[multilingual]" --no-cache-dir

# Step 5: List available tasks to verify registration
echo "Step 5: Verifying task registration..."
# Check if the custom tasks appear in the list
# python -m lm_eval --tasks list --verbosity INFO | grep -E 'indic_sentiment_te|mmlu_te' || echo "Custom tasks not found in list, check YAML files and paths."

# Step 6: Run benchmarks
echo "Step 6: Running benchmarks with accelerate..."
OUTPUT_DIR="../telugu_benchmark_results" # Results saved outside the git repo
mkdir -p "$OUTPUT_DIR"

MODEL_PATH="bharathkumar1922001/Gemma3-12b-Indic"
BATCH_SIZE=auto # Use auto batch size detection if possible, adjust if OOM occurs
CUSTOM_TASKS_PATH=$(realpath lm_eval/tasks/custom) # Get absolute path

# Add the main library path to PYTHONPATH in case of import issues
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
    --verbosity DEBUG

# Run MMLU benchmark with accelerate launch
echo "Running MMLU benchmark..."
accelerate launch -m lm_eval \
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
echo "After benchmarking, find your results in the ../telugu_benchmark_results directory relative to the lm-evaluation-harness folder."
echo "From the parent directory ('bug-free-meme/benchmarks'), the path is 'telugu_benchmark_results'."
echo "You can copy them back to your local machine using (adjust path as needed):"
echo "scp -r user@your-server:$(pwd)/../telugu_benchmark_results ." # Assumes you are still in lm-evaluation-harness dir