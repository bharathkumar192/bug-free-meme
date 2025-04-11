#!/bin/bash
# Complete Setup and Benchmarking Script for Telugu Language Model Evaluation
# This script handles everything from installation to running benchmarks

set -e  # Exit on any error

# Login to Hugging Face
echo "Logging in to Hugging Face..."
# echo "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" | huggingface-cli login

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

# Step 3: Create custom Python task definitions instead of YAML
echo "Step 3: Creating custom task Python definitions..."

# Create a Python file with the task definitions
cat > lm_eval/tasks/custom/telugu_tasks.py << 'EOL'
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.registry import register_task
import datasets

@register_task("indic_sentiment_te")
class IndicSentimentTelugu(ConfigurableTask):
    def get_config(self):
        return {
            "description": "Telugu sentiment analysis from IndicSentiment dataset",
            "dataset_path": "ai4bharat/IndicSentiment",
            "dataset_name": "te",
            "output_type": "multiple_choice",
            "doc_to_text": lambda x: f"Classify the sentiment of the following Telugu review as Positive, Negative, or Neutral:\n\n{x['INDIC REVIEW']}",
            "doc_to_target": lambda x: ["Positive", "Negative", "Neutral"].index(x["LABEL"]),
            "doc_to_choice": lambda x: ["Positive", "Negative", "Neutral"],
            "metrics": ["accuracy", "f1_macro", "f1_micro", "precision", "recall", "mcc"]
        }

@register_task("mmlu_te")
class MMLUTelugu(ConfigurableTask):
    def get_config(self):
        return {
            "description": "Telugu version of the MMLU benchmark",
            "dataset_path": "sarvamai/mmlu-indic",
            "dataset_filter": lambda x: x["language"] == "te",
            "output_type": "multiple_choice",
            "doc_to_text": lambda x: f"{x['question']}\n\nA. {x['choices'][0]}\nB. {x['choices'][1]}\nC. {x['choices'][2]}\nD. {x['choices'][3]}\n\nAnswer:",
            "doc_to_target": lambda x: ["A", "B", "C", "D"].index(x["answer"]),
            "doc_to_choice": lambda x: ["A", "B", "C", "D"],
            "metrics": ["accuracy"]
        }
EOL

# Update the __init__.py to import the tasks
cat > lm_eval/tasks/custom/__init__.py << 'EOL'
# Import the custom Telugu tasks
try:
    from . import telugu_tasks
    print("Successfully imported Telugu tasks!")
except ImportError as e:
    print(f"Error importing Telugu tasks: {e}")
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

# Add the custom path to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(realpath .)"

# Run IndicSentiment benchmark with standard accelerate launch
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

# Run MMLU benchmark with standard accelerate launch
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
echo "After benchmarking, find your results in the ../telugu_benchmark_results directory."
echo "You can copy them back to your local machine using:"
echo "scp -r user@your-server:path/to/bug-free-meme/telugu_benchmark_results ."