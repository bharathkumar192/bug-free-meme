#!/bin/bash
# Complete Setup and Benchmarking Script for Telugu Language Model Evaluation
# This script handles everything from installation to running benchmarks

set -e  # Exit on any error

# Step 1: Clone LM Evaluation Harness
echo "Step 1: Cloning LM Evaluation Harness repository..."
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness

# Step 2: Install the package with dependencies
echo "Step 2: Installing dependencies..."
pip install -e ".[multilingual]"

# Step 3: Create directory for custom tasks
echo "Step 3: Setting up custom task directory..."
mkdir -p lm_eval/tasks/custom
touch lm_eval/tasks/custom/__init__.py

# Step 4: Create custom task definition file
echo "Step 4: Creating custom task definitions..."
cat > lm_eval/tasks/custom/telugu_tasks.py << 'EOL'
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.registry import register_task

# Register IndicSentiment for Telugu
@register_task("indic_sentiment_te")
class IndicSentimentTelugu(ConfigurableTask):
    def get_config(self):
        return {
            "description": "Telugu sentiment analysis from IndicSentiment dataset",
            "group": "telugu_benchmarks",
            "dataset_path": "ai4bharat/IndicSentiment",
            "dataset_name": "te",
            "output_type": "multiple_choice",
            "doc_to_text": lambda x: f"Classify the sentiment of the following Telugu review as Positive, Negative, or Neutral:\n\n{x['INDIC REVIEW']}",
            "doc_to_target": lambda x: ["Positive", "Negative", "Neutral"].index(x["LABEL"]),
            "doc_to_choice": lambda x: ["Positive", "Negative", "Neutral"],
            "metrics": [
                "accuracy",           # Percentage of correct predictions
                "f1",                 # F1 score (harmonic mean of precision and recall)
                "f1_macro",           # Macro-averaged F1 across all classes
                "f1_micro",           # Micro-averaged F1 across all classes
                "precision",          # Precision score
                "recall",             # Recall score
                "mcc",                # Matthews Correlation Coefficient
                "confusion_matrix"    # Generates a confusion matrix for analysis
            ],
            "split": "test"  # Use the test split
        }

# Register MMLU for Telugu
@register_task("mmlu_te")
class MMLUTelugu(ConfigurableTask):
    def get_config(self):
        return {
            "description": "Telugu version of the MMLU benchmark",
            "group": "telugu_benchmarks", 
            "dataset_path": "sarvamai/mmlu-indic",
            "dataset_filter": lambda x: x["language"] == "te",  # Filter for Telugu only
            "output_type": "multiple_choice",
            "doc_to_text": lambda x: f"{x['question']}\n\nA. {x['choices'][0]}\nB. {x['choices'][1]}\nC. {x['choices'][2]}\nD. {x['choices'][3]}\n\nAnswer:",
            "doc_to_target": lambda x: x["answer"],  # The index of the correct answer
            "doc_to_choice": lambda x: ["A", "B", "C", "D"],
            "metrics": [
                "accuracy",           # Overall accuracy
                "group_accuracy",     # Accuracy broken down by subject groups
                "per_group_metrics",  # Detailed metrics for each subject category
                "mc_correct_ratio",   # Ratio of correct choices to total choices
                "calibration",        # How well calibrated the model predictions are
                "perplexity"          # Lower is better - measures how "surprised" the model is
            ],
            "split": "test",         # Use the test split
            "num_samples": 2000      # Limit to 2000 samples from test
        }
EOL

# Update __init__.py to import the tasks
echo "from . import telugu_tasks" > lm_eval/tasks/custom/__init__.py

# Step 5: Create benchmarking script
echo "Step 5: Creating benchmarking script..."
cat > run_telugu_benchmarks.sh << 'EOL'
#!/bin/bash
# Telugu Benchmarking Script for A100 GPUs

# Configuration
MODEL_PATH="bharathkumar1922001/Gemma3-12b-Indic"
OUTPUT_DIR="telugu_benchmark_results"
BATCH_SIZE=32  # Manually set for maximum throughput on A100 80GB

# Create output directory
mkdir -p $OUTPUT_DIR

# Run IndicSentiment benchmark
echo "Running IndicSentiment benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks indic_sentiment_te \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_DIR/indic_sentiment_results.json \
    --log_samples

# Run MMLU benchmark
echo "Running MMLU benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks mmlu_te \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_DIR/mmlu_results.json \
    --log_samples

echo "All benchmarks completed. Results saved to $OUTPUT_DIR"
EOL

# Make the script executable
chmod +x run_telugu_benchmarks.sh

# Step 6: Run benchmarks
echo "Step 6: Ready to run benchmarks. Execute with:"
echo "./run_telugu_benchmarks.sh"
echo ""
echo "After benchmarking, find your results in the telugu_benchmark_results directory."
echo "You can copy them back to your local machine using:"
echo "scp -r user@your-server:path/to/lm-evaluation-harness/telugu_benchmark_results ."