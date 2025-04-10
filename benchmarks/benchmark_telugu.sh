#!/bin/bash
# Complete Setup and Benchmarking Script for Telugu Language Model Evaluation
# This script handles everything from installation to running benchmarks

set -e  # Exit on any error

# Define the lm-evaluation-harness directory
LM_EVAL_DIR="lm-evaluation-harness"
CUSTOM_TASKS_PATH=$(realpath lm_eval/tasks/custom)

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
echo "from . import telugu_tasks" > lm_eval/tasks/custom/__init__.py

# Step 3: Create custom task definition file (using the provided content)
echo "Step 3: Creating custom task definitions in lm_eval/tasks/custom/telugu_tasks.py..."
cat > lm_eval/tasks/custom/telugu_tasks.py << 'EOL'
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.registry import register_task
import datasets

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
                "accuracy",
                "f1",
                "f1_macro",
                "f1_micro",
                "precision",
                "recall",
                "mcc",
                "confusion_matrix"
            ],
            "split": "test"
        }

@register_task("mmlu_te")
class MMLUTelugu(ConfigurableTask):
    def get_config(self):
        return {
            "description": "Telugu version of the MMLU benchmark",
            "group": "telugu_benchmarks",
            "dataset_path": "sarvamai/mmlu-indic",
            "dataset_filter": lambda x: x["language"] == "te",
            "dataset_preprocessing": lambda dataset: self._sample_stratified(dataset, 1000),
            "output_type": "multiple_choice",
            "doc_to_text": lambda x: f"{x['question']}\n\nA. {x['choices'][0]}\nB. {x['choices'][1]}\nC. {x['choices'][2]}\nD. {x['choices'][3]}\n\nAnswer:",
            "doc_to_target": lambda x: x["answer"],
            "doc_to_choice": ["A", "B", "C", "D"],
            "metrics": ["accuracy", "group_accuracy", "mc_correct_ratio", "calibration", "perplexity"],
            "split": "test"
        }

    def _sample_stratified(self, dataset, n_samples):
        """Sample stratified by subject if available, otherwise random sample."""
        if "subject" in dataset.features:
            subjects = dataset.unique("subject")
            samples_per_subject = max(1, n_samples // len(subjects))

            sampled_datasets = []
            for subject in subjects:
                subject_dataset = dataset.filter(lambda x: x["subject"] == subject)
                if len(subject_dataset) > samples_per_subject:
                    sampled_datasets.append(subject_dataset.select(range(samples_per_subject)))
                else:
                    sampled_datasets.append(subject_dataset)

            return datasets.concatenate_datasets(sampled_datasets)
        else:
            if len(dataset) > n_samples:
                return dataset.select(range(n_samples))
            return dataset
EOL

# Step 4: Install the package with dependencies (AFTER creating custom tasks)
echo "Step 4: Installing/Re-installing dependencies (to include custom tasks)..."
python3 -m pip install -e ".[multilingual]" --no-cache-dir

# Step 5: Run benchmarks
echo "Step 5: Running benchmarks with accelerate (2 GPUs)..."
OUTPUT_DIR="../telugu_benchmark_results"
mkdir -p "$OUTPUT_DIR"
echo "lm-harness tasks declared are : "
python -m lm_eval --tasks list --verbosity DEBUG

MODEL_PATH="bharathkumar1922001/Gemma3-12b-Indic"
BATCH_SIZE=32
export PYTHONPATH="$PYTHONPATH:$CUSTOM_TASKS_PATH"

# Run IndicSentiment benchmark with accelerate
echo "Running IndicSentiment benchmark..."
accelerate launch --num_processes=2 -m lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH",trust_remote_code=True \
    --tasks indic_sentiment_te \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR/indic_sentiment_results_indic_sentiment.json" \
    --log_samples \
    --include_path "$CUSTOM_TASKS_PATH"

# Run MMLU benchmark with accelerate
echo "Running MMLU benchmark..."
accelerate launch --num_processes=2 -m lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH",trust_remote_code=True \
    --tasks mmlu_te \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR/mmlu_results_mmlu.json" \
    --log_samples \
    --include_path "$CUSTOM_TASKS_PATH"

echo "All benchmarks completed. Results saved to $OUTPUT_DIR"

# Step 6: Instructions for finding results (adjusted path)
echo ""
echo "After benchmarking, find your results in the ../telugu_benchmark_results directory."
echo "You can copy them back to your local machine using:"
echo "scp -r user@your-server:path/to/bug-free-meme/telugu_benchmark_results ."