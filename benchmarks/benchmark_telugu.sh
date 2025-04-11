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

# Step 3: Create Python task definitions that don't use split parameter
echo "Step 3: Creating custom Python task definitions..."

# First, check the TaskConfig class to see what parameters it accepts
cat > check_taskconfig.py << 'EOL'
from lm_eval.api.task import TaskConfig
import inspect

print("TaskConfig parameters:")
print(inspect.signature(TaskConfig.__init__))
EOL

python check_taskconfig.py

# Create a Python file with the minimal task definitions
cat > lm_eval/tasks/custom/telugu_tasks.py << 'EOL'
from lm_eval.api.task import Task
from lm_eval.api.registry import register_task
import datasets

@register_task("indic_sentiment_te")
class IndicSentimentTelugu(Task):
    VERSION = 1
    DATASET_PATH = "ai4bharat/IndicSentiment"
    DATASET_NAME = "te"
    
    def has_training_docs(self):
        return True
    
    def has_validation_docs(self):
        return False
    
    def has_test_docs(self):
        return True
    
    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs
    
    def test_docs(self):
        if self._test_docs is None:
            self._test_docs = list(self.dataset["test"])
        return self._test_docs
    
    def validation_docs(self):
        return []
    
    def doc_to_text(self, doc):
        return f"Classify the sentiment of the following Telugu review as Positive, Negative, or Neutral:\n\n{doc['INDIC REVIEW']}"
    
    def doc_to_target(self, doc):
        return doc["LABEL"]
    
    def process_results(self, docs, results):
        preds = []
        labels = []
        for pred, doc in zip(results, docs):
            pred_label = ["Positive", "Negative", "Neutral"][pred.argmax()]
            preds.append(pred_label)
            labels.append(doc["LABEL"])
        
        # Calculate accuracy
        correct = sum(p == l for p, l in zip(preds, labels))
        return {"accuracy": correct / len(docs)}
    
    def construct_requests(self, doc, ctx):
        return [
            {"doc": doc, "choices": ["Positive", "Negative", "Neutral"], "instruction": self.doc_to_text(doc)}
        ]
    
    def doc_to_choice(self, doc):
        return ["Positive", "Negative", "Neutral"]


@register_task("mmlu_te")
class MMLUTelugu(Task):
    VERSION = 1
    DATASET_PATH = "sarvamai/mmlu-indic"
    
    def __init__(self):
        super().__init__()
        self._dataset = None
    
    def download(self):
        self._dataset = datasets.load_dataset(self.DATASET_PATH)
        # Filter for Telugu samples
        self._dataset = self._dataset.filter(lambda x: x["language"] == "te")
    
    def has_training_docs(self):
        return False
    
    def has_validation_docs(self):
        return False
    
    def has_test_docs(self):
        return True
    
    def training_docs(self):
        return []
    
    def validation_docs(self):
        return []
    
    def test_docs(self):
        if self._test_docs is None:
            self._test_docs = list(self._dataset["test"])
        return self._test_docs
    
    def doc_to_text(self, doc):
        return f"{doc['question']}\n\nA. {doc['choices'][0]}\nB. {doc['choices'][1]}\nC. {doc['choices'][2]}\nD. {doc['choices'][3]}\n\nAnswer:"
    
    def doc_to_target(self, doc):
        return doc["answer"]
    
    def process_results(self, docs, results):
        preds = []
        labels = []
        
        for pred, doc in zip(results, docs):
            pred_label = ["A", "B", "C", "D"][pred.argmax()]
            preds.append(pred_label)
            labels.append(doc["answer"])
        
        # Calculate accuracy
        correct = sum(p == l for p, l in zip(preds, labels))
        return {"accuracy": correct / len(docs)}
    
    def construct_requests(self, doc, ctx):
        return [
            {"doc": doc, "choices": ["A", "B", "C", "D"], "instruction": self.doc_to_text(doc)}
        ]
    
    def doc_to_choice(self, doc):
        return ["A", "B", "C", "D"]
EOL

# Update the __init__.py to import the tasks
cat > lm_eval/tasks/custom/__init__.py << 'EOL'
from . import telugu_tasks
print("Successfully imported Telugu tasks!")
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