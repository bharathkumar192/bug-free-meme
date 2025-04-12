#!/bin/bash
# setup_benchmark.sh

# Clone the lm-evaluation-harness repository from the main branch
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness

# Install dependencies
pip install -e .
pip install -e ".[vllm]"  # Optional: for faster inference
pip install -e ".[multilingual]"  # For multilingual support
pip install datasets huggingface_hub

# Create directories for custom tasks
mkdir -p lm_eval/tasks/custom_tasks
mkdir -p ../tasks_data  # Changed to be relative to the parent directory

echo "Setup completed successfully!"