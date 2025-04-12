#!/bin/bash
# Script to upload benchmark results to Hugging Face Hub

set -e # Exit on error

# --- Configuration ---
# !! IMPORTANT: Your Hugging Face Hub token (ensure it has write access) !!
# Using an environment variable is safer than hardcoding directly in the script.
export HF_TOKEN="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" 

# Absolute path to the directory containing the results JSON files inside the container/machine
# Based on your previous logs, this should be correct.
RESULTS_DIR="/bug-free-meme/benchmarks/telugu_benchmark_results" 

# Hugging Face Hub repository ID (username/repo_name)
REPO_ID="bharathkumar1922001/Gemma3-12b-Indic"

# Optional: Subdirectory within the HF repo to upload results to.
# Creates the directory if it doesn't exist. Good for organization.
# Leave empty ("") to upload to the repository root.
PATH_IN_REPO="lm_evaluation_harness_results" 
# --- End Configuration ---

echo "--- Starting Hugging Face Hub Upload ---"
echo "Target Repository: $REPO_ID"
echo "Results Source Directory: $RESULTS_DIR"
[ -n "$PATH_IN_REPO" ] && echo "Target Path in Repo: $PATH_IN_REPO"

# 1. Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found at $RESULTS_DIR"
    exit 1
fi

# 2. Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo "Please set your Hugging Face Hub token (with write access) at the top of this script."
    exit 1
fi
echo "Using provided HF Token for authentication."

# 3. Install/Upgrade huggingface-hub library (needed for the upload function)
echo "Installing/Updating huggingface-hub library..."
pip install --upgrade huggingface-hub --quiet

# 4. Run Python script using a heredoc to perform the upload
echo "Attempting to upload contents..."

python - <<EOF
from huggingface_hub import upload_folder
import os
import sys

results_dir = "$RESULTS_DIR"
repo_id = "$REPO_ID"
path_in_repo = "$PATH_IN_REPO" if len("$PATH_IN_REPO") > 0 else None # Use None if path_in_repo is empty string
token = os.getenv("HF_TOKEN")

try:
    print(f"Uploading contents of '{results_dir}' to '{repo_id}'" + (f" in folder '{path_in_repo}'" if path_in_repo else " at root") + "...")
    
    # Construct commit message including directory name for context
    commit_msg = f"Add evaluation results from {os.path.basename(results_dir)}"

    upload_folder(
        folder_path=results_dir,
        repo_id=repo_id,
        repo_type="model", # Uploading to a model repository
        path_in_repo=path_in_repo, # Target directory in the repo (optional)
        token=token,
        commit_message=commit_msg,
        # ignore_patterns=["*.ckpt", "*.safetensors"], # Optional: ignore specific file patterns if needed
    )
    print("--- Upload successful! ---")
    print(f"View files at: https://huggingface.co/{repo_id}/tree/main" + (f"/{path_in_repo}" if path_in_repo else ""))

except Exception as e:
    print(f"ERROR: Upload failed: {e}", file=sys.stderr)
    sys.exit(1) # Exit with a non-zero code to indicate failure to the shell
EOF

# Capture the exit code from the python script
upload_status=$? 

if [ $upload_status -eq 0 ]; then
    echo "--- Upload script finished successfully. ---"
else
    echo "--- Upload script finished with errors. ---"
    exit $upload_status # Propagate the error code
fi