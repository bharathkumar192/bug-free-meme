#!/bin/bash

# Shell script to push specific files from current directory to Hugging Face Hub
# Usage: ./push_to_hf.sh -t <token> -r <repo_id> -f <file1,file2,...>

set -e  # Exit on error

# ANSI color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
HF_TOKEN="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"
REPO_ID="maya-ai/randomized_llm"
FILES="tests_results.json, inference.log"
REPO_TYPE="model"  # Default repo type

# Function to display usage
usage() {
  echo -e "${BLUE}Usage:${NC} $0 [options]"
  echo ""
  echo "Options:"
  echo "  -t, --token TOKEN       Hugging Face API token (required)"
  echo "  -r, --repo REPO_ID      Repository ID (username/repo-name) (required)"
  echo "  -f, --files FILES       Comma-separated list of files to upload (required)"
  echo "  -y, --type TYPE         Repository type: model, dataset, space (default: model)"
  echo "  -h, --help              Display this help message"
  echo ""
  echo "Example:"
  echo "  $0 -t hf_xxxxxxxx -r username/model-name -f model.safetensors,config.json"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--token)
      HF_TOKEN="$2"
      shift 2
      ;;
    -r|--repo)
      REPO_ID="$2"
      shift 2
      ;;
    -f|--files)
      FILES="$2"
      shift 2
      ;;
    -y|--type)
      REPO_TYPE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo -e "${RED}Error: Unknown option $1${NC}"
      usage
      ;;
  esac
done

# Check required parameters
if [ -z "$HF_TOKEN" ]; then
  echo -e "${RED}Error: Hugging Face token is required${NC}"
  usage
fi

if [ -z "$REPO_ID" ]; then
  echo -e "${RED}Error: Repository ID is required${NC}"
  usage
fi

if [ -z "$FILES" ]; then
  echo -e "${RED}Error: File list (-f) must be specified${NC}"
  usage
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check and install required packages
echo "Installing required Python packages..."
python3 -m pip install huggingface_hub tqdm --quiet

# Create a temporary Python script for uploading
TEMP_SCRIPT=$(mktemp)
echo "Creating temporary upload script..."

cat > "$TEMP_SCRIPT" << 'ENDPYTHON'
import os
import sys
import time
from huggingface_hub import login, upload_file
from tqdm import tqdm
import argparse

def upload_files(token, repo_id, file_list, repo_type):
    """Upload specific files to HF Hub"""
    # Login to Hugging Face
    login(token=token)
    print(f"Authenticated with Hugging Face Hub")
    
    # Track success and failures
    successful = []
    failed = []
    
    # Process each file
    for file_name in file_list:
        if not os.path.exists(file_name):
            print(f"Warning: File {file_name} does not exist in current directory. Skipping.")
            failed.append(file_name)
            continue
        
        file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
        print(f"Uploading {file_name} ({file_size_mb:.2f} MB) to {repo_id}...")
        
        try:
            # For large files, show progress
            if file_size_mb > 10:
                with tqdm(total=100, desc=f"Uploading {file_name}", unit="%") as pbar:
                    last_percent = 0
                    
                    def progress_callback(current, total):
                        nonlocal last_percent
                        percent = int(current * 100 / total)
                        if percent > last_percent:
                            pbar.update(percent - last_percent)
                            last_percent = percent
                    
                    upload_file(
                        path_or_fileobj=file_name,
                        path_in_repo=file_name,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                        progress_callback=progress_callback
                    )
            else:
                upload_file(
                    path_or_fileobj=file_name,
                    path_in_repo=file_name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token
                )
            
            successful.append(file_name)
            print(f"✓ Successfully uploaded {file_name}")
            
        except Exception as e:
            print(f"✗ Error uploading {file_name}: {str(e)}")
            failed.append(file_name)
    
    return successful, failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files to Hugging Face Hub")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID (username/repo-name)")
    parser.add_argument("--files", type=str, required=True, help="Comma-separated list of files to upload")
    parser.add_argument("--repo_type", type=str, default="model", help="Repository type (model, dataset, space)")
    
    args = parser.parse_args()
    
    # Parse file list
    file_list = [f.strip() for f in args.files.split(",")]
    
    # Start timing
    start_time = time.time()
    
    # Upload files
    successful, failed = upload_files(args.token, args.repo_id, file_list, args.repo_type)
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Upload Summary:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Files uploaded successfully: {len(successful)}/{len(file_list)}")
    
    if successful:
        print("\nSuccessful uploads:")
        for file in successful:
            print(f"  - {file}")
    
    if failed:
        print("\nFailed uploads:")
        for file in failed:
            print(f"  - {file}")
    
    print("=" * 50)
    
    # Return exit code based on success
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)
ENDPYTHON

# Execute the Python script
echo -e "${BLUE}Starting upload to Hugging Face Hub...${NC}"
python3 "$TEMP_SCRIPT" --token "$HF_TOKEN" --repo_id "$REPO_ID" --files "$FILES" --repo_type "$REPO_TYPE"

# Clean up
rm "$TEMP_SCRIPT"
echo -e "${GREEN}Upload process completed.${NC}"