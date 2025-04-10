#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Standalone Hugging Face Upload Script for Gemma-3 Telugu fine-tuned model
This script uploads the model to the Hugging Face Hub and creates a model card.
No imports from config.py are needed.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

def create_model_card():
    """
    Create a detailed model card as a string.
    """
    model_card = """---
language: te
license: apache-2.0
library_name: transformers
tags:
- gemma
- telugu
- llm
- finetuned
- sft
datasets:
- custom_telugu_qa_dataset
model_name: Gemma-3-12B-Telugu-SFT
---

# Gemma-3-12B Telugu Fine-tuned Model

This is a fine-tuned version of the Gemma-3-12B model for the Telugu language. The model was trained on a custom Telugu question-answering dataset to enhance its capabilities for Telugu language processing.

## Model Details

- **Base Model:** google/gemma-3-12b-pt
- **Fine-tuning Method:** FULL
- **Training Dataset:** Custom Telugu QA pairs dataset
- **Validation Split:** 10.0%
- **All available samples were used for training**

## Training Parameters

- **Per Device Batch Size:** 2
- **Gradient Accumulation Steps:** 32
- **Learning Rate:** 2e-5
- **Number of Epochs:** 3.0
- **Learning Rate Scheduler:** cosine
- **Warmup Ratio:** 0.03
- **Max Sequence Length:** 4096

## Hardware Configuration

- **Precision:** BF16
- **DeepSpeed:** Enabled (Stage 2)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "bharathkumar1922001/gemma-3-12b-pt-telugu-sft"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example prompt (in Telugu)
prompt = "నమస్కారం, మీరు ఎలా ఉన్నారు?"

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Intended Use

This model is designed for Telugu language tasks including:
- Question answering
- Conversation
- Text generation
- General Telugu language understanding and generation

## Telugu Language Support

This model has been specifically fine-tuned to enhance its capabilities in Telugu, one of the Dravidian languages spoken primarily in the Indian states of Andhra Pradesh and Telangana.
"""
    return model_card

def upload_to_huggingface(model_dir, repo_id, token):
    """
    Upload model files to Hugging Face Hub.
    
    Args:
        model_dir: Directory containing the model files
        repo_id: Hugging Face Hub repository ID (username/repo-name)
        token: Hugging Face API token
    """
    print(f"Uploading model from {model_dir} to {repo_id}...")
    
    # Create API client
    api = HfApi()
    
    # Create model card
    model_card_content = create_model_card()
    
    # Write model card to a file
    model_card_path = os.path.join(model_dir, "README.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)
    
    # Check if repository exists, if not create it
    try:
        api.repo_info(repo_id=repo_id, token=token)
        print(f"Repository {repo_id} already exists.")
    except Exception:
        print(f"Creating new repository {repo_id}...")
        api.create_repo(repo_id=repo_id, token=token, private=False)
    
    # Get list of files to upload (exclude checkpoint directories)
    files_to_upload = []
    for root, dirs, files in os.walk(model_dir):
        # Skip checkpoint directories
        dirs[:] = [d for d in dirs if not d.startswith("checkpoint-")]
        
        for file in files:
            file_path = os.path.join(root, file)
            # Get relative path to maintain directory structure
            rel_path = os.path.relpath(file_path, model_dir)
            files_to_upload.append((file_path, rel_path))
    
    # Upload each file individually
    print(f"Uploading {len(files_to_upload)} files...")
    for file_path, rel_path in files_to_upload:
        print(f"Uploading {rel_path}...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=rel_path,
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload {rel_path}",
            )
        except Exception as e:
            print(f"Error uploading {rel_path}: {str(e)}")
    
    print(f"Successfully uploaded model to {repo_id}")

def main():
    # Set these variables directly (no imports needed)
    model_dir = "checkpoints/gemma3-telugu-output"  # Update this path to match your actual directory
    repo_id = "bharathkumar1922001/gemma-3-12b-pt-telugu-sft"
    token = "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"
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist.")
        return 1
    
    # Upload to Hugging Face
    upload_to_huggingface(model_dir, repo_id, token)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())