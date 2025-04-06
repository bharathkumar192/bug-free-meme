#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for Telugu dataset
This script converts your JSON format to the format required by LLaMA Factory
"""

import json
import os
import argparse
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Telugu QA data to LLaMA Factory format")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with questions and responses")
    parser.add_argument("--output_dir", type=str, default="./data/telugu", help="Output directory for processed data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    return parser.parse_args()

def convert_to_alpaca_format(data):
    """Convert the Telugu QA data to Alpaca format used by LLaMA Factory"""
    formatted_data = []
    
    # Extract questions from the data
    questions = data.get("questions", [])
    
    for item in questions:
        formatted_item = {
            "instruction": item["prompt"],
            "input": "",  # No separate input in this format
            "output": item["response"]
        }
        formatted_data.append(formatted_item)
    
    return formatted_data

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the input data
    print(f"Loading data from {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to Alpaca format
    formatted_data = convert_to_alpaca_format(data)
    print(f"Converted {len(formatted_data)} QA pairs to Alpaca format")
    
    # Limit samples if specified
    if args.max_samples and args.max_samples < len(formatted_data):
        random.shuffle(formatted_data)
        formatted_data = formatted_data[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")
    
    # Split into train and validation sets
    if args.val_split > 0:
        random.shuffle(formatted_data)
        split_idx = int(len(formatted_data) * (1 - args.val_split))
        train_data = formatted_data[:split_idx]
        val_data = formatted_data[split_idx:]
        
        print(f"Split data into {len(train_data)} training and {len(val_data)} validation samples")
        
        # Save training data
        train_file = os.path.join(args.output_dir, "telugu_train.json")
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # Save validation data
        val_file = os.path.join(args.output_dir, "telugu_val.json")
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved training data to {train_file}")
        print(f"Saved validation data to {val_file}")
    else:
        # Save all data without splitting
        output_file = os.path.join(args.output_dir, "telugu_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved all data to {output_file}")
    
    # Create dataset_info.json for LLaMA Factory
    dataset_info = {
        "telugu_data": {
            "file_name": "telugu_data.json",
            "ranking": "sft"
        }
    }
    
    if args.val_split > 0:
        dataset_info = {
            "telugu_train": {
                "file_name": "telugu_train.json",
                "ranking": "sft"
            },
            "telugu_val": {
                "file_name": "telugu_val.json",
                "ranking": "sft"
            }
        }
    
    # Save dataset info
    dataset_info_file = os.path.join(args.output_dir, "dataset_info.json")
    with open(dataset_info_file, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"Saved dataset info to {dataset_info_file}")
    print("Data preparation completed!")

if __name__ == "__main__":
    main()