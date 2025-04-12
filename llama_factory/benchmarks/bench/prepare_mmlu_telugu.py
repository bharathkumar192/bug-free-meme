# prepare_mmlu_telugu.py

import os
import json
from datasets import load_dataset

def download_and_filter_mmlu_indic():
    """
    Download and filter the MMLU Indic dataset for Telugu.
    """
    print("Downloading MMLU Indic dataset...")
    dataset = load_dataset("sarvamai/mmlu-indic", trust_remote_code=True)
    
    # Extract Telugu data
    print("Filtering for Telugu data...")
    
    telugu_data = {}
    for split in dataset:
        # Filter for Telugu language
        telugu_examples = [ex for ex in dataset[split] if ex['language'] == 'te']
        telugu_data[split] = telugu_examples
        
        # Save to disk
        os.makedirs("tasks_data/mmlu_telugu", exist_ok=True)
        
        # Format data for easier consumption by lm-evaluation-harness
        formatted_data = []
        for example in telugu_examples:
            formatted_example = {
                "question": example["question"],
                "choices": example["choices"],
                "answer": example["answer"]
            }
            formatted_data.append(formatted_example)
        
        output_file = f"tasks_data/mmlu_telugu/{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(formatted_data)} Telugu examples to {output_file}")

if __name__ == "__main__":
    download_and_filter_mmlu_indic()