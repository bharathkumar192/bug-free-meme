#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for fine-tuned Gemma-3-12B Telugu model
This script evaluates the model on a test set and computes metrics
"""

import json
import argparse
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# For metrics
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma model on Telugu test data")
    
    # Model arguments
    parser.add_argument("--base_model_path", type=str, required=True, 
                        help="Path to base model or Hugging Face model ID")
    parser.add_argument("--adapter_path", type=str, default=None, 
                        help="Path to LoRA adapter (if using LoRA)")
    parser.add_argument("--merged_model_path", type=str, default=None, 
                        help="Path to merged model (if not using adapter)")
    
    # Data arguments
    parser.add_argument("--test_file", type=str, required=True, 
                        help="Path to test data JSON file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", 
                        help="Directory to save evaluation results")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, 
                        help="Top-k sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1, 
                        help="Number of beams for beam search (1 means greedy)")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for evaluation")
    
    # Miscellaneous
    parser.add_argument("--use_4bit", action="store_true", 
                        help="Use 4-bit quantization for faster evaluation")
    parser.add_argument("--use_8bit", action="store_true", 
                        help="Use 8-bit quantization for faster evaluation")
    parser.add_argument("--use_fp16", action="store_true", 
                        help="Use FP16 for faster evaluation")
    parser.add_argument("--use_bf16", action="store_true", 
                        help="Use BF16 for faster evaluation")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load model and tokenizer based on arguments"""
    print("Loading model and tokenizer...")
    
    # Determine data type
    dtype = torch.float32
    if args.use_fp16:
        dtype = torch.float16
    elif args.use_bf16:
        dtype = torch.bfloat16
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, 
        trust_remote_code=True,
        use_fast=True
    )
    
    # Ensure the tokenizer has padding settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Quantization configuration
    quantization_config = None
    if args.use_4bit or args.use_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            load_in_8bit=args.use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model based on provided arguments
    if args.merged_model_path:
        # Load merged model (full or merged LoRA)
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_model_path,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
    elif args.adapter_path:
        # Load base model and adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            torch_dtype=dtype,
            device_map="auto"
        )
    else:
        # Load just base model
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def format_prompt(question: str, tokenizer) -> str:
    """Format the question using Gemma-3 template format"""
    # Following Gemma-3 chat template
    system_prompt = "You are an AI assistant that follows instructions and responds in Telugu language."
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    return prompt

def generate_responses(model, tokenizer, test_data, args):
    """Generate responses for test questions"""
    print("Generating responses...")
    
    results = []
    
    # Process in batches
    for i in range(0, len(test_data), args.batch_size):
        batch = test_data[i:i + args.batch_size]
        batch_questions = [item["question"] for item in batch]
        batch_references = [item["response"] for item in batch]
        
        # Format prompts
        batch_prompts = [format_prompt(q, tokenizer) for q in batch_questions]
        
        # Tokenize
        batch_inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=(args.temperature > 0)
            )
        
        # Decode generated responses
        batch_generated = tokenizer.batch_decode(
            outputs[:, batch_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clean up responses (remove trailing model tokens if any)
        batch_generated = [resp.split("<end_of_turn>")[0].strip() for resp in batch_generated]
        
        # Store results
        for q, ref, gen in zip(batch_questions, batch_references, batch_generated):
            results.append({
                "question": q,
                "reference": ref,
                "generated": gen
            })
    
    return results

def compute_metrics(results):
    """Compute evaluation metrics"""
    print("Computing metrics...")
    
    # Initialize metrics
    bleu = BLEU()
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    # Store metrics
    metrics = {
        "bleu": [],
        "rouge1_f": [],
        "rouge2_f": [],
        "rougeL_f": []
    }
    
    # Compute metrics for each result
    for result in results:
        # BLEU score
        bleu_score = bleu.corpus_score(
            [result["generated"]], 
            [[result["reference"]]]
        ).score
        metrics["bleu"].append(bleu_score)
        
        # ROUGE scores
        rouge_scores = scorer.score(result["reference"], result["generated"])
        metrics["rouge1_f"].append(rouge_scores["rouge1"].fmeasure)
        metrics["rouge2_f"].append(rouge_scores["rouge2"].fmeasure)
        metrics["rougeL_f"].append(rouge_scores["rougeL"].fmeasure)
    
    # Calculate averages
    avg_metrics = {
        "bleu": np.mean(metrics["bleu"]),
        "rouge1_f": np.mean(metrics["rouge1_f"]),
        "rouge2_f": np.mean(metrics["rouge2_f"]),
        "rougeL_f": np.mean(metrics["rougeL_f"])
    }
    
    return avg_metrics, metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load test data
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # Extract test samples (assuming the same format as the training data)
    test_samples = []
    for item in test_data.get("questions", []):
        test_samples.append({
            "question": item["question"],
            "response": item["response"]
        })
    
    # Generate responses
    start_time = time.time()
    results = generate_responses(model, tokenizer, test_samples, args)
    generation_time = time.time() - start_time
    
    # Compute metrics
    avg_metrics, all_metrics = compute_metrics(results)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Number of test samples: {len(test_samples)}")
    print(f"Total generation time: {generation_time:.2f} seconds")
    print(f"Average generation time per sample: {generation_time / len(test_samples):.2f} seconds")
    print("\nAverage Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    output = {
        "config": vars(args),
        "results": results,
        "avg_metrics": avg_metrics,
        "all_metrics": all_metrics,
        "generation_time": generation_time
    }
    
    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()