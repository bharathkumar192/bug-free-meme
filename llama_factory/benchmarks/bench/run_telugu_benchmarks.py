#!/usr/bin/env python
# run_telugu_benchmarks.py

import os
import argparse
import json
import datetime
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Telugu language benchmarks on a fine-tuned model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to your fine-tuned model or model identifier on Hugging Face")
    parser.add_argument("--model_type", type=str, default="hf",
                        choices=["hf", "vllm"],
                        help="Model type to use for evaluation (hf or vllm)")
    parser.add_argument("--tasks", type=str, default="telugu_sentiment,mmlu_telugu",
                        help="Comma-separated list of tasks to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of examples to use for few-shot evaluation")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run evaluation on")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code from Hugging Face")
    
    return parser.parse_args()

def setup_task_configs():
    """
    Ensure custom task configs are in the right location
    """
    src_dir = Path(".")
    dst_dir = Path("lm_eval/tasks/custom_tasks")
    
    for yaml_file in ["telugu_sentiment.yaml", "mmlu_telugu.yaml"]:
        src_file = src_dir / yaml_file
        dst_file = dst_dir / yaml_file
        
        # Only copy if source exists and destination doesn't
        if src_file.exists() and not dst_file.exists():
            print(f"Copying {yaml_file} to {dst_dir}")
            os.makedirs(dst_dir, exist_ok=True)
            with open(src_file, 'r') as f_src:
                content = f_src.read()
                with open(dst_file, 'w') as f_dst:
                    f_dst.write(content)

def run_benchmark(args):
    """
    Run the benchmark evaluations
    """
    # Setup task configs
    setup_task_configs()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.model_path)
    results_dir = os.path.join(args.output_dir, f"{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Build the command
    model_args = f"pretrained={args.model_path}"
    if args.trust_remote_code:
        model_args += ",trust_remote_code=True"
    
    cmd = f"python -m lm_eval \
        --model {args.model_type} \
        --model_args {model_args} \
        --tasks {args.tasks} \
        --device {args.device} \
        --batch_size {args.batch_size} \
        --num_fewshot {args.num_fewshot} \
        --output_path {os.path.join(results_dir, 'results.json')} \
        --include_path lm_eval/tasks/custom_tasks \
        --log_samples"
    
    print(f"Running benchmark with command:\n{cmd}")
    os.system(cmd)
    
    # Generate a summary report
    generate_summary(results_dir, args)
    
    return results_dir

def generate_summary(results_dir, args):
    """
    Generate a human-readable summary of the benchmark results
    """
    results_file = os.path.join(results_dir, 'results.json')
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found at {results_file}")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        summary_file = os.path.join(results_dir, 'summary.md')
        with open(summary_file, 'w') as f:
            f.write(f"# Telugu Benchmark Results\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Task | Metric | Value | Std Error |\n")
            f.write("|------|--------|-------|----------|\n")
            
            for task, task_results in results['results'].items():
                for metric, value in task_results.items():
                    if not metric.endswith('_stderr'):
                        std_err = task_results.get(f"{metric}_stderr", "N/A")
                        f.write(f"| {task} | {metric} | {value:.4f} | {std_err} |\n")
            
            f.write("\n\n## Configuration\n\n")
            f.write("```\n")
            f.write(f"Tasks: {args.tasks}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Few-shot examples: {args.num_fewshot}\n")
            f.write(f"Device: {args.device}\n")
            f.write("```\n")
        
        print(f"Summary generated at {summary_file}")
    
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    results_dir = run_benchmark(args)
    print(f"Benchmark completed. Results saved to {results_dir}")