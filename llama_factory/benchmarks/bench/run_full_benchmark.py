#!/usr/bin/env python
# run_full_benchmark.py

import os
import argparse
import subprocess
import datetime
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run full Telugu benchmarking pipeline")
    
    parser.add_argument("--model_path", required=True, 
                        help="Path to the model or model ID on Hugging Face")
    parser.add_argument("--model_name", default=None,
                        help="Custom name for the model (defaults to model_path basename)")
    parser.add_argument("--output_dir", default="telugu_benchmarks",
                        help="Output directory for benchmark results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=5,
                        help="Number of few-shot examples to use")
    parser.add_argument("--device", default="cuda:0",
                        help="Device to use for evaluation")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code from Hugging Face")
    parser.add_argument("--skip_setup", action="store_true",
                        help="Skip setup if already done")
    
    return parser.parse_args()

def run_command(cmd, description=None):
    """Run a command and log its output"""
    if description:
        print(f"\n{'='*80}\n{description}\n{'='*80}")
    
    print(f"Running: {cmd}")
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, check=True)
    duration = time.time() - start_time
    
    print(f"Command completed in {duration:.2f} seconds with exit code {result.returncode}")
    return result.returncode == 0

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model name
    model_name = args.model_name or os.path.basename(args.model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Log parameters
    with open(os.path.join(run_dir, "benchmark_params.txt"), "w") as f:
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Few-shot examples: {args.num_fewshot}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Trust remote code: {args.trust_remote_code}\n")
    
    # Setup environment and datasets
    if not args.skip_setup:
        run_command("bash setup_benchmark.sh", "Setting up environment")
        run_command("python prepare_indic_sentiment_telugu.py", "Preparing Telugu sentiment dataset")
        run_command("python prepare_mmlu_telugu.py", "Preparing Telugu MMLU dataset")
    
    # Run benchmarks
    trust_flag = "--trust_remote_code" if args.trust_remote_code else ""
    benchmark_cmd = f"python run_telugu_benchmarks.py \
        --model_path {args.model_path} \
        --device {args.device} \
        --batch_size {args.batch_size} \
        --num_fewshot {args.num_fewshot} \
        --output_dir {run_dir} \
        {trust_flag}"
    
    benchmark_success = run_command(benchmark_cmd, "Running benchmarks")
    
    # Run analysis if benchmarks succeeded
    if benchmark_success:
        analysis_cmd = f"python analyze_results.py --results_dir {run_dir}"
        run_command(analysis_cmd, "Analyzing results")
        
        print(f"\n{'='*80}")
        print(f"Benchmark completed successfully!")
        print(f"Results saved to: {run_dir}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Benchmark failed.")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()