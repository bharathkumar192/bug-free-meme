#!/bin/bash
# benchmark_telugu_model.sh

# Set your model path
MODEL_PATH="bharathkumar1922001/gemma-3-12b-pt-telugu-sft"  # Your model path
DEVICE="cuda:0"  # or "cuda:0,cuda:1" for multiple GPUs

# Set up environment and prepare datasets
bash setup_benchmark.sh
python prepare_indic_sentiment_telugu.py
python prepare_mmlu_telugu.py

# Run the benchmarks
python run_telugu_benchmarks.py \
    --model_path $MODEL_PATH \
    --model_type hf \
    --device $DEVICE \
    --batch_size 4 \
    --num_fewshot 5 \
    --trust_remote_code \
    --output_dir "./telugu_benchmark_results"

echo "Benchmarking completed!"