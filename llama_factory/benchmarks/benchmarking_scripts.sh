#!/bin/bash
# Telugu Model Benchmarking Script
# This script runs multiple benchmarks on your fine-tuned Telugu model and saves results

# Configuration
MODEL_PATH="bharathkumar1922001/gemma-3-12b-pt-telugu-sft"  # Your fine-tuned model
OUTPUT_DIR="benchmark_results"
DEVICE="cuda:0,1"  # Change to your desired GPU

# Create output directory
mkdir -p $OUTPUT_DIR

# 1. IndicSentiment - Text Classification (0-shot)
echo "Running IndicSentiment benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks indicsentiment_te \
    --device $DEVICE \
    --batch_size auto \
    --output_path $OUTPUT_DIR/indicsentiment_te_results.json \
    --log_samples

# 2. IndicXParaphrase - Text Generation (0-shot)
echo "Running IndicXParaphrase benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks indicxparaphrase_te \
    --device $DEVICE \
    --batch_size auto \
    --output_path $OUTPUT_DIR/indicxparaphrase_te_results.json \
    --log_samples

# 3. IndicQA - Question Answering (0-shot)
echo "Running IndicQA benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks indicqa_te \
    --device $DEVICE \
    --batch_size auto \
    --output_path $OUTPUT_DIR/indicqa_te_results.json \
    --log_samples

# 4. FloresIN Translation te-en (1-shot)
echo "Running FloresIN Translation (te-en) benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks flores_te_en \
    --device $DEVICE \
    --batch_size auto \
    --output_path $OUTPUT_DIR/flores_te_en_results.json \
    --log_samples

# 5. FloresIN Translation en-te (1-shot)
echo "Running FloresIN Translation (en-te) benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks flores_en_te \
    --device $DEVICE \
    --batch_size auto \
    --output_path $OUTPUT_DIR/flores_en_te_results.json \
    --log_samples

# 6. MILU - Telugu General (0-shot)
echo "Running MILU Telugu benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks milu_te_general \
    --device $DEVICE \
    --batch_size auto \
    --output_path $OUTPUT_DIR/milu_te_results.json \
    --log_samples

# 7. Grammar Correction (5-shot)
echo "Running Grammar Correction benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks grammar_correction_te \
    --device $DEVICE \
    --batch_size auto \
    --output_path $OUTPUT_DIR/grammar_correction_te_results.json \
    --num_fewshot 5 \
    --log_samples

# 8. Multi-turn Conversation (0-shot)
echo "Running Multi-turn Conversation benchmark..."
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks multiturn_te \
    --device $DEVICE \
    --batch_size auto \
    --output_path $OUTPUT_DIR/multiturn_te_results.json \
    --log_samples

echo "All benchmarks completed. Results saved to $OUTPUT_DIR"