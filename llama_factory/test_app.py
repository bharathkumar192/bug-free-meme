import modal
import os
import json
from pathlib import Path
import time
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define CUDA tag for container
tag = "12.4.0-devel-ubuntu20.04"

# Create the image
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.9")
    .apt_install(
        "python3", 
        "python3-pip", 
        "python3-dev",
        "build-essential",
        "wget",
        "git"
    )
    # Install necessary packages
    .run_commands("pip install torch transformers accelerate sentencepiece protobuf")
    # Add HF token for accessing the model
    .env({"HF_TOKEN": "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"})
    # Add the inference code and prompts file
    .add_local_dir(".", remote_path="/app", copy=True)
)

# Create cache volume for huggingface models
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("inference-results", create_if_missing=True)

# Create the app
app = modal.App("gemma-telugu-inference", image=image)

@app.function(
    gpu="H100",  # Request an H100 GPU
    timeout=3600,  # 1-hour timeout
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol
    },
    memory=32768,  # 32GB memory
)
def run_inference():
    """Run inference with the fine-tuned model on Telugu questions."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Record start time
    start_time = time.time()
    
    # Path to prompts file
    prompts_file = "prompts.json"
    
    # Load prompts file
    logger.info(f"Loading prompts from {prompts_file}")
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    logger.info(f"Loaded {len(questions)} questions")
    
    # Model configuration
    model_name = "bharathkumar1922001/gemma-3-12b-pt-telugu-sft"
    
    # Load tokenizer and model
    logger.info(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=os.environ.get("HF_TOKEN")
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use BF16 for efficiency
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=os.environ.get("HF_TOKEN")
    )
    
    logger.info("Model loaded successfully")
    
    # Configure generation parameters
    generation_config = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True
    }
    
    # Record results
    results = []
    
    # Process each question
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}")
        
        # Format using Gemma's chat template
        prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                **generation_config,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode output and extract model's response
        full_output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
        
        # Extract only the model's response by removing the input prompt
        model_response = full_output.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
        
        # Add to results
        results.append({
            "question": question,
            "response": model_response
        })
        
        logger.info(f"Response: {model_response[:100]}...")
    
    # Save results
    results_file = "/results/telugu_inference_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "results": results,
            "metadata": {
                "generation_config": generation_config,
                "inference_time": time.time() - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Total inference time: {time.time() - start_time:.2f} seconds")
    
    # Also create a human-readable format
    readable_results_file = "/results/telugu_inference_results.txt"
    with open(readable_results_file, "w", encoding="utf-8") as f:
        f.write(f"# Telugu Inference Results for Model: {model_name}\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Inference time: {time.time() - start_time:.2f} seconds\n\n")
        
        for i, item in enumerate(results):
            f.write(f"## Question {i+1}\n\n")
            f.write(f"**Q:** {item['question']}\n\n")
            f.write(f"**A:** {item['response']}\n\n")
            f.write("-" * 80 + "\n\n")
    
    logger.info(f"Human-readable results saved to {readable_results_file}")
    
    return {
        "status": "success",
        "processed_questions": len(questions),
        "inference_time": time.time() - start_time,
        "results_path": results_file
    }

@app.local_entrypoint()
def main():
    """Main entry point for the Modal application."""
    logger.info("Starting Telugu model inference...")
    
    # Make sure prompts.json exists
    prompts_file = Path("prompts.json")
    if not prompts_file.exists():
        logger.error(f"Error: {prompts_file} not found. Please create this file with your test questions.")
        return
    
    # Run inference
    result = run_inference.remote()
    
    logger.info("\n--- Inference Results ---")
    logger.info(f"Status: {result.get('status', 'Unknown')}")
    
    if result.get('status') == "success":
        logger.info(f"Processed {result.get('processed_questions', 0)} questions")
        logger.info(f"Total inference time: {result.get('inference_time', 0):.2f} seconds")
        logger.info(f"Results saved to: {result.get('results_path', 'Unknown')}")
        logger.info("\nYou can check the results in the 'inference-results' Modal volume.")
    else:
        logger.error(f"Inference failed: {result.get('message', 'No error message provided.')}")
    
    logger.info("--- Run Finished ---")