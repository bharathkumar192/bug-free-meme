#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict, Any, Optional
import yaml
import subprocess
import logging
import random
from pathlib import Path
import copy
from huggingface_hub import HfApi

# Optional import for Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import configuration
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments to override config values."""
    parser = argparse.ArgumentParser(
        description='Fine-tune Gemma-3-12B model on Telugu data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--input_file', type=str, default=config.INPUT_FILE,
                        help='Path to input JSON file containing Telugu QA pairs')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help='Directory to save model and checkpoints')
    parser.add_argument('--dataset_dir', type=str, default=config.DATASET_DIR,
                        help='Directory to save processed dataset')
    parser.add_argument('--max_samples', type=int, default=config.MAX_SAMPLES,
                        help='Maximum number of samples to use for training')
    parser.add_argument('--val_size', type=float, default=config.VAL_SIZE,
                        help='Size of validation split')
    
    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, default=config.MODEL_NAME_OR_PATH,
                        help='Path to pre-trained model or Hugging Face model identifier')
    parser.add_argument('--use_flash_attn', action='store_true', default=config.USE_FLASH_ATTN,
                        help='Use Flash Attention 2 for faster training')
    
    # Training method arguments
    parser.add_argument('--finetuning_type', type=str, default=config.FINETUNING_TYPE,
                        choices=['lora', 'qlora', 'full'], help='Fine-tuning method to use')
    
    # LoRA configuration
    parser.add_argument('--lora_rank', type=int, default=config.LORA_CONFIG["lora_rank"],
                        help='Rank for LoRA adaptation')
    parser.add_argument('--lora_alpha', type=int, default=config.LORA_CONFIG["lora_alpha"],
                        help='Alpha for LoRA adaptation')
    parser.add_argument('--lora_dropout', type=float, default=config.LORA_CONFIG["lora_dropout"],
                        help='Dropout for LoRA adaptation')
    
    # QLoRA configuration
    parser.add_argument('--quantization_bit', type=int, default=config.QLORA_CONFIG["quantization_bit"] if config.FINETUNING_TYPE == "qlora" else None,
                        choices=[4, 8], help='Quantization bit for QLoRA')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=config.TRAINING_CONFIG["per_device_train_batch_size"],
                        help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=config.TRAINING_CONFIG["gradient_accumulation_steps"],
                        help='Number of updates steps to accumulate before backward pass')
    parser.add_argument('--num_epochs', type=float, default=config.TRAINING_CONFIG["num_train_epochs"],
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=config.TRAINING_CONFIG["learning_rate"],
                        help='Learning rate')
    parser.add_argument('--cutoff_len', type=int, default=config.TRAINING_CONFIG["cutoff_len"],
                        help='Maximum sequence length after tokenization')
    parser.add_argument('--warmup_ratio', type=float, default=config.TRAINING_CONFIG["warmup_ratio"],
                        help='Warm up ratio for learning rate scheduler')
    parser.add_argument('--logging_steps', type=int, default=config.TRAINING_CONFIG["logging_steps"],
                        help='Log training metrics every X steps')
    parser.add_argument('--save_steps', type=int, default=config.TRAINING_CONFIG["save_steps"],
                        help='Save checkpoint every X steps')
    
    # Deepspeed arguments
    parser.add_argument('--use_deepspeed', action='store_true', default=config.USE_DEEPSPEED,
                        help='Use DeepSpeed for training')
    parser.add_argument('--deepspeed_stage', type=int, default=config.DEEPSPEED_STAGE,
                        choices=[0, 2, 3], help='DeepSpeed ZeRO stage')
    parser.add_argument('--offload_optimizer', action='store_true', default=config.OFFLOAD_OPTIMIZER,
                        help='Offload optimizer states to CPU (for DeepSpeed ZeRO-2/3)')
    parser.add_argument('--offload_param', action='store_true', default=config.OFFLOAD_PARAM,
                        help='Offload parameters to CPU (for DeepSpeed ZeRO-3)')
    
    # Miscellaneous
    parser.add_argument('--bf16', action='store_true', default=config.USE_BF16,
                        help='Use bfloat16 precision')
    parser.add_argument('--fp16', action='store_true', default=config.USE_FP16,
                        help='Use float16 precision')
    parser.add_argument('--no_wandb', action='store_true', default=config.NO_WANDB,
                        help='Disable Weights & Biases logging')
    parser.add_argument('--resume_from_checkpoint', type=str, default=config.RESUME_FROM_CHECKPOINT,
                        help='Resume training from a checkpoint')
    parser.add_argument('--seed', type=int, default=config.SEED,
                        help='Random seed for initialization')
    
    return parser.parse_args()

def prepare_dataset(input_file: str, output_dir: str, val_size: float = 0.1, max_samples: Optional[int] = None) -> Dict[str, str]:

    logger.info(f"Preparing dataset from {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get('questions', [])
    logger.info(f"Loaded {len(questions)} QA pairs from input file")
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(questions):
        logger.info(f"Limiting to {max_samples} samples as specified")
        random.shuffle(questions)
        questions = questions[:max_samples]
    
    # Convert to LLaMA Factory format
    converted_data = []
    for item in questions:
        converted_item = {
            "instruction": item['question'],
            "input": "",  # No separate input in this format
            "output": item['response']
        }
        converted_data.append(converted_item)
    
    # Split into train and validation sets
    if val_size > 0:
        split_index = int(len(converted_data) * (1 - val_size))
        random.shuffle(converted_data)
        train_data = converted_data[:split_index]
        val_data = converted_data[split_index:]
        
        logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation samples")
        
        train_file = os.path.join(output_dir, "telugu_train.json")
        val_file = os.path.join(output_dir, "telugu_val.json")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        return {"train": train_file, "val": val_file}
    else:
        # No validation split
        dataset_file = os.path.join(output_dir, "telugu_dataset.json")
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(converted_data)} samples to {dataset_file}")
        return {"train": dataset_file}

def create_deepspeed_config(args, output_dir: str) -> Optional[str]:

    if not args.use_deepspeed:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get DeepSpeed configuration from config.py
    ds_config = copy.deepcopy(config.DEEPSPEED_CONFIG)
    
    # Update config with CLI arguments (which override config.py)
    ds_config["bf16"]["enabled"] = args.bf16
    ds_config["fp16"]["enabled"] = args.fp16
    ds_config["zero_optimization"]["stage"] = args.deepspeed_stage
    
    # Update offload settings based on CLI arguments
    if args.offload_optimizer and args.deepspeed_stage in [2, 3]:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    elif "offload_optimizer" in ds_config["zero_optimization"]:
        del ds_config["zero_optimization"]["offload_optimizer"]
    
    if args.offload_param and args.deepspeed_stage == 3:
        ds_config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    elif "offload_param" in ds_config["zero_optimization"]:
        del ds_config["zero_optimization"]["offload_param"]
    
    # Add ZeRO-3 specific configurations if needed
    if args.deepspeed_stage == 3 and not any(k in ds_config["zero_optimization"] for k in ["sub_group_size", "stage3_prefetch_bucket_size"]):
        ds_config["zero_optimization"].update({
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        })
    
    # Save configuration
    config_path = os.path.join(output_dir, f"ds_z{args.deepspeed_stage}_config.json")
    with open(config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    
    logger.info(f"Created DeepSpeed configuration at {config_path}")
    return config_path

def create_training_config(args, dataset_files, deepspeed_config: Optional[str] = None) -> str:

    logger.info("Creating training configuration")
    
    # Start with the base configuration from config.py
    training_config = copy.deepcopy(config.LLAMAFACTORY_TRAINING_CONFIG)
    
    # Update configuration with command-line arguments
    training_config.update({
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "finetuning_type": args.finetuning_type,
        "cutoff_len": args.cutoff_len,
        "max_samples": args.max_samples,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "bf16": args.bf16,
        "fp16": args.fp16,
    })
    
    # Define dataset based on validation split
    if "val" in dataset_files:
        training_config.update({
            "dataset": f"custom:{dataset_files['train']}",
            "eval_dataset": f"custom:{dataset_files['val']}",
            "val_size": 0  # Since we're explicitly providing a validation dataset
        })
    else:
        training_config.update({
            "dataset": f"custom:{dataset_files['train']}",
            "val_size": 0  # No validation
        })
    
    # Add LoRA configuration if using LoRA
    if args.finetuning_type in ["lora", "qlora"]:
        training_config.update({
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target": config.LORA_CONFIG["lora_target"]
        })
    
    # Add quantization configuration if using QLoRA
    if args.finetuning_type == "qlora":
        if args.quantization_bit is None:
            raise ValueError("quantization_bit must be specified if finetuning_type is qlora")
        
        training_config.update({
            "quantization_bit": args.quantization_bit,
            "quantization_method": config.QLORA_CONFIG["quantization_method"]
        })
    
    # Add DeepSpeed configuration if using DeepSpeed
    if deepspeed_config:
        training_config["deepspeed"] = deepspeed_config
    
    # Flash Attention
    if args.use_flash_attn:
        training_config["flash_attn"] = "fa2"
    
    # Weights & Biases logging
    if args.no_wandb and "report_to" in training_config:
        del training_config["report_to"]
        del training_config["run_name"]
    
    # Save configuration
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    config_path = os.path.join(os.path.dirname(args.output_dir), "train_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(training_config, f, default_flow_style=False)
    
    logger.info(f"Created training configuration at {config_path}")
    return config_path

def run_training(config_path: str, use_deepspeed: bool) -> None:

    logger.info("Starting training")
    
    command = ["llamafactory-cli", "train", config_path]
    
    if use_deepspeed:
        os.environ["FORCE_TORCHRUN"] = "1"
        logger.info("Using DeepSpeed for training")
    
    # Set CUDA_VISIBLE_DEVICES if specified in config
    if config.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={config.CUDA_VISIBLE_DEVICES}")
    
    logger.info(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)
    
    logger.info("Training completed")

def export_model(args) -> None:
    """
    Export the trained model and optionally push to Hugging Face Hub.
    
    Args:
        args: Arguments from argument parser
    """
    if args.finetuning_type in ["lora", "qlora"]:
        logger.info("Exporting (merging) LoRA model")
        
        # Use export configuration from config.py
        export_config = copy.deepcopy(config.EXPORT_CONFIG)
        
        # Update with the current arguments
        export_config["model_name_or_path"] = args.model_name_or_path
        export_config["adapter_name_or_path"] = args.output_dir
        export_config["export_dir"] = os.path.join(args.output_dir, "merged")
        
        # Save configuration
        export_path = os.path.join(os.path.dirname(args.output_dir), "export_config.yaml")
        with open(export_path, 'w') as f:
            yaml.dump(export_config, f, default_flow_style=False)
        
        # Run export
        subprocess.run(["llamafactory-cli", "export", export_path], check=True)
        logger.info(f"Model exported to {export_config['export_dir']}")
        
        # Push to Hugging Face Hub if configured
        if hasattr(config, 'HUGGINGFACE_REPO') and config.HUGGINGFACE_REPO:
            try:
                logger.info(f"Pushing model to Hugging Face Hub: {config.HUGGINGFACE_REPO}")
                
                # Set environment variable for Hugging Face API token if provided
                if hasattr(config, 'HUGGINGFACE_KEY') and config.HUGGINGFACE_KEY:
                    os.environ["HF_TOKEN"] = config.HUGGINGFACE_KEY
                    logger.info("Using Hugging Face API token from config")
                
                # Push the exported model to the Hub
                api = HfApi()
                model_path = export_config["export_dir"]
                
                # Create repository description
                model_card = f"""
                # Telugu Fine-tuned Gemma-3 Model
                
                This model is a fine-tuned version of `{args.model_name_or_path}` on Telugu question-answering data.
                
                ## Training Details
                
                - **Fine-tuning method:** {args.finetuning_type}
                - **Base model:** {args.model_name_or_path}
                - **Framework:** LLaMA Factory
                
                ## Usage
                
                ```python
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                tokenizer = AutoTokenizer.from_pretrained("{config.HUGGINGFACE_REPO}")
                model = AutoModelForCausalLM.from_pretrained("{config.HUGGINGFACE_REPO}")
                
                # Example usage
                input_text = "Your Telugu question here"
                inputs = tokenizer(input_text, return_tensors="pt")
                outputs = model.generate(**inputs, max_length=100)
                print(tokenizer.decode(outputs[0], skip_special_tokens=True))
                ```
                """
                
                # Push to Hub with commit message
                result = api.create_repo(
                    repo_id=config.HUGGINGFACE_REPO,
                    exist_ok=True,
                )
                
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=config.HUGGINGFACE_REPO,
                    commit_message=f"Upload fine-tuned {args.model_name_or_path} model with {args.finetuning_type}"
                )
                
                # Upload README
                with open(os.path.join(model_path, "README.md"), "w") as f:
                    f.write(model_card)
                
                api.upload_file(
                    path_or_fileobj=os.path.join(model_path, "README.md"),
                    path_in_repo="README.md",
                    repo_id=config.HUGGINGFACE_REPO,
                    commit_message="Update model card"
                )
                
                logger.info(f"Successfully pushed model to Hugging Face Hub: {config.HUGGINGFACE_REPO}")
                logger.info(f"View your model at: https://huggingface.co/{config.HUGGINGFACE_REPO}")
                
            except Exception as e:
                logger.error(f"Failed to push model to Hugging Face Hub: {str(e)}")
                logger.info("Continuing without pushing to Hugging Face Hub")

def main():
    # Parse command line arguments (which override config values)
    args = parse_arguments()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    logger.info("Starting Gemma-3-12B Telugu fine-tuning")
    logger.info(f"Using fine-tuning method: {args.finetuning_type}")
    
    # Check and validate arguments
    if args.finetuning_type == "qlora" and args.quantization_bit is None:
        logger.error("quantization_bit must be specified if finetuning_type is qlora")
        return
    
    if args.finetuning_type == "full" and args.use_deepspeed == False:
        logger.warning("Full fine-tuning without DeepSpeed may require a lot of GPU memory")
    
    # Prepare dataset
    dataset_dir = os.path.join(args.dataset_dir, "telugu")
    dataset_files = prepare_dataset(
        args.input_file, 
        dataset_dir, 
        val_size=args.val_size,
        max_samples=args.max_samples
    )
    
    # Create DeepSpeed configuration if needed
    deepspeed_config = None
    if args.use_deepspeed:
        deepspeed_dir = os.path.join(os.path.dirname(args.output_dir), "deepspeed")
        deepspeed_config = create_deepspeed_config(args, deepspeed_dir)
    
    # Create training configuration
    config_path = create_training_config(args, dataset_files, deepspeed_config)
    
    # Run training
    run_training(config_path, args.use_deepspeed)
    
    # Export model
    export_model(args)
    
    logger.info("Fine-tuning process completed")

if __name__ == "__main__":
    main()