#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete configuration for Gemma-3 Telugu finetuning

This file contains ALL configurable parameters for the finetuning process.
Modify these values instead of changing the main script.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Path to input JSON file containing Telugu QA pairs
INPUT_FILE = "telugu_final.json"

# Directory to save model and checkpoints
OUTPUT_DIR = "./gemma3-telugu-output"

# Directory to save processed dataset
DATASET_DIR = "./data"

# Maximum number of samples to use for training (None to use all)
MAX_SAMPLES = None

# Size of validation split (0 to 1)
VAL_SIZE = 0.1


# ============================================================================
# HUGGING FACE HUB CONFIGURATION
# ============================================================================

# Hugging Face Hub repository to push the model to (username/repo-name)
HUGGINGFACE_REPO = "bharathkumar1922001/gemma-3-12b-pt-telugu-sft"

# Hugging Face API key for accessing gated models and pushing to Hub
# You can either set it here or use environment variables for better security
HUGGINGFACE_KEY = "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"  
# Set to your API key or leave as None to use HF_TOKEN environment variable

# ============================================================================
# WEIGHTS & BIASES CONFIGURATION
# ============================================================================

# Whether to use Weights & Biases for logging
USE_WANDB = True

# Weights & Biases project name
WANDB_PROJECT = "gemma-telugu"  

# Weights & Biases API key
# You can either set it here or use environment variables for better security
WANDB_API_KEY = "bc746fafa585f61730cdfd3c8226492c777f875a"  
# Set to your API key or leave as None to use WANDB_API_KEY environment variable


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Path to pre-trained model or Hugging Face model identifier
MODEL_NAME_OR_PATH = "google/gemma-3-12b-pt"

# Template to use for formatting prompts
MODEL_TEMPLATE = "gemma3"

# Whether to trust remote code
TRUST_REMOTE_CODE = True

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

# GPUs to use (None to use all available GPUs)
# Example: "0,1,2,3" to use first 4 GPUs
CUDA_VISIBLE_DEVICES = None

# Use Flash Attention 2 for faster training
USE_FLASH_ATTN = True

# Precision settings (choose one)
USE_BF16 = True
USE_FP16 = False

# ============================================================================
# TRAINING METHOD CONFIGURATION
# ============================================================================

# Fine-tuning method: "lora", "qlora", or "full"
FINETUNING_TYPE = "full"

# LoRA configuration (used if FINETUNING_TYPE is "lora" or "qlora")
LORA_CONFIG = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target": "all"  # Target modules for LoRA
}

# QLoRA configuration (used if FINETUNING_TYPE is "qlora")
QLORA_CONFIG = {
    "quantization_bit": 4,  # 4 or 8
    "quantization_method": "bitsandbytes"  # currently only bitsandbytes is supported
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

TRAINING_CONFIG = {
    # Batch sizes and steps
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 32,
    
    # Optimization parameters
    "learning_rate": 2e-5,
    "num_train_epochs": 3.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    
    # Sequence length
    "cutoff_len": 4096,
    
    # Preprocessing
    "preprocessing_num_workers": 16,
    "overwrite_cache": True,
    
    # Logging and checkpointing
    "logging_steps": 50,
    "save_steps": 100,
    "eval_steps": 100,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "save_safetensors": True,
    
    # Additional flags (not using currently)
    "ddp_timeout": 180000000
}

# ============================================================================
# DEEPSPEED CONFIGURATION
# ============================================================================

# Use DeepSpeed for training
USE_DEEPSPEED = True

# DeepSpeed ZeRO stage (0, 2, or 3)
DEEPSPEED_STAGE = 2

# Offload optimizer states to CPU (for DeepSpeed ZeRO-2/3)
OFFLOAD_OPTIMIZER = True

# Offload parameters to CPU (for DeepSpeed ZeRO-3)
OFFLOAD_PARAM = True if DEEPSPEED_STAGE == 3 else False

# Advanced DeepSpeed Configuration
DEEPSPEED_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": True,
    
    # Precision settings will be populated from USE_BF16 and USE_FP16
    "bf16": {
        "enabled": USE_BF16
    },
    "fp16": {
        "enabled": USE_FP16,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    # ZeRO optimization settings
    "zero_optimization": {
        "stage": DEEPSPEED_STAGE,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "round_robin_gradients": True
    }
}

# Add offload options if enabled
if OFFLOAD_OPTIMIZER and DEEPSPEED_STAGE in [2, 3]:
    DEEPSPEED_CONFIG["zero_optimization"]["offload_optimizer"] = {
        "device": "cpu",
        "pin_memory": True
    }

if OFFLOAD_PARAM and DEEPSPEED_STAGE == 3:
    DEEPSPEED_CONFIG["zero_optimization"]["offload_param"] = {
        "device": "cpu",
        "pin_memory": True
    }

# For ZeRO-3, add additional configuration
if DEEPSPEED_STAGE == 3:
    DEEPSPEED_CONFIG["zero_optimization"].update({
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    })

# ============================================================================
# LLAMAFACTORY TRAINING CONFIGURATION
# ============================================================================

# This is the configuration used by LLaMA Factory for training
# It will be automatically populated from the settings above
LLAMAFACTORY_TRAINING_CONFIG = {
    # Model configuration
    "model_name_or_path": MODEL_NAME_OR_PATH,
    "trust_remote_code": TRUST_REMOTE_CODE,
    
    # Method configuration
    "stage": "sft",
    "do_train": True,
    "finetuning_type": FINETUNING_TYPE,
    
    # Dataset configuration - will be filled at runtime
    # "dataset": "custom:{dataset_file}",
    # "eval_dataset": "custom:{val_dataset_file}",  # Only if validation split is used
    "template": MODEL_TEMPLATE,
    "cutoff_len": TRAINING_CONFIG["cutoff_len"],
    "max_samples": MAX_SAMPLES,
    "overwrite_cache": TRAINING_CONFIG["overwrite_cache"],
    "preprocessing_num_workers": TRAINING_CONFIG["preprocessing_num_workers"],
    
    # Output configuration
    "output_dir": OUTPUT_DIR,
    "logging_steps": TRAINING_CONFIG["logging_steps"],
    "save_steps": TRAINING_CONFIG["save_steps"],
    "eval_steps": TRAINING_CONFIG["eval_steps"],
    "plot_loss": TRAINING_CONFIG["plot_loss"],
    "overwrite_output_dir": TRAINING_CONFIG["overwrite_output_dir"],
    "save_safetensors": TRAINING_CONFIG["save_safetensors"],
    
    # Training configuration
    "per_device_train_batch_size": TRAINING_CONFIG["per_device_train_batch_size"],
    "gradient_accumulation_steps": TRAINING_CONFIG["gradient_accumulation_steps"],
    "learning_rate": TRAINING_CONFIG["learning_rate"],
    "num_train_epochs": TRAINING_CONFIG["num_train_epochs"],
    "lr_scheduler_type": TRAINING_CONFIG["lr_scheduler_type"],
    "warmup_ratio": TRAINING_CONFIG["warmup_ratio"],
    "ddp_timeout": TRAINING_CONFIG["ddp_timeout"],
    
    # Floating point precision
    "bf16": USE_BF16,
    "fp16": USE_FP16,
    
    # Additional features
    "resume_from_checkpoint": None  # Will be set via CLI if needed
}

# Add LoRA configuration if using LoRA
if FINETUNING_TYPE in ["lora", "qlora"]:
    LLAMAFACTORY_TRAINING_CONFIG.update({
        "lora_rank": LORA_CONFIG["lora_rank"],
        "lora_alpha": LORA_CONFIG["lora_alpha"],
        "lora_dropout": LORA_CONFIG["lora_dropout"],
        "lora_target": LORA_CONFIG["lora_target"]
    })

# Add quantization configuration if using QLoRA
if FINETUNING_TYPE == "qlora":
    LLAMAFACTORY_TRAINING_CONFIG.update({
        "quantization_bit": QLORA_CONFIG["quantization_bit"],
        "quantization_method": QLORA_CONFIG["quantization_method"]
    })

# Flash Attention
if USE_FLASH_ATTN:
    LLAMAFACTORY_TRAINING_CONFIG["flash_attn"] = "fa2"

# ============================================================================
# WEIGHT & BIASES CONFIGURATION
# ============================================================================

# Disable Weights & Biases logging
NO_WANDB = False

# If Weights & Biases is enabled, set up the configuration
if not NO_WANDB:
    LLAMAFACTORY_TRAINING_CONFIG.update({
        "report_to": "wandb",
        "run_name": f"gemma-3-12b-telugu-{FINETUNING_TYPE}"
    })

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

# Export configuration for merging LoRA adapters with base model
EXPORT_CONFIG = {
    # Will be populated during runtime
    "model_name_or_path": HUGGINGFACE_REPO,
    "adapter_name_or_path": None,  # Will be set to OUTPUT_DIR during export
    "template": MODEL_TEMPLATE,
    "trust_remote_code": TRUST_REMOTE_CODE,
    "export_dir": None,  # Will be set to OUTPUT_DIR/merged during export
    "export_size": 2,  # Number of largest shards to save
    "export_device": "cpu",
    "export_legacy_format": False
}

# ============================================================================
# MISCELLANEOUS
# ============================================================================

# Random seed for reproducibility
SEED = 42

# Resume training from a checkpoint
RESUME_FROM_CHECKPOINT = None