import modal
import os
import sys
import logging
import importlib.util
from pathlib import Path
from types import SimpleNamespace

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Define CUDA tag
tag = "12.4.0-devel-ubuntu20.04"
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
    # First install PyTorch with specific version known to work with DeepSpeed
    .run_commands("pip install torch torchvision torchaudio packaging ninja wheel importlib-metadata pydantic transformers accelerate peft datasets deepspeed trl bitsandbytes wandb huggingface_hub pyyaml rouge-score sacrebleu ")
    # .run_commands("export DISABLE_VERSION_CHECK=1")
    .run_commands("pip install --upgrade git+https://github.com/huggingface/transformers.git")
    .run_commands("pip install --upgrade git+https://github.com/hiyouga/LLaMA-Factory.git")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    # Add HF_TOKEN and WANDB_API_KEY
    .env({"HF_TOKEN": "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"})
    .env({"WANDB_API_KEY": "bc746fafa585f61730cdfd3c8226492c777f875a"})
    .add_local_dir("C:/Users/bhara/OneDrive/Documents/hindi_LLM/Gemini_dataset/training_files/llama_factory", remote_path="/training", copy=True)
)

# Create volumes for caching models and storing training artifacts
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
training_vol = modal.Volume.from_name("gemma-telugu-training", create_if_missing=True)
output_vol = modal.Volume.from_name("gemma-telugu-output-vol", create_if_missing=True)

# Create the Modal app
app = modal.App("gemma-telugu-finetuning", image=image)

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



# Request 8 A100 80GB GPUs
# Request 6 H100 GPUs
@app.function(
    gpu="H100:8",  
    timeout=60*60*24,   # 24-hour timeout for long training
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/checkpoints": output_vol  
    },
    memory=128384, # Increased memory
)
def run_finetune():
    """Runs the finetune.py functionality directly without using the shell script."""
    base_training_dir = Path("/training")
    
    logger.info(f"Changing working directory to {base_training_dir}")
    try:
        os.chdir(base_training_dir)
    except FileNotFoundError:
        logger.error(f"Error: Could not change directory to {base_training_dir}. Check volume mount.")
        return {"status": "error", "message": f"Failed to change directory to {base_training_dir}"}
    
    # Set environment variables for distributed training
    os.environ["FORCE_TORCHRUN"] = "1"
    logger.info("Set FORCE_TORCHRUN=1")
    
    try:
        # Import the config module
        logger.info("Importing config and finetune modules...")
        config_module = import_module_from_path("config", base_training_dir / "config.py")
        finetune_module = import_module_from_path("finetune", base_training_dir / "finetune.py")
        
        # Set up Weights & Biases logging if enabled
        if hasattr(config_module, 'USE_WANDB') and config_module.USE_WANDB:
            if hasattr(config_module, 'WANDB_API_KEY') and config_module.WANDB_API_KEY:
                os.environ["WANDB_API_KEY"] = config_module.WANDB_API_KEY
                logger.info("Set WANDB_API_KEY from config")
            
            if hasattr(config_module, 'WANDB_PROJECT') and config_module.WANDB_PROJECT:
                os.environ["WANDB_PROJECT"] = config_module.WANDB_PROJECT
                logger.info(f"Set WANDB_PROJECT={config_module.WANDB_PROJECT}")
            
            logger.info("Weights & Biases logging is enabled")
        
        # Set up Hugging Face authentication if enabled
        if hasattr(config_module, 'HUGGINGFACE_REPO') and config_module.HUGGINGFACE_REPO:
            if hasattr(config_module, 'HUGGINGFACE_KEY') and config_module.HUGGINGFACE_KEY:
                os.environ["HF_TOKEN"] = config_module.HUGGINGFACE_KEY
                logger.info("Set HF_TOKEN from config")
            
            logger.info("Hugging Face Hub integration is enabled")
        
        # Log key configuration settings
        logger.info(f"Model: {config_module.MODEL_NAME_OR_PATH}")
        logger.info(f"Fine-tuning method: {config_module.FINETUNING_TYPE}")
        logger.info(f"Output directory: {config_module.OUTPUT_DIR}")
        logger.info(f"Batch size: {config_module.TRAINING_CONFIG['per_device_train_batch_size']}")
        logger.info(f"Gradient accumulation steps: {config_module.TRAINING_CONFIG['gradient_accumulation_steps']}")
        logger.info(f"Learning rate: {config_module.TRAINING_CONFIG['learning_rate']}")
        logger.info(f"Epochs: {config_module.TRAINING_CONFIG['num_train_epochs']}")
        
        # Create args object with properties matching config.py settings
        # This simulates the argparse result without using the parse_args() method
        args = SimpleNamespace(
            input_file=config_module.INPUT_FILE,
            output_dir=config_module.OUTPUT_DIR,
            dataset_dir=config_module.DATASET_DIR,
            max_samples=config_module.MAX_SAMPLES,
            val_size=config_module.VAL_SIZE,
            model_name_or_path=config_module.MODEL_NAME_OR_PATH,
            use_flash_attn=config_module.USE_FLASH_ATTN,
            finetuning_type=config_module.FINETUNING_TYPE,
            lora_rank=config_module.LORA_CONFIG["lora_rank"],
            lora_alpha=config_module.LORA_CONFIG["lora_alpha"],
            lora_dropout=config_module.LORA_CONFIG["lora_dropout"],
            quantization_bit=config_module.QLORA_CONFIG["quantization_bit"] if config_module.FINETUNING_TYPE == "qlora" else None,
            batch_size=config_module.TRAINING_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=config_module.TRAINING_CONFIG["gradient_accumulation_steps"],
            num_epochs=config_module.TRAINING_CONFIG["num_train_epochs"],
            learning_rate=config_module.TRAINING_CONFIG["learning_rate"],
            cutoff_len=config_module.TRAINING_CONFIG["cutoff_len"],
            warmup_ratio=config_module.TRAINING_CONFIG["warmup_ratio"],
            logging_steps=config_module.TRAINING_CONFIG["logging_steps"],
            save_steps=config_module.TRAINING_CONFIG["save_steps"],
            use_deepspeed=config_module.USE_DEEPSPEED,
            deepspeed_stage=config_module.DEEPSPEED_STAGE,
            offload_optimizer=config_module.OFFLOAD_OPTIMIZER,
            offload_param=config_module.OFFLOAD_PARAM,
            bf16=config_module.USE_BF16,
            fp16=config_module.USE_FP16,
            no_wandb=config_module.NO_WANDB,
            resume_from_checkpoint=config_module.RESUME_FROM_CHECKPOINT,
            seed=config_module.SEED,
        )
        
        # Run the individual functions from finetune.py with our args object
        logger.info("Starting the finetuning process...")
        
        # Set random seed for reproducibility
        import random
        random.seed(args.seed)
        
        # Execute the steps in finetune.py's main() function directly
        # 1. Prepare dataset
        logger.info("Preparing dataset...")
        dataset_dir = os.path.join(args.dataset_dir, "telugu")
        dataset_files = finetune_module.prepare_dataset(
            args.input_file, 
            dataset_dir, 
            val_size=args.val_size,
            max_samples=args.max_samples
        )
        
        # 2. Create DeepSpeed configuration if needed
        deepspeed_config = None
        if args.use_deepspeed:
            logger.info("Creating DeepSpeed configuration...")
            deepspeed_dir = os.path.join(os.path.dirname(args.output_dir), "deepspeed")
            deepspeed_config = finetune_module.create_deepspeed_config(args, deepspeed_dir)
        
        # 3. Create training configuration
        logger.info("Creating training configuration...")
        config_path = finetune_module.create_training_config(args, dataset_files, deepspeed_config)
        
        # 4. Run training
        logger.info("Running training...")
        finetune_module.run_training(config_path, args.use_deepspeed)
        
        # 5. Export model if needed
        if args.finetuning_type in ["lora", "qlora"]:
            logger.info("Exporting model...")
            finetune_module.export_model(args)
        
        # Define expected output directory
        output_dir = base_training_dir / args.output_dir
        logger.info(f"Fine-tuning completed successfully! Model saved to {output_dir}")
        
        return {
            "status": "success", 
            "output_dir": str(output_dir),
            "model_name": args.model_name_or_path,
            "finetuning_type": args.finetuning_type
        }
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Fine-tuning failed: {str(e)}"}

@app.local_entrypoint()
def main():
    """Main entry point for the Modal application."""
    logger.info("Starting remote fine-tuning function...")
    result = run_finetune.remote()

    logger.info("\n--- Fine-tuning Result ---")
    logger.info(f"Status: {result.get('status', 'Unknown')}")
    
    if result.get('status') == "success":
        logger.info(f"Fine-tuning completed successfully!")
        output_dir = result.get('output_dir', 'N/A')
        model_name = result.get('model_name', 'N/A')
        finetuning_type = result.get('finetuning_type', 'N/A')
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Fine-tuning type: {finetuning_type}")
        logger.info(f"Output directory: {output_dir}")
        
        if finetuning_type in ["lora", "qlora"]:
            merged_dir = f"{output_dir}/merged"
            logger.info(f"For merged model (if available): {merged_dir}")
    else:
        error_message = result.get('message', 'No error message provided.')
        logger.error(f"Fine-tuning failed: {error_message}")

    logger.info("--- Run Finished ---")