#!/bin/bash

# Run script for Gemma-3-12B Telugu finetuning
# This script relies on the config.py file for configuration parameters

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Check if the finetune.py script exists
if [ ! -f "finetune.py" ]; then
    echo "finetune.py not found. Please make sure you are in the correct directory."
    exit 1
fi

# Check if the config.py file exists
if [ ! -f "config.py" ]; then
    echo "config.py not found. Please make sure the configuration file exists."
    exit 1
fi

# Check if llamafactory-cli is installed
if ! command -v llamafactory-cli &> /dev/null; then
    echo "llamafactory-cli not found. Please install LLaMA Factory."
    exit 1
fi

# Display configuration information
echo "======================================================================"
echo "                Gemma-3-12B Telugu Fine-tuning"
echo "======================================================================"
echo

# Extract key values from the config.py file using Python
MODEL_NAME=$(python3 -c "import config; print(config.MODEL_NAME_OR_PATH)")
FINETUNING_TYPE=$(python3 -c "import config; print(config.FINETUNING_TYPE)")
USE_DEEPSPEED=$(python3 -c "import config; print(config.USE_DEEPSPEED)")
DEEPSPEED_STAGE=$(python3 -c "import config; print(config.DEEPSPEED_STAGE)")
CUDA_VISIBLE_DEVICES_CONFIG=$(python3 -c "import config; print(config.CUDA_VISIBLE_DEVICES if config.CUDA_VISIBLE_DEVICES is not None else 'all available')")
BATCH_SIZE=$(python3 -c "import config; print(config.TRAINING_CONFIG['per_device_train_batch_size'])")
GRAD_ACCUM=$(python3 -c "import config; print(config.TRAINING_CONFIG['gradient_accumulation_steps'])")
EPOCHS=$(python3 -c "import config; print(config.TRAINING_CONFIG['num_train_epochs'])")
LR=$(python3 -c "import config; print(config.TRAINING_CONFIG['learning_rate'])")
INPUT_FILE=$(python3 -c "import config; print(config.INPUT_FILE)")
OUTPUT_DIR=$(python3 -c "import config; print(config.OUTPUT_DIR)")

# Display configuration summary
echo "Model: $MODEL_NAME"
echo "Fine-tuning method: $FINETUNING_TYPE"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo
echo "Training configuration:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation steps: $GRAD_ACCUM"
echo "  - Epochs: $EPOCHS"
echo "  - Learning rate: $LR"
echo
echo "Hardware configuration:"
echo "  - GPUs: $CUDA_VISIBLE_DEVICES_CONFIG"
echo "  - DeepSpeed: $USE_DEEPSPEED (Stage $DEEPSPEED_STAGE)"

# Display specific configuration based on fine-tuning type
if [ "$FINETUNING_TYPE" == "lora" ] || [ "$FINETUNING_TYPE" == "qlora" ]; then
    LORA_RANK=$(python3 -c "import config; print(config.LORA_CONFIG['lora_rank'])")
    LORA_ALPHA=$(python3 -c "import config; print(config.LORA_CONFIG['lora_alpha'])")
    echo
    echo "LoRA configuration:"
    echo "  - Rank: $LORA_RANK"
    echo "  - Alpha: $LORA_ALPHA"
fi

if [ "$FINETUNING_TYPE" == "qlora" ]; then
    QUANT_BIT=$(python3 -c "import config; print(config.QLORA_CONFIG['quantization_bit'])")
    echo "  - Quantization: $QUANT_BIT-bit"
fi

# Check if a previous run exists in the output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo
    echo "WARNING: Output directory already exists at $OUTPUT_DIR"
    echo "This might overwrite previous training results."
fi

echo
echo "======================================================================"



# Set CUDA_VISIBLE_DEVICES environment variable if specified in config
CUDA_VISIBLE_DEVICES_VALUE=$(python3 -c "import config; print(config.CUDA_VISIBLE_DEVICES if config.CUDA_VISIBLE_DEVICES is not None else '')")
if [ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
    echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi


# Handle Weights & Biases authentication
WANDB_ENABLED=$(python3 -c "import config; print(config.USE_WANDB)")
if [ "$WANDB_ENABLED" = "True" ]; then
    WANDB_API_KEY_VALUE=$(python3 -c "import config; print(config.WANDB_API_KEY if config.WANDB_API_KEY is not None else '')")
    
    if [ -n "$WANDB_API_KEY_VALUE" ]; then
        # Use API key from config
        export WANDB_API_KEY="$WANDB_API_KEY_VALUE"
        echo "Set WANDB_API_KEY from config"
    else
        # Check if WANDB_API_KEY is already set in environment
        if [ -z "$WANDB_API_KEY" ]; then
            echo "WARNING: Weights & Biases is enabled but no API key is provided."
            echo "Please set WANDB_API_KEY environment variable or provide it in config.py."
        else
            echo "Using WANDB_API_KEY from environment"
        fi
    fi
    
    # Set W&B project if defined
    WANDB_PROJECT_VALUE=$(python3 -c "import config; print(config.WANDB_PROJECT if hasattr(config, 'WANDB_PROJECT') else '')")
    if [ -n "$WANDB_PROJECT_VALUE" ]; then
        export WANDB_PROJECT="$WANDB_PROJECT_VALUE"
        echo "Set WANDB_PROJECT=$WANDB_PROJECT"
    fi
    
    echo "Weights & Biases logging is enabled"
fi

# Add this to handle Hugging Face authentication as well
HF_REPO_ENABLED=$(python3 -c "import config; print(hasattr(config, 'HUGGINGFACE_REPO') and bool(config.HUGGINGFACE_REPO))")
if [ "$HF_REPO_ENABLED" = "True" ]; then
    HF_API_KEY_VALUE=$(python3 -c "import config; print(config.HUGGINGFACE_KEY if hasattr(config, 'HUGGINGFACE_KEY') and config.HUGGINGFACE_KEY is not None else '')")
    
    if [ -n "$HF_API_KEY_VALUE" ]; then
        # Use API key from config
        export HF_TOKEN="$HF_API_KEY_VALUE"
        echo "Set HF_TOKEN from config"
    else
        # Check if HF_TOKEN is already set in environment
        if [ -z "$HF_TOKEN" ]; then
            echo "WARNING: Hugging Face Hub upload is enabled but no API key is provided."
            echo "Please set HF_TOKEN environment variable or provide HUGGINGFACE_KEY in config.py."
        else
            echo "Using HF_TOKEN from environment"
        fi
    fi
    
    echo "Hugging Face Hub integration is enabled"
fi





# Run the training script
echo "Starting training process at $(date)"
echo "======================================================================"
time python3 finetune.py
TRAINING_STATUS=$?

# Check if training completed successfully
if [ $TRAINING_STATUS -eq 0 ]; then
    echo "======================================================================"
    echo "Training completed successfully at $(date)!"
    
    # Get output directory path
    OUTPUT_DIR=$(python3 -c "import config; print(config.OUTPUT_DIR)")
    echo "Model saved at: $OUTPUT_DIR"
    
    # Check if merged model exists (for LoRA/QLoRA)
    if [ "$FINETUNING_TYPE" == "lora" ] || [ "$FINETUNING_TYPE" == "qlora" ]; then
        MERGED_DIR="$OUTPUT_DIR/merged"
        if [ -d "$MERGED_DIR" ]; then
            echo "Merged model saved at: $MERGED_DIR"
        fi
    fi
else
    echo "======================================================================"
    echo "Training failed with status code $TRAINING_STATUS. Please check the logs for details."
    exit 1
fi

# Print exit message
echo
echo "You can use the trained model with:"
echo "  llamafactory-cli chat --model_name_or_path $OUTPUT_DIR"
echo
if [ "$FINETUNING_TYPE" == "lora" ] || [ "$FINETUNING_TYPE" == "qlora" ]; then
    if [ -d "$OUTPUT_DIR/merged" ]; then
        echo "Or use the merged model with:"
        echo "  llamafactory-cli chat --model_name_or_path $OUTPUT_DIR/merged"
    fi
fi
echo "======================================================================"