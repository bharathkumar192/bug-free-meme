#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# ANSI color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}     Setting up environment for Gemma-3 Telugu fine-tuning    ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check CUDA availability and GPU memory
check_gpu() {
  if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected.${NC}"
    nvidia-smi
    
    # Check available memory
    FREE_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    TOTAL_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    
    echo -e "${GREEN}GPU memory: ${FREE_MEMORY}MB free / ${TOTAL_MEMORY}MB total${NC}"
    
    if [ "$TOTAL_MEMORY" -lt 48000 ]; then
      echo -e "${YELLOW}Warning: Less than 48GB GPU memory available for full fine-tuning of Gemma-3-12b.${NC}"
      echo -e "${YELLOW}You might want to consider enabling PEFT/LoRA or quantization in your config.${NC}"
      # echo -e "${YELLOW}Continue? (y/n)${NC}"
      # read -r continue_choice
      # if [[ "$continue_choice" != "y" && "$continue_choice" != "Y" ]]; then
      #   echo -e "${RED}Setup aborted.${NC}"
      #   exit 1
      # fi
    elif [ "$TOTAL_MEMORY" -lt 70000 ]; then
      echo -e "${YELLOW}You have sufficient memory for fine-tuning but may need to adjust batch sizes${NC}"
    else
      echo -e "${GREEN}Excellent! You have 70GB+ GPU memory, which is ideal for full fine-tuning${NC}"
    fi
  else
    echo -e "${RED}No NVIDIA GPU detected. Cannot proceed with fine-tuning.${NC}"
    exit 1
  fi
}

# Install dependencies
install_dependencies() {
  echo -e "${BLUE}Installing required packages...${NC}"
  
  # Check if requirements.txt exists, otherwise install necessary packages
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install transformers datasets accelerate wandb tensorboard peft
  pip install bitsandbytes safetensors pandas pyyaml tqdm huggingface-hub
  pip install unsloth
  
  # Install latest Hugging Face for Gemma-3
  pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
  
  echo -e "${GREEN}Dependencies installed successfully.${NC}"
}

# Load config and prepare environment
prepare_environment() {
  # Check if config.yaml exists
  if [ ! -f "config.yaml" ]; then
    echo -e "${RED}config.yaml not found. Please ensure you have a valid configuration file.${NC}"
    exit 1
  fi
  
  # Create output directory from config
  OUTPUT_DIR=$(grep "output_dir:" config.yaml | head -1 | awk '{print $2}' | tr -d '"')
  mkdir -p "$OUTPUT_DIR"
  mkdir -p "${OUTPUT_DIR}/logs"
  mkdir -p "${OUTPUT_DIR}/tensorboard"
  
  echo -e "${GREEN}Environment prepared successfully.${NC}"
}

# Main execution
main() {
  echo -e "${BLUE}Checking GPU...${NC}"
  check_gpu
  
  echo -e "${BLUE}Installing dependencies...${NC}"
  install_dependencies
  
  echo -e "${BLUE}Preparing environment...${NC}"
  prepare_environment
  
  echo -e "${BLUE}Starting fine-tuning...${NC}"
  echo -e "${GREEN}Command: python finetune_gemma3_telugu.py${NC}"
  
  # Export large tensor threshold to avoid warnings
  export TORCH_SHOW_CPP_STACKTRACES=1
  export NCCL_DEBUG=INFO
  
  # Enable flash attention if available
  export TRANSFORMERS_ATTN_IMPLEMENTATION="flash_attention_2"
  
  # Memory optimization
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

  apt-get update && apt-get install -y git-lfs
  git lfs install
  git lfs pull
  
  # Start training
  # python finetune_gemma3_telugu.py
}

# Execute main function
main