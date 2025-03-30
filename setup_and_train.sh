#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# ANSI color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}   Setting up environment for Gemma-3-12B fine-tuning    ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}This script may need sudo privileges for system-level installations.${NC}"
  echo -e "${YELLOW}If it fails, consider running with sudo.${NC}"
fi

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check and install packages
check_and_install() {
  if ! command_exists $1; then
    echo -e "${YELLOW}$1 is not installed. Installing...${NC}"
    if command_exists apt-get; then
      sudo apt-get update && sudo apt-get install -y $2
    elif command_exists yum; then
      sudo yum install -y $2
    else
      echo -e "${RED}Unsupported package manager. Please install $1 manually.${NC}"
      exit 1
    fi
  else
    echo -e "${GREEN}$1 is already installed.${NC}"
  fi
}

# Check and install system dependencies
echo -e "${BLUE}Checking system dependencies...${NC}"
check_and_install python3 python3
check_and_install pip3 python3-pip
check_and_install git git
check_and_install wget wget
check_and_install nvidia-smi nvidia-driver

# Check CUDA availability
echo -e "${BLUE}Checking CUDA availability...${NC}"
if command_exists nvidia-smi; then
  echo -e "${GREEN}NVIDIA GPU detected.${NC}"
  nvidia-smi
else
  echo -e "${RED}NVIDIA GPU not detected. This script requires NVIDIA GPU with CUDA support.${NC}"
  exit 1
fi

# Create a Python virtual environment
echo -e "${BLUE}Creating Python virtual environment...${NC}"
VENV_DIR="gemma_env"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv $VENV_DIR
  echo -e "${GREEN}Virtual environment created at $VENV_DIR${NC}"
else
  echo -e "${GREEN}Virtual environment already exists at $VENV_DIR${NC}"
fi

# Activate the virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source $VENV_DIR/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch with CUDA support
echo -e "${BLUE}Installing PyTorch with CUDA support...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
echo -e "${BLUE}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Login to Hugging Face
echo -e "${BLUE}Setting up Hugging Face credentials...${NC}"
echo -e "${YELLOW}Please enter your Hugging Face token (or press Enter to skip):${NC}"
read HF_TOKEN

if [ ! -z "$HF_TOKEN" ]; then
  echo -e "${GREEN}Setting up Hugging Face token...${NC}"
  mkdir -p ~/.huggingface
  echo $HF_TOKEN > ~/.huggingface/token
else
  echo -e "${YELLOW}Skipping Hugging Face authentication.${NC}"
fi

# Login to Weights & Biases
echo -e "${BLUE}Setting up Weights & Biases...${NC}"
echo -e "${YELLOW}Do you want to log in to Weights & Biases? (y/n)${NC}"
read WANDB_LOGIN

if [ "$WANDB_LOGIN" = "y" ]; then
  wandb login
else
  echo -e "${YELLOW}Skipping Weights & Biases login.${NC}"
fi

# Create DeepSpeed configuration
echo -e "${BLUE}Creating DeepSpeed configuration...${NC}"
cat > ds_config.json << 'EOL'
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-6,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 2,
  "wall_clock_breakdown": false
}
EOL
echo -e "${GREEN}DeepSpeed configuration created.${NC}"

# Configure training parameters
echo -e "${BLUE}Configuring training parameters...${NC}"
echo -e "${YELLOW}Please select your GPU: ${NC}"
echo "1) NVIDIA A100 (80GB)"
echo "2) NVIDIA A100 (40GB) / A6000 (48GB)"
echo "3) NVIDIA A10 (24GB) or other"
read GPU_OPTION

case $GPU_OPTION in
  1)
    BATCH_SIZE=4
    GRAD_ACCUM=8
    USE_DEEPSPEED="n"
    ;;
  2)
    BATCH_SIZE=2
    GRAD_ACCUM=16
    USE_DEEPSPEED="n"
    ;;
  3)
    BATCH_SIZE=1
    GRAD_ACCUM=32
    echo -e "${YELLOW}Do you want to use DeepSpeed for memory optimization? (y/n)${NC}"
    read USE_DEEPSPEED
    ;;
  *)
    echo -e "${RED}Invalid option. Using conservative settings.${NC}"
    BATCH_SIZE=1
    GRAD_ACCUM=32
    USE_DEEPSPEED="y"
    ;;
esac

# Configure output directory
echo -e "${YELLOW}Enter output directory name [default: finetuned_gemma]:${NC}"
read OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-finetuned_gemma}

# Configure WANDB project
echo -e "${YELLOW}Enter Weights & Biases project name [default: telugu-gemma-sft]:${NC}"
read WANDB_PROJECT
WANDB_PROJECT=${WANDB_PROJECT:-telugu-gemma-sft}

# Configure HF Hub
echo -e "${YELLOW}Do you want to push to Hugging Face Hub? (y/n) [default: n]:${NC}"
read PUSH_TO_HUB
PUSH_TO_HUB=${PUSH_TO_HUB:-n}

HUB_MODEL_ID=""
if [ "$PUSH_TO_HUB" = "y" ]; then
  echo -e "${YELLOW}Enter Hugging Face Hub model ID (username/model-name):${NC}"
  read HUB_MODEL_ID
fi

# Prepare run command
RUN_CMD="python finetune_llm.py \
--model_name google/gemma-3-12b-pt \
--dataset_path telugu_results.json \
--output_dir $OUTPUT_DIR \
--num_train_epochs 3 \
--per_device_train_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $GRAD_ACCUM \
--wandb_project $WANDB_PROJECT"

if [ "$USE_DEEPSPEED" = "y" ]; then
  RUN_CMD="$RUN_CMD --deepspeed_config ds_config.json"
fi

if [ "$PUSH_TO_HUB" = "y" ] && [ ! -z "$HUB_MODEL_ID" ]; then
  RUN_CMD="$RUN_CMD --push_to_hub True --hub_model_id $HUB_MODEL_ID"
fi

# Create a run script
echo -e "${BLUE}Creating run script...${NC}"
cat > run_training.sh << EOL
#!/bin/bash
source $VENV_DIR/bin/activate
$RUN_CMD
EOL

chmod +x run_training.sh
echo -e "${GREEN}Run script created as run_training.sh${NC}"

# Create a tmux script for running in background
echo -e "${BLUE}Creating tmux script for background running...${NC}"
cat > run_in_tmux.sh << EOL
#!/bin/bash
SESSION_NAME="gemma-training"
tmux new-session -d -s \$SESSION_NAME
tmux send-keys -t \$SESSION_NAME "source $VENV_DIR/bin/activate" C-m
tmux send-keys -t \$SESSION_NAME "$RUN_CMD" C-m
echo "Training started in tmux session '\$SESSION_NAME'"
echo "To attach to the session, run: tmux attach-session -t \$SESSION_NAME"
echo "To detach from the session (once attached), press: Ctrl+B, then D"
EOL

chmod +x run_in_tmux.sh
echo -e "${GREEN}Tmux script created as run_in_tmux.sh${NC}"

# Final instructions
echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${BLUE}To start training, you have two options:${NC}"
echo -e "${YELLOW}1. Run directly: ./run_training.sh${NC}"
echo -e "${YELLOW}2. Run in background with tmux: ./run_in_tmux.sh${NC}"
echo -e "${BLUE}Using tmux is recommended for long training sessions.${NC}"
echo -e "${BLUE}It allows the training to continue even if you disconnect.${NC}"
echo -e "${BLUE}=========================================================${NC}" 