#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# ANSI color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}     Setting up environment for LLM fine-tuning          ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python_version() {
  local required_version="3.8.0"
  local current_version=$(python3 --version 2>&1 | awk '{print $2}')
  
  if ! command_exists python3; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
  fi
  
  # Compare versions
  if python3 -c "import sys; from packaging import version; sys.exit(0 if version.parse('$current_version') >= version.parse('$required_version') else 1)" 2>/dev/null; then
    echo -e "${GREEN}Python version $current_version is compatible.${NC}"
  else
    echo -e "${RED}Python version $current_version is not compatible. Please install Python 3.8 or higher.${NC}"
    exit 1
  fi
}

# Check CUDA availability
check_cuda() {
  if command_exists nvidia-smi; then
    echo -e "${GREEN}NVIDIA GPU detected.${NC}"
    nvidia-smi
    
    # Check CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo -e "${GREEN}CUDA Driver Version: $CUDA_VERSION${NC}"
    
    # Check available memory
    FREE_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    echo -e "${GREEN}Available GPU memory: $FREE_MEMORY MB${NC}"
    
    if [ "$FREE_MEMORY" -lt 8000 ]; then
      echo -e "${YELLOW}Warning: Less than 8GB GPU memory available. Consider enabling 4-bit quantization and LoRA in your config.${NC}"
    fi
  else
    echo -e "${YELLOW}NVIDIA GPU not detected. Training may be very slow on CPU.${NC}"
  fi
}

# Create and activate virtual environment
setup_virtualenv() {
  if ! command_exists pip3; then
    echo -e "${RED}pip3 is not installed. Please install pip for Python 3.${NC}"
    exit 1
  fi
  
  # Try to install virtualenv if not available
  if ! command_exists virtualenv; then
    echo -e "${YELLOW}virtualenv not found. Installing...${NC}"
    pip3 install virtualenv
  fi
  
  # Create virtual environment if it doesn't exist
  VENV_DIR="llm_env"
  if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Creating virtual environment at $VENV_DIR...${NC}"
    virtualenv "$VENV_DIR"
    source llm_env/bin/activate
  else
    echo -e "${GREEN}Using existing virtual environment at $VENV_DIR${NC}"
    source llm_env/bin/activate
  fi
  
  # Activate virtual environment
  echo -e "${BLUE}Activating virtual environment...${NC}"
  source "$VENV_DIR/bin/activate"
  
  # Upgrade pip
  echo -e "${BLUE}Upgrading pip...${NC}"
  pip install --upgrade pip
}

# Install dependencies
install_dependencies() {
  echo -e "${BLUE}Installing required packages...${NC}"
  
  # Check if requirements.txt exists
  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
  else
    # Install common dependencies if requirements.txt is not available
    echo -e "${YELLOW}requirements.txt not found. Installing common dependencies...${NC}"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets accelerate wandb tensorboard peft bitsandbytes \
                safetensors pandas scikit-learn pyyaml tqdm huggingface-hub loguru \
                packaging
  fi
  
  echo -e "${GREEN}Dependencies installed successfully.${NC}"
}

# Check if config file exists
check_config() {
  if [ ! -f "config.yaml" ]; then
    echo -e "${RED}config.yaml not found. Please ensure you have a valid configuration file.${NC}"
    exit 1
  else
    echo -e "${GREEN}Configuration file found: config.yaml${NC}"
  fi
}

# Check for required Python files
check_required_files() {
  REQUIRED_FILES=("config.py" "prepare_data.py" "finetune_llm.py")
  MISSING_FILES=()
  
  for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
      MISSING_FILES+=("$file")
    fi
  done
  
  if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${RED}Some required files are missing:${NC}"
    for file in "${MISSING_FILES[@]}"; do
      echo -e "${RED}- $file${NC}"
    done
    exit 1
  else
    echo -e "${GREEN}All required Python files are present.${NC}"
  fi
}

# Setup Weights & Biases
setup_wandb() {
  # Check if wandb is in the report_to list in config.yaml
  if grep -q "report_to:.*wandb" config.yaml; then
    if ! grep -q "wandb_api_key:" config.yaml || grep -q "wandb_api_key: null" config.yaml; then
      echo -e "${YELLOW}W&B API key not found in config.yaml. Do you want to log in to Weights & Biases? (y/n)${NC}"
      read -r wandb_login_choice
      
      if [[ "$wandb_login_choice" == "y" || "$wandb_login_choice" == "Y" ]]; then
        echo -e "${BLUE}Please enter your Weights & Biases API key (or press Enter to run the login flow):${NC}"
        read -r wandb_key
        
        if [ -n "$wandb_key" ]; then
          # Update config.yaml with the provided API key
          sed -i "s/wandb_api_key: null/wandb_api_key: \"$wandb_key\"/" config.yaml
          echo -e "${GREEN}W&B API key added to config.yaml${NC}"
        else
          # Run wandb login flow
          wandb login
        fi
      else
        echo -e "${YELLOW}Skipping W&B login. Tracking will be disabled.${NC}"
        # Remove wandb from report_to list
        sed -i 's/- wandb//' config.yaml
      fi
    else
      echo -e "${GREEN}W&B API key found in config.yaml${NC}"
    fi
  else
    echo -e "${YELLOW}W&B tracking not enabled in config.yaml${NC}"
  fi
}

# Setup HuggingFace credentials
setup_huggingface() {
  # Check if push_to_hub is enabled in config.yaml
  if grep -q "push_to_hub: true" config.yaml; then
    if ! grep -q "hf_token:" config.yaml || grep -q "hf_token: null" config.yaml; then
      echo -e "${YELLOW}HuggingFace token not found in config.yaml but push_to_hub is enabled.${NC}"
      echo -e "${BLUE}Please enter your HuggingFace token:${NC}"
      read -r hf_token
      
      if [ -n "$hf_token" ]; then
        # Update config.yaml with the provided token
        sed -i "s/hf_token: null/hf_token: \"$hf_token\"/" config.yaml
        echo -e "${GREEN}HuggingFace token added to config.yaml${NC}"
        
        # Check if hub_model_id is set
        if ! grep -q "hub_model_id:" config.yaml || grep -q "hub_model_id: null" config.yaml; then
          echo -e "${BLUE}Please enter your HuggingFace model ID (username/model-name):${NC}"
          read -r hub_model_id
          
          if [ -n "$hub_model_id" ]; then
            sed -i "s/hub_model_id: null/hub_model_id: \"$hub_model_id\"/" config.yaml
            echo -e "${GREEN}HuggingFace model ID added to config.yaml${NC}"
          else
            echo -e "${YELLOW}No model ID provided. Will need to be set before pushing to Hub.${NC}"
          fi
        fi
      else
        echo -e "${YELLOW}No HuggingFace token provided. Disabling push_to_hub.${NC}"
        sed -i "s/push_to_hub: true/push_to_hub: false/" config.yaml
      fi
    else
      echo -e "${GREEN}HuggingFace token found in config.yaml${NC}"
    fi
  else
    echo -e "${YELLOW}push_to_hub not enabled in config.yaml${NC}"
  fi
}

# Create directories needed by the configuration
create_directories() {
  # Extract output directories from config.yaml
  DATA_OUTPUT_DIR=$(grep "output_dir:" config.yaml | head -1 | awk '{print $2}' | tr -d '"')
  MODEL_OUTPUT_DIR=$(grep "output_dir:" config.yaml | tail -1 | awk '{print $2}' | tr -d '"')
  
  # Create directories if they don't exist
  for dir in "$DATA_OUTPUT_DIR" "$MODEL_OUTPUT_DIR"; do
    if [ ! -d "$dir" ]; then
      echo -e "${BLUE}Creating directory: $dir${NC}"
      mkdir -p "$dir"
    fi
  done
  
  # Create logs directory
  mkdir -p "${MODEL_OUTPUT_DIR}/logs"
  
  echo -e "${GREEN}Required directories created.${NC}"
}

# Check disk space
check_disk_space() {
  # Get available disk space in MB
  AVAILABLE_SPACE=$(df -m . | awk 'NR==2 {print $4}')
  
  if [ "$AVAILABLE_SPACE" -lt 10000 ]; then
    echo -e "${YELLOW}Warning: Less than 10GB of free disk space available. Fine-tuning may require more space.${NC}"
    echo -e "${YELLOW}Available space: $AVAILABLE_SPACE MB${NC}"
    echo -e "${YELLOW}Do you want to continue? (y/n)${NC}"
    read -r continue_choice
    if [[ "$continue_choice" != "y" && "$continue_choice" != "Y" ]]; then
      echo -e "${RED}Setup aborted.${NC}"
      exit 1
    fi
  else
    echo -e "${GREEN}Sufficient disk space available: $AVAILABLE_SPACE MB${NC}"
  fi
}

# Create run script for the main pipeline
# create_run_script() {
#   cat > run_training.sh << 'EOL'
# #!/bin/bash
# set -e

# Activate virtual environment
# source llm_env/bin/activate

# # Step 1: Prepare data
# echo "Step 1: Preparing data..."
# python prepare_data.py --config config.yaml

# # Step 2: Fine-tune model
# echo "Step 2: Fine-tuning model..."
# python finetune_llm.py --config config.yaml

# echo "Fine-tuning pipeline completed successfully!"
# EOL

#   chmod +x run_training.sh
#   echo -e "${GREEN}Run script created as run_training.sh${NC}"
# }

# Create tmux script for background running
# create_tmux_script() {
#   if ! command_exists tmux; then
#     echo -e "${YELLOW}tmux not found. Skipping tmux script creation.${NC}"
#     return
#   fi
  
#   cat > run_in_tmux.sh << 'EOL'
# #!/bin/bash
# SESSION_NAME="llm-training"
# tmux new-session -d -s $SESSION_NAME
# tmux send-keys -t $SESSION_NAME "source llm_env/bin/activate" C-m
# tmux send-keys -t $SESSION_NAME "./run_training.sh" C-m
# echo "Training started in tmux session '$SESSION_NAME'"
# echo "To attach to the session, run: tmux attach-session -t $SESSION_NAME"
# echo "To detach from the session (once attached), press: Ctrl+B, then D"
# EOL

#   chmod +x run_in_tmux.sh
#   echo -e "${GREEN}Tmux script created as run_in_tmux.sh${NC}"
# }

# Main execution
main() {
  echo -e "${BLUE}Checking system requirements...${NC}"
  check_python_version
  check_cuda
  check_disk_space
  
  echo -e "${BLUE}Setting up environment...${NC}"
  setup_virtualenv
  install_dependencies
  
  echo -e "${BLUE}Checking project files...${NC}"
  check_required_files
  check_config
  
  echo -e "${BLUE}Setting up integrations...${NC}"
  setup_wandb
  setup_huggingface
  
  echo -e "${BLUE}Creating directories and scripts...${NC}"
  create_directories
  # create_run_script
  # create_tmux_script
  
  echo -e "${BLUE}=========================================================${NC}"
  echo -e "${GREEN}Setup completed successfully!${NC}"
  echo -e "${BLUE}To start training, you have two options:${NC}"
  echo -e "${YELLOW}1. Run directly: python prepare_data.py --config config.yaml${NC}"
  echo -e "${YELLOW}2. Run python finetune_llm.py --config config.yaml${NC}"
  echo -e "${BLUE}Using tmux is recommended for long training sessions.${NC}"
  echo -e "${BLUE}It allows the training to continue even if you disconnect.${NC}"
  echo -e "${BLUE}=========================================================${NC}"
}

# Execute main function
main