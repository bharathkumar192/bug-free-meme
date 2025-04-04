#!/bin/bash
# Run enhanced inference on a Hugging Face model

# ANSI color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color


# Check if unsloth is installed
if ! pip list | grep -q unsloth; then
  echo -e "${YELLOW}Warning: unsloth package is not installed. Installing now...${NC}"
  pip install unsloth
fi

# Default values
MODEL_ID="bharathkumar1922001/gemma-3-12b-telugu"
HF_TOKEN="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"
PROMPT=""
PROMPT_FILE="prompts.json"
OUTPUT_FILE="tests_results.json"
MAX_NEW_TOKENS=512
TEMPERATURE=0.3
TOP_P=0.95
TOP_K=50
USE_CHAT_TEMPLATE=true
QUESTION_PROMPT=false
DEFAULT_SYSTEM_PROMPT=true
SYSTEM_PROMPT=""
DEFAULT_SYSTEM_PROMPT_TEXT="Provide a helpful, direct answer to the user's question in Telugu with valuable additional information and a natural tone. Focus only on answering what was asked without repeating the question."
DEVICE="auto"



# Function to display usage
usage() {
  echo -e "${BLUE}Usage:${NC} $0 [options]"
  echo ""
  echo "Options:"
  echo "  --model_id MODEL_ID       Hugging Face model ID or local path (required)"
  echo "  --hf_token TOKEN          Hugging Face token for private repos (or set HF_TOKEN env var)"
  echo "  --prompt TEXT             Single prompt for text generation"
  echo "  --prompt_file FILE        JSON file containing prompts for batch generation"
  echo "  --output_file FILE        Output file for results (default: results.json)"
  echo "  --max_new_tokens N        Maximum number of tokens to generate (default: 1024)"
  echo "  --temperature N           Temperature for sampling (default: 0.7)"
  echo "  --top_p N                 Top-p sampling parameter (default: 0.95)"
  echo "  --top_k N                 Top-k sampling parameter (default: 50)"
  echo "  --use_chat_template       Use the model's chat template for generation"
  echo "  --system_prompt TEXT      System prompt to guide model responses"
  echo "  --default_system_prompt   Use default system prompt for better quality responses"
  echo "  --question_prompt         Format prompt as 'Question: X\\nAnswer:' (for non-chat inference)"
  echo "  --device TYPE             Device to run inference on (auto, cpu, cuda) (default: auto)"
  echo "  --help                    Display this help message"
  echo ""
  echo "Example:"
  echo "  $0 --model_id your-username/your-model --prompt \"How are you today?\" --use_chat_template --default_system_prompt"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id)
      MODEL_ID="$2"
      shift 2
      ;;
    --hf_token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --prompt_file)
      PROMPT_FILE="$2"
      shift 2
      ;;
    --output_file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
      shift 2
      ;;
    --use_chat_template)
      USE_CHAT_TEMPLATE=true
      shift
      ;;
    --system_prompt)
      SYSTEM_PROMPT="$2"
      shift 2
      ;;
    --default_system_prompt)
      DEFAULT_SYSTEM_PROMPT=true
      shift
      ;;
    --question_prompt)
      QUESTION_PROMPT=true
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo -e "${RED}Error: Unknown option $1${NC}"
      usage
      ;;
  esac
done

# Check if model ID is provided
if [ -z "$MODEL_ID" ]; then
  echo -e "${RED}Error: --model_id is required${NC}"
  usage
fi

# Check if either prompt or prompt_file is provided
if [ -z "$PROMPT" ] && [ -z "$PROMPT_FILE" ]; then
  echo -e "${RED}Error: Either --prompt or --prompt_file must be provided${NC}"
  usage
fi

# Check for HF token in environment variable if not provided
if [ -z "$HF_TOKEN" ]; then
  HF_TOKEN="$HF_TOKEN"
  if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: No Hugging Face token provided. Using HF_TOKEN environment variable if set.${NC}"
  fi
fi

# Build the command
CMD="python inference_tests.py --model_id \"$MODEL_ID\""

if [ ! -z "$HF_TOKEN" ]; then
  CMD="$CMD --hf_token \"$HF_TOKEN\""
fi

if [ ! -z "$PROMPT" ]; then
  CMD="$CMD --prompt \"$PROMPT\""
fi

if [ ! -z "$PROMPT_FILE" ]; then
  CMD="$CMD --prompt_file \"$PROMPT_FILE\""
fi

CMD="$CMD --output_file \"$OUTPUT_FILE\""
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --top_k $TOP_K"
CMD="$CMD --device $DEVICE"

if [ "$USE_CHAT_TEMPLATE" = true ]; then
  CMD="$CMD --use_chat_template"
fi

if [ ! -z "$SYSTEM_PROMPT" ]; then
  CMD="$CMD --system_prompt \"$SYSTEM_PROMPT\""
fi

if [ "$DEFAULT_SYSTEM_PROMPT" = true ]; then
  CMD="$CMD --default_system_prompt"
  # Add a note about what default system prompt is being used
  echo -e "${YELLOW}Using default system prompt: \"$DEFAULT_SYSTEM_PROMPT_TEXT\"${NC}"
fi

if [ "$QUESTION_PROMPT" = true ]; then
  CMD="$CMD --question_prompt"
fi

# Print execution settings summary
echo -e "${BLUE}========== INFERENCE SETTINGS SUMMARY ==========${NC}"
echo -e "${GREEN}Model ID:${NC}           $MODEL_ID"
echo -e "${GREEN}Output File:${NC}        $OUTPUT_FILE"
echo -e "${GREEN}Device:${NC}             $DEVICE"
echo -e "${GREEN}Max New Tokens:${NC}     $MAX_NEW_TOKENS"
echo -e "${GREEN}Temperature:${NC}        $TEMPERATURE"
echo -e "${GREEN}Top-p:${NC}              $TOP_P"
echo -e "${GREEN}Top-k:${NC}              $TOP_K"
echo -e "${GREEN}Using Chat Template:${NC} $USE_CHAT_TEMPLATE"
echo -e "${GREEN}Using Question Format:${NC} $QUESTION_PROMPT"

if [ "$DEFAULT_SYSTEM_PROMPT" = true ]; then
  echo -e "${GREEN}System Prompt:${NC}      [DEFAULT] \"$DEFAULT_SYSTEM_PROMPT_TEXT\""
elif [ ! -z "$SYSTEM_PROMPT" ]; then
  echo -e "${GREEN}System Prompt:${NC}      \"$SYSTEM_PROMPT\""
else
  echo -e "${GREEN}System Prompt:${NC}      None"
fi

if [ ! -z "$PROMPT" ]; then
  echo -e "${GREEN}Prompt:${NC}             \"$PROMPT\""
elif [ ! -z "$PROMPT_FILE" ]; then
  echo -e "${GREEN}Prompt File:${NC}        $PROMPT_FILE"
fi

echo -e "${BLUE}=============================================${NC}"
echo ""

# Print the command
echo -e "${BLUE}Running command:${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo ""

# Execute the command
eval $CMD

# Check if the command was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Inference completed successfully!${NC}"
  echo -e "${GREEN}Results saved to: $OUTPUT_FILE${NC}"
else
  echo -e "${RED}Inference failed!${NC}"
fi