#!/bin/bash
# Run inference on a Hugging Face model

# ANSI color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if the inference.py script exists
if [ ! -f "inference_tests.py" ]; then
  echo -e "${RED}Error: inference_tests.py script not found in the current directory${NC}"
  exit 1
fi

# Check if unsloth is installed
if ! pip list | grep -q unsloth; then
  echo -e "${YELLOW}Warning: unsloth package is not installed. Installing now...${NC}"
  pip install unsloth
fi

# Default values
MODEL_ID="bharathkumar1922001/gemma-3-12b-telugu"
HF_TOKEN="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"
QUESTION_FORMAT=false
MAX_NEW_TOKENS=512

# Help function
usage() {
  echo "Usage: $0 --model_id MODEL_ID --prompt PROMPT [options]"
  echo ""
  echo "Options:"
  echo "  --model_id MODEL_ID       Model ID (required)"
  echo "  --prompt PROMPT           Prompt text (required)"
  echo "  --hf_token TOKEN          HuggingFace token (or set HF_TOKEN env var)"
  echo "  --question_format         Format prompt as question/answer"
  echo "  --max_new_tokens N        Max new tokens (default: 512)"
  echo "  --help                    Display this help"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id)
      MODEL_ID="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --hf_token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --question_format)
      QUESTION_FORMAT=true
      shift
      ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"
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

# Check required arguments
if [ -z "$MODEL_ID" ]; then
  echo -e "${RED}Error: --model_id is required${NC}"
  usage
fi

if [ -z "$PROMPT" ]; then
  echo -e "${RED}Error: --prompt is required${NC}"
  usage
fi

# Check for HF token in environment variable if not provided
if [ -z "$HF_TOKEN" ]; then
  HF_TOKEN="$HF_TOKEN"
  if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: No HuggingFace token provided. Using HF_TOKEN environment variable if set.${NC}"
  fi
fi

# Build the command
CMD="python minimal_inference.py --model_id \"$MODEL_ID\" --prompt \"$PROMPT\""

if [ ! -z "$HF_TOKEN" ]; then
  CMD="$CMD --hf_token \"$HF_TOKEN\""
fi

if [ "$QUESTION_FORMAT" = true ]; then
  CMD="$CMD --question_format"
fi

CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"

# Print the command
echo -e "${BLUE}Running command:${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo ""

# Execute the command
eval $CMD

# Check if the command was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Inference completed successfully!${NC}"
else
  echo -e "${RED}Inference failed!${NC}"
fi