#!/bin/bash
# Run simplified Gemma-3 Telugu inference

# ANSI color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if unsloth is installed
if ! pip list | grep -q unsloth; then
  echo -e "${YELLOW}Installing unsloth...${NC}"
  pip install unsloth
fi

# Default values
# MODEL_ID="bharathkumar1922001/gemma-3-12b-telugu"
MODEL_ID="bharathkumar1922001/gemma-3-12b-telugu-v1"
HF_TOKEN="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"
PROMPT=""
PROMPT_FILE="prompts.json"
OUTPUT_FILE="results.json"
MAX_TOKENS=1024
TEMPERATURE=0.7
DEVICE="auto"
USE_QUESTION_FORMAT=false
SYSTEM_PROMPT="Answer this question: "

# Help function
usage() {
  echo -e "${BLUE}Simplified Gemma-3 Telugu Inference${NC}"
  echo "Usage: $0 --model_id MODEL_ID [options]"
  echo ""
  echo "Required:"
  echo "  --model_id MODEL_ID       Model ID (e.g., username/model-name)"
  echo ""
  echo "Input options:"
  echo "  --prompt TEXT             Single prompt text"
  echo "  --prompt_file FILE        JSON file with prompts"
  echo "  --output_file FILE        Output file (default: results.json)"
  echo ""
  echo "Authentication:"
  echo "  --hf_token TOKEN          HuggingFace token (or set HF_TOKEN env var)"
  echo ""
  echo "Generation options:"
  echo "  --max_tokens N            Maximum tokens to generate (default: 512)"
  echo "  --temperature N           Temperature (default: 0.7)"
  echo "  --device TYPE             Device: auto, cpu, cuda (default: auto)"
  echo ""
  echo "Formatting options:"
  echo "  --use_question_format     Format prompts as Question/Answer"
  echo "  --system_prompt TEXT      System prompt to use"
  echo "  --system_prompt_file FILE Load system prompt from file"
  echo ""
  echo "Examples:"
  echo "  $0 --model_id username/model --prompt \"హలో, నా పేరు రాము\""
  echo "  $0 --model_id username/model --prompt_file prompts.json --use_question_format"
  exit 1
}

# Parse arguments
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
    --prompt_file)
      PROMPT_FILE="$2"
      shift 2
      ;;
    --output_file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --hf_token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --use_question_format)
      USE_QUESTION_FORMAT=true
      shift
      ;;
    --system_prompt)
      SYSTEM_PROMPT="$2"
      shift 2
      ;;
    --system_prompt_file)
      if [ -f "$2" ]; then
        SYSTEM_PROMPT=$(cat "$2")
      else
        echo -e "${RED}Error: System prompt file not found: $2${NC}"
        exit 1
      fi
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo -e "${RED}Error: Unknown option: $1${NC}"
      usage
      ;;
  esac
done

# Check required arguments
if [ -z "$MODEL_ID" ]; then
  echo -e "${RED}Error: --model_id is required${NC}"
  usage
fi

if [ -z "$PROMPT" ] && [ -z "$PROMPT_FILE" ]; then
  echo -e "${RED}Error: Either --prompt or --prompt_file is required${NC}"
  usage
fi

# Get HF token from environment if not provided
if [ -z "$HF_TOKEN" ]; then
  HF_TOKEN="$HF_TOKEN"
fi

# Build command
CMD="python inference.py --model_id \"$MODEL_ID\""

if [ ! -z "$PROMPT" ]; then
  CMD="$CMD --prompt \"$PROMPT\""
fi

if [ ! -z "$PROMPT_FILE" ]; then
  CMD="$CMD --prompt_file \"$PROMPT_FILE\""
fi

CMD="$CMD --output_file \"$OUTPUT_FILE\""

if [ ! -z "$HF_TOKEN" ]; then
  CMD="$CMD --hf_token \"$HF_TOKEN\""
fi

CMD="$CMD --max_new_tokens $MAX_TOKENS --temperature $TEMPERATURE --device $DEVICE"

if [ "$USE_QUESTION_FORMAT" = true ]; then
  CMD="$CMD --use_question_format"
fi

if [ ! -z "$SYSTEM_PROMPT" ]; then
  CMD="$CMD --system_prompt \"$SYSTEM_PROMPT\""
fi

# Print and execute command
echo -e "${BLUE}Running command:${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo ""

eval $CMD

# Check result
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Inference completed successfully!${NC}"
  if [ ! -z "$OUTPUT_FILE" ]; then
    echo -e "${GREEN}Results saved to: $OUTPUT_FILE${NC}"
  fi
else
  echo -e "${RED}Inference failed!${NC}"
fi