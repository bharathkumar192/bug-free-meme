import unsloth  # Import unsloth first
import torch
import argparse
import logging
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from unsloth.chat_templates import get_chat_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_model(model_id, hf_token, device="auto"):
    """
    Load the model and tokenizer
    
    Args:
        model_id (str): Hugging Face model ID
        hf_token (str): Hugging Face token
        device (str): Device to run on
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Login to HuggingFace
    login(token=hf_token)
    logger.info(f"Loading model from {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply Gemma-3 chat template if needed
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is None:
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Select appropriate dtype based on device
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None
    )
    
    logger.info(f"Model loaded successfully on {device}")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, system_prompt=None, generation_params=None):
    """
    Generate text from a prompt
    
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        prompt (str): Input prompt
        system_prompt (str): Optional system prompt
        generation_params (dict): Generation parameters
        
    Returns:
        str: Generated text
    """
    start_time = time.time()
    logger.info(f"Generating text for: {prompt[:50]}...")
    
    # Apply system prompt if provided
    if system_prompt:
        # Format with both system and user messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Just use the plain prompt
        formatted_prompt = prompt
    
    # Print what's being sent to the model
    print("\n" + "=" * 40)
    print("SENDING TO MODEL:")
    print("-" * 40)
    print(formatted_prompt)
    print("=" * 40 + "\n")
    
    # Default generation parameters
    if generation_params is None:
        generation_params = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Get device from model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        output = model.generate(**inputs, **generation_params)
    
    # Decode
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the response
    if generated_text.startswith(formatted_prompt):
        response = generated_text[len(formatted_prompt):].strip()
    else:
        response = generated_text.strip()
    
    # Clean up response if using chat template
    if system_prompt:
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[1].strip()
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0].strip()
    
    generation_time = time.time() - start_time
    logger.info(f"Generated text in {generation_time:.2f}s")
    
    return response

def process_batch(model, tokenizer, prompts, system_prompt=None, generation_params=None, use_question_format=False):
    """
    Process a batch of prompts
    
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        prompts (list): List of prompts
        system_prompt (str): Optional system prompt
        generation_params (dict): Generation parameters
        use_question_format (bool): Format prompts as questions
        
    Returns:
        list: List of generated responses
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}")
        
        # Format as question if needed
        if use_question_format:
            formatted_prompt = f"Question: {prompt}\nAnswer:"
        else:
            formatted_prompt = prompt
        
        # Generate response
        try:
            response = generate_text(model, tokenizer, formatted_prompt, system_prompt, generation_params)
            results.append(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            results.append(f"Error: {str(e)}")
    
    return results

def save_results(prompts, responses, output_file):
    """
    Save results to a JSON file
    
    Args:
        prompts (list): List of prompts
        responses (list): List of responses
        output_file (str): Output file path
    """
    results = []
    for prompt, response in zip(prompts, responses):
        results.append({
            "prompt": prompt,
            "response": response
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Simplified Gemma-3 Telugu Inference")
    
    # Basic arguments
    parser.add_argument("--model_id", type=str, required=True, help="Model ID")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt")
    parser.add_argument("--prompt_file", type=str, default=None, help="JSON file with prompts")
    parser.add_argument("--output_file", type=str, default="results.json", help="Output file")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    
    # Generation options
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    
    # Formatting options
    parser.add_argument("--use_question_format", action="store_true", help="Format as Question/Answer")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt")
    
    args = parser.parse_args()
    
    # Check for token in environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token required. Use --hf_token or set HF_TOKEN env var")
    
    # Check for prompt or prompt file
    if args.prompt is None and args.prompt_file is None:
        raise ValueError("Either --prompt or --prompt_file is required")
    
    # Load model and tokenizer
    model, tokenizer = setup_model(args.model_id, hf_token, args.device)
    
    # Set generation parameters
    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    # For single prompt
    if args.prompt:
        response = generate_text(
            model, 
            tokenizer, 
            f"Question: {args.prompt}\nAnswer:" if args.use_question_format else args.prompt,
            args.system_prompt,
            generation_params
        )
        
        print("\nGenerated Text:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        save_results([args.prompt], [response], args.output_file)
    
    # For batch processing
    elif args.prompt_file:
        # Load prompts
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract prompts from different formats
        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict) and "prompts" in data:
            prompts = data["prompts"]
        elif isinstance(data, dict) and "questions" in data:
            prompts = [item["question"] for item in data["questions"]]
        else:
            raise ValueError("Unsupported prompt file format")
        
        logger.info(f"Loaded {len(prompts)} prompts")
        
        # Process batch
        responses = process_batch(
            model,
            tokenizer,
            prompts,
            args.system_prompt,
            generation_params,
            args.use_question_format
        )
        
        # Save results
        save_results(prompts, responses, args.output_file)

if __name__ == "__main__":
    main()