import torch
import argparse
import logging
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from typing import List, Dict, Any, Optional

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

class ModelInference:
    def __init__(
        self,
        model_id: str,
        hf_token: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        device: str = "auto",
    ):
        """
        Initialize the inference class with model and generation parameters
        
        Args:
            model_id (str): Hugging Face model ID or local path
            hf_token (str): Hugging Face token for accessing private repositories
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling
            top_p (float): Top-p sampling parameter
            top_k (int): Top-k sampling parameter
            device (str): Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.model_id = model_id
        self.hf_token = hf_token
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Automatically determine device if set to 'auto'
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize model and tokenizer
        self._init_model()
    
    def _init_model(self):
        """Initialize the model and tokenizer with authentication"""
        try:
            # Log in to Hugging Face
            logger.info(f"Logging in to Hugging Face Hub with provided token")
            login(token=self.hf_token)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                use_auth_token=self.hf_token,
                trust_remote_code=True,
                padding_side="right",
            )
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine data type based on device
            if self.device == "cuda":
                torch_dtype = torch.float16  # Use fp16 for GPU
                logger.info("Using float16 precision for CUDA device")
            else:
                torch_dtype = torch.float32
                logger.info("Using float32 precision for CPU device")
            
            # Load model
            logger.info(f"Loading model from {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                use_auth_token=self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
                use_cache=True,
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Get model details
            model_parameters = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(f"Model size: {model_parameters:.2f}B parameters")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        Generate text based on the given prompt
        
        Args:
            prompt (str): Input prompt for text generation
            
        Returns:
            str: Generated text
        """
        try:
            logger.info(f"Generating text for prompt (first 50 chars): {prompt[:50]}...")
            
            start_time = time.time()
            
            # Prepare generation parameters
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Generate text
            output = self.generator(
                prompt,
                **generation_config
            )
            
            # Extract generated text
            generated_text = output[0]["generated_text"]
            
            # Remove the original prompt from output if needed
            response = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_per_second = self.max_new_tokens / generation_time
            
            logger.info(f"Text generated in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_with_chat_template(self, user_message: str) -> str:
        """
        Generate text using the model's chat template (if available)
        
        Args:
            user_message (str): User message
            
        Returns:
            str: Assistant's response
        """
        try:
            # Check if tokenizer has chat template
            if not hasattr(self.tokenizer, 'apply_chat_template'):
                logger.warning("Tokenizer does not have chat template, falling back to regular generation")
                return self.generate(user_message)
            
            # Format as chat
            messages = [{"role": "user", "content": user_message}]
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            logger.info("Using chat template for generation")
            
            # Generate response
            response = self.generate(prompt)
            
            return response
        
        except Exception as e:
            logger.error(f"Error during chat generation: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_batch(self, prompts: List[str], use_chat_template: bool = False) -> List[str]:
        """
        Generate text for a batch of prompts
        
        Args:
            prompts (List[str]): List of input prompts
            use_chat_template (bool): Whether to use chat template
            
        Returns:
            List[str]: List of generated texts
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            if use_chat_template:
                response = self.generate_with_chat_template(prompt)
            else:
                response = self.generate(prompt)
                
            results.append(response)
            
        return results
    
    def save_to_json(self, prompts: List[str], responses: List[str], output_file: str):
        """
        Save prompts and responses to a JSON file
        
        Args:
            prompts (List[str]): List of input prompts
            responses (List[str]): List of generated responses
            output_file (str): Path to output JSON file
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference on a Hugging Face model")
    
    # Required arguments
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model ID or local path")
    
    # Input options
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for text generation")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="JSON file containing prompts for batch generation")
    parser.add_argument("--output_file", type=str, default="inference_results.json",
                        help="Output file for batch generation results")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    
    # Chat template option
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use the model's chat template for generation")
    
    # Device selection
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run inference on")
    
    # Authentication
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token for accessing private repositories")
    
    args = parser.parse_args()
    
    # Check for HF token in environment variable if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        parser.error("Hugging Face token must be provided either via --hf_token argument or HF_TOKEN environment variable")
    
    # Check if either prompt or prompt_file is provided
    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt_file must be provided")
    
    # Initialize inference class
    inference = ModelInference(
        model_id=args.model_id,
        hf_token=hf_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
    )
    
    # Single prompt generation
    if args.prompt is not None:
        if args.use_chat_template:
            response = inference.generate_with_chat_template(args.prompt)
        else:
            response = inference.generate(args.prompt)
            
        print("\nGenerated Text:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Save single result to JSON
        inference.save_to_json([args.prompt], [response], args.output_file)
    
    # Batch generation from file
    elif args.prompt_file is not None:
        try:
            # Load prompts from file
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract prompts
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict) and "prompts" in data:
                prompts = data["prompts"]
            elif isinstance(data, dict) and "questions" in data:
                prompts = [item["question"] for item in data["questions"]]
            else:
                logger.error(f"Unsupported prompt file format")
                return
                
            logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
            
            # Generate responses
            responses = inference.generate_batch(prompts, args.use_chat_template)
            
            # Save results
            inference.save_to_json(prompts, responses, args.output_file)
            
        except Exception as e:
            logger.error(f"Error processing prompt file: {str(e)}")


if __name__ == "__main__":
    main()