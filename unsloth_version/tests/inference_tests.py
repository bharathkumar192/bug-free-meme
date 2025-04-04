import unsloth  # Import unsloth first to apply all optimizations
import torch
import argparse
import logging
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from typing import List, Dict, Any, Optional
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
                token=self.hf_token,
                trust_remote_code=True,
                padding_side="right",
            )
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Apply Gemma-3 chat template if needed
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is None:
                logger.info("Applying Gemma-3 chat template")
                self.tokenizer = get_chat_template(
                    self.tokenizer,
                    chat_template="gemma-3"
                )
            
            # Determine data type based on device
            if self.device == "cuda":
                torch_dtype = torch.bfloat16  # Use bfloat16 for GPU (not fp16/float16)
                logger.info("Using bfloat16 precision for CUDA device")
            else:
                torch_dtype = torch.float32
                logger.info("Using float32 precision for CPU device")
            
            # Load model
            logger.info(f"Loading model from {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                # use_cache=True
            )
            
            # Create text generation pipeline
            try:
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                )
                logger.info("Text generation pipeline created successfully")
            except Exception as e:
                logger.warning(f"Could not create text generation pipeline: {e}")
                logger.info("Will fall back to direct model generation")
            
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
            
            # Use the direct generation method
            return self._generate_directly(prompt, {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            })
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return f"Error: {str(e)}"
    
    def _generate_directly(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """
        Directly generate text using the model without the pipeline
        
        Args:
            prompt (str): Input prompt
            generation_config (Dict[str, Any]): Generation parameters
            
        Returns:
            str: Generated text
        """
        try:
            start_time = time.time()
            
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove the original prompt from output if needed
            response = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_per_second = generation_config["max_new_tokens"] / generation_time if generation_time > 0 else 0
            
            logger.info(f"Text generated in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error in direct generation: {str(e)}")
            raise
    
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
                return self._generate_directly(user_message, {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "do_sample": self.temperature > 0,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                })
            
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
            response = self._generate_directly(prompt, {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            })
            
            # Remove the system message if present in the response
            if "<start_of_turn>model" in response:
                response = response.split("<start_of_turn>model")[1].strip()
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0].strip()
            
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
            
            try:
                if use_chat_template:
                    response = self.generate_with_chat_template(prompt)
                else:
                    response = self._generate_directly(prompt, {
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "do_sample": self.temperature > 0,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                    })
            except Exception as e:
                logger.error(f"Error during text generation: {str(e)}")
                response = f"Error: {str(e)}"
                
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
    parser.add_argument("--question_prompt", type=str, 
                        default=None,
                        help="Question prompt to append to each prompt (for 'Question: X Answer:' format)")
    
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
        # Apply question format if needed
        prompt = args.prompt
        if args.question_prompt and not args.use_chat_template:
            prompt = f"Question: {args.prompt}\nAnswer:"
            
        if args.use_chat_template:
            response = inference.generate_with_chat_template(args.prompt)
        else:
            response = inference.generate(prompt)
            
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
            
            # Apply question format if needed
            if args.question_prompt and not args.use_chat_template:
                formatted_prompts = [f"Question: {p}\nAnswer:" for p in prompts]
                # Generate responses with formatted prompts
                responses = inference.generate_batch(formatted_prompts, args.use_chat_template)
            else:
                # Generate responses with original prompts
                responses = inference.generate_batch(prompts, args.use_chat_template)
            
            # Save results
            inference.save_to_json(prompts, responses, args.output_file)
            
        except Exception as e:
            logger.error(f"Error processing prompt file: {str(e)}")


if __name__ == "__main__":
    main()