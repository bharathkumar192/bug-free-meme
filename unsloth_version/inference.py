#!/usr/bin/env python
# coding=utf-8

import os
import torch
import json
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TeluguModelTester:
    def __init__(self, model_path, use_unsloth=True):
        """Initialize the model tester"""
        self.model_path = model_path
        self.use_unsloth = use_unsloth
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model and tokenizer
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model and tokenizer from {self.model_path}")
        
        if self.use_unsloth:
            # Use Unsloth for faster inference
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            # Use standard HuggingFace loading
            from transformers import AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        # Set up the chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma-3"
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7):
        """Generate a response for the given prompt"""
        # Format as conversation
        messages = [{
            "role": "user",
            "content": prompt
        }]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        # Generate response
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=64,
            do_sample=True,
        )
        
        # Decode the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = generated_text.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
        
        return response
    
    def evaluate_sample_questions(self, questions_file, output_file=None):
        """Evaluate the model on sample questions"""
        logger.info(f"Evaluating model on questions from {questions_file}")
        
        # Load questions
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get('questions', [])
        results = []
        
        for i, item in enumerate(questions):
            if 'question' not in item:
                continue
                
            question = item['question'].strip()
            gold_response = item.get('response', '').strip()
            
            logger.info(f"Processing question {i+1}/{len(questions)}")
            logger.info(f"Question: {question}")
            
            # Generate response
            model_response = self.generate_response(question)
            
            results.append({
                'question': question,
                'gold_response': gold_response,
                'model_response': model_response
            })
            
            logger.info(f"Gold response: {gold_response}")
            logger.info(f"Model response: {model_response}")
            logger.info("-" * 50)
        
        # Save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({'results': results}, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def interactive_mode(self):
        """Start an interactive session with the model"""
        logger.info("Starting interactive mode. Type 'exit' to quit.")
        
        while True:
            try:
                user_input = input("\nEnter your question in Telugu (or 'exit' to quit): ")
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not user_input.strip():
                    continue
                
                # Generate and print response
                response = self.generate_response(user_input)
                print("\nModel response:")
                print(response)
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
        
        logger.info("Interactive session ended")

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Gemma-3 Telugu model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--questions_file", type=str, help="Path to the questions file for evaluation")
    parser.add_argument("--output_file", type=str, help="Path to save evaluation results")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--standard_loading", action="store_true", help="Use standard HuggingFace loading instead of Unsloth")
    
    args = parser.parse_args()
    
    # Initialize the model tester
    tester = TeluguModelTester(
        model_path=args.model_path,
        use_unsloth=not args.standard_loading
    )
    
    # Run in evaluation or interactive mode
    if args.interactive:
        tester.interactive_mode()
    elif args.questions_file:
        tester.evaluate_sample_questions(args.questions_file, args.output_file)
    else:
        logger.error("Either --questions_file or --interactive must be specified")

if __name__ == "__main__":
    main()
