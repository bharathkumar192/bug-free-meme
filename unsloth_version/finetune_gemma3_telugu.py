#!/usr/bin/env python
# coding=utf-8
import os
import time
import torch
import json
import logging
import wandb
import random
import numpy as np
import argparse
from pathlib import Path
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_data_formats
# from trl import SFTTrainer
from transformers import Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments, 
    EarlyStoppingCallback,
    # Trainer,
    # AutoProcessor, 
    # Gemma3ForConditionalGeneration
)

# Load configuration
from config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune_full.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class TeluguDataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_and_process_data(self):
        """Load and process the Telugu dataset more efficiently"""
        logger.info(f"Loading dataset from {self.config['input_file']}")
        
        try:
            # Load the JSON file
            with open(self.config['input_file'], 'r', encoding='utf-8') as f:
                questions = json.load(f).get('questions', [])
            
            if not questions:
                raise ValueError("Dataset does not contain valid 'questions' data")
            
            logger.info(f"Loaded {len(questions)} examples from dataset")
            
            # Process and filter data in a single pass
            processed_data = []
            for item in questions:
                question = item.get('question', '').strip()
                response = item.get('response', '').strip()
                
                if question and response:
                    processed_data.append({
                        "conversations": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response}
                        ]
                    })
            
            # Create dataset and split
            dataset = Dataset.from_list(processed_data)
            logger.info(f"Processed {len(dataset)} valid examples")
            
            # Standardize the format first (this helps with compatibility)
            # dataset = standardize_data_formats(dataset)
            
            if self.config["eval_split"] > 0:
                split = dataset.train_test_split(test_size=self.config["eval_split"], seed=self.config["seed"])
                train_dataset, eval_dataset = split["train"], split["test"]
            else:
                train_dataset, eval_dataset = dataset, None
            
            # Apply chat template in a single mapping operation - ensuring string outputs
            def apply_template(examples):
                result = {"text": [self.tokenizer.apply_chat_template(conv) for conv in examples["conversations"]]}
                # Ensure all values are strings to avoid 'int' has no attribute 'startswith' error
                return result
            
            train_dataset = train_dataset.map(
                apply_template, 
                batched=True
            )
            
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    apply_template, 
                    batched=True
                )
                logger.info(f"Train/eval datasets: {len(train_dataset)}/{len(eval_dataset)} examples")
            else:
                logger.info(f"Training dataset: {len(train_dataset)} examples")
            
            return train_dataset, eval_dataset
        
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

class TeluguFineTuner:
    def __init__(self, config):
        """Initialize the fine-tuner with configuration"""
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.setup_directories()
        self.setup_wandb()
        set_seed(self.config["seed"])
        self.setup_model_and_tokenizer()
        
        # Setup dataset processor
        self.data_processor = TeluguDataProcessor(config)
        self.data_processor.set_tokenizer(self.tokenizer)
        
    def setup_directories(self):
        """Setup training directories"""
        # Create output directory
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Create logs directory
        logs_dir = os.path.join(self.config["output_dir"], "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create tensorboard directory
        tensorboard_dir = os.path.join(self.config["output_dir"], "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        
    def setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        if self.config["use_wandb"]:
            try:
                wandb.login(key=self.config["wandb_api_key"])
                wandb.init(
                    project=self.config["wandb_project"],
                    entity=self.config["wandb_entity"],
                    config=self.config,
                    name=f"gemma-3-telugu-{self.config['run_name']}",
                    tags=["gemma-3", "telugu", "full-finetune"],
                )
                logger.info("Successfully initialized Weights & Biases tracking")
            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")
                self.config["use_wandb"] = False
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with standard HuggingFace"""
        logger.info(f"Loading model and tokenizer from {self.config['model_name']} using standard HuggingFace")
        
        try:
            # Load tokenizer
            tokenizer_kwargs = {
                "trust_remote_code": True,
                "token": self.config.get("hf_token", None)
            }
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"], **tokenizer_kwargs)
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            # self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=self.config.get("hf_token", None),
            )
            
            # Enable gradient checkpointing for memory efficiency
            if self.model.supports_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            # Apply Gemma-3 chat template
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is None:
                # Apply Gemma-3 chat template
                self.tokenizer = get_chat_template(
                    self.tokenizer,
                    chat_template="gemma-3"
                )
                logger.info("Applied Gemma-3 chat template")
            
            # Calculate parameters
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e9
            
            logger.info(f"Model loaded successfully with {total_params:.2f}B parameters")
            logger.info(f"Trainable parameters: {trainable_params:.2f}B ({100 * trainable_params / total_params:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error loading model or tokenizer with standard HuggingFace: {str(e)}")
            raise
            
    def train(self):
        """Execute the training process"""
        start_time = time.time()
        
        try:
            # Load and process the data
            train_dataset, eval_dataset = self.data_processor.load_and_process_data()
            
            # Add unsloth performance info
            logger.info("Using Unsloth for faster training")
            

            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=self.config["output_dir"],
                num_train_epochs=self.config["num_train_epochs"],
                per_device_train_batch_size=self.config["per_device_train_batch_size"],
                per_device_eval_batch_size=self.config["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                learning_rate=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                max_grad_norm=self.config["max_grad_norm"],
                warmup_ratio=self.config["warmup_ratio"],
                lr_scheduler_type=self.config["lr_scheduler_type"],
                logging_steps=self.config["logging_steps"],
                save_steps=self.config["save_steps"],
                eval_steps=self.config["eval_steps"] if eval_dataset else None,
                save_total_limit=self.config["save_total_limit"],
                # Change evaluation_strategy to eval_strategy
                eval_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                report_to="wandb" if self.config["use_wandb"] else "none",
                fp16=False,
                bf16=True,  # Use bfloat16 precision
                gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
                optim=self.config["optimizer"],
                push_to_hub=self.config["push_to_hub"],
                hub_model_id=self.config["hub_model_id"] if self.config["push_to_hub"] else None,
                hub_token=self.config.get("hf_token", None),
                seed=self.config["seed"],
                group_by_length=True,
                ddp_find_unused_parameters=False,
            )
            
            # Define training process for full supervised fine-tuning
            logger.info("Initializing SFT trainer for full supervised fine-tuning")
            
            # Initialize SFT trainer
            # trainer = SFTTrainer(
            #     model=self.model,
            #     tokenizer=self.tokenizer,
            #     train_dataset=train_dataset,
            #     eval_dataset=eval_dataset,
            #     args=training_args,
            #     dataset_text_field="text",
            #     max_seq_length=self.config["max_seq_length"],
            #     callbacks=[
            #         EarlyStoppingCallback(
            #             early_stopping_patience=self.config["early_stopping_patience"]
            #         )
            #     ] if eval_dataset else None,
            # )

            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer, 
                    mlm=False
                ),
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config["early_stopping_patience"]
                    )
                ] if eval_dataset else None,
            )
            
            # Apply train-on-responses-only optimization if requested
            if self.config["train_on_responses_only"]:
                try:
                    logger.info("Applying train-on-responses-only optimization")
                    trainer = train_on_responses_only(
                        trainer,
                        instruction_part="<start_of_turn>user\n",
                        response_part="<start_of_turn>model\n",
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply train-on-responses-only: {e}")
                    logger.warning("Proceeding with regular training")
            
            # Start training
            logger.info("Starting full supervised fine-tuning...")
            trainer.train()
            
            # Save final model and tokenizer
            logger.info("Saving final model...")
            trainer.save_model(self.config["output_dir"])
            self.tokenizer.save_pretrained(self.config["output_dir"])
            
            # Calculate training time
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"Training completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            
            # Push to hub if configured
            if self.config["push_to_hub"]:
                logger.info(f"Pushing model to Hugging Face Hub as {self.config['hub_model_id']}")
                trainer.push_to_hub()
            
            logger.info("Full supervised fine-tuning completed successfully!")
            
            # Close wandb
            if self.config["use_wandb"]:
                wandb.finish()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

def main():
    parser = argparse.ArgumentParser(description="Full Fine-tune Gemma-3 model for Telugu")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Print GPU info
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available. Training will be extremely slow.")
    
    # Set seed for reproducibility
    set_seed(config["seed"])
    
    # Initialize trainer with configuration
    trainer = TeluguFineTuner(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()