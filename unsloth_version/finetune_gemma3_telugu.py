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
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_data_formats

# Load configuration
from Gemini_dataset.training_files.unsloth_version.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune_unsloth.log"),
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
        """Load and process the Telugu dataset"""
        logger.info(f"Loading dataset from {self.config['input_file']}")
        
        try:
            # Load the JSON file
            with open(self.config['input_file'], 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            if 'questions' not in raw_data:
                raise ValueError("Dataset does not contain 'questions' key")
            
            questions = raw_data['questions']
            logger.info(f"Loaded {len(questions)} examples from dataset")
            
            # Process data into appropriate format for training
            processed_data = []
            
            for item in questions:
                if 'question' not in item or 'response' not in item:
                    continue
                    
                question = item['question'].strip()
                response = item['response'].strip()
                
                if not question or not response:
                    continue
                
                # Format data for chat template
                conversation = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
                
                processed_data.append({"conversations": conversation})
            
            # Create Hugging Face dataset
            dataset = Dataset.from_list(processed_data)
            logger.info(f"Processed {len(dataset)} valid examples")
            
            # Standardize data format
            dataset = standardize_data_formats(dataset)
            
            # Split dataset
            if self.config["eval_split"] > 0:
                dataset = dataset.train_test_split(
                    test_size=self.config["eval_split"], 
                    seed=self.config["seed"]
                )
                train_dataset = dataset["train"]
                eval_dataset = dataset["test"]
            else:
                train_dataset = dataset
                eval_dataset = None
            
            # Apply chat template
            def tokenize_function(examples):
                texts = self.tokenizer.apply_chat_template(examples["conversations"])
                return {"text": texts}
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            if eval_dataset:
                eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            # Log dataset statistics
            logger.info(f"Train dataset size: {len(train_dataset)}")
            if eval_dataset:
                logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            
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
        """Initialize model and tokenizer using Unsloth"""
        logger.info(f"Loading model and tokenizer from {self.config['model_name']}")
        
        try:
            # For full parameter fine-tuning, we set use_peft=False
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.config["model_name"],
                max_seq_length=self.config["max_seq_length"],
                dtype=torch.bfloat16,  # Using bfloat16 for stability
                load_in_4bit=False,  # Not using 4-bit quantization for full fine-tuning
                load_in_8bit=False,  # Not using 8-bit quantization for full fine-tuning
                # For full parameter fine-tuning:
                use_peft=False,
                full_finetune=True,  # Enable full fine-tuning
                # HF token for gated model access
                token=self.config.get("hf_token", None)
            )
            
            # Set up the chat template for Gemma-3
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="gemma-3"
            )
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Calculate parameters
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e9
            
            logger.info(f"Model loaded successfully with {total_params:.2f}B parameters")
            logger.info(f"Trainable parameters: {trainable_params:.2f}B ({100 * trainable_params / total_params:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise
            
    def train(self):
        """Execute the training process"""
        start_time = time.time()
        
        try:
            # Load and process the data
            train_dataset, eval_dataset = self.data_processor.load_and_process_data()
            
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
                evaluation_strategy="steps" if eval_dataset else "no",
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
            
            # Initialize SFT trainer
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                dataset_text_field="text",
                max_seq_length=self.config["max_seq_length"],
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config["early_stopping_patience"]
                    )
                ] if eval_dataset else None,
            )
            
            # Apply train-on-responses-only optimization
            if self.config["train_on_responses_only"]:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<start_of_turn>user\n",
                    response_part="<start_of_turn>model\n",
                )
            
            # Start training
            logger.info("Starting training...")
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
            
            logger.info("Training completed successfully!")
            
            # Close wandb
            if self.config["use_wandb"]:
                wandb.finish()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3 model for Telugu")
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