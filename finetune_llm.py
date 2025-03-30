import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
    HfArgumentParser,
)
from datasets import load_dataset, Dataset
import wandb
from typing import Dict, Sequence, List, Any, Optional
import logging
import numpy as np
import math
from transformers.trainer_utils import get_last_checkpoint
import json
from pathlib import Path
import random
import glob
from copy import deepcopy
import shutil
import re
from tqdm import tqdm
import time
from datetime import datetime
import sys
import argparse

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UnifiedConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_and_process_data(self):
        """Load and process the dataset"""
        logger.info(f"Loading dataset from {self.config.data.input_file}")
        
        try:
            # Load the JSON file
            with open(self.config.data.input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            logger.info(f"Loaded {len(raw_data['questions'])} examples from dataset")
            
            # Check for data quality
            self._validate_data(raw_data)
            
            # Process data into appropriate format for training
            processed_data = self._process_data(raw_data)
            
            # Split the dataset
            train_data, val_data, test_data = self._split_data(processed_data)
            
            logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test examples")
            
            # Convert to HuggingFace datasets
            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data)
            test_dataset = Dataset.from_list(test_data)
            
            # Tokenize the datasets
            tokenized_datasets = self._tokenize_datasets(train_dataset, val_dataset, test_dataset)
            
            return tokenized_datasets
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise
    
    def _validate_data(self, data):
        """Validate data quality"""
        if 'questions' not in data:
            raise ValueError("Dataset does not contain 'questions' key")
        
        if len(data['questions']) == 0:
            raise ValueError("Dataset is empty")
        
        # Check random samples
        random.seed(self.config.integration.seed)
        sample_size = min(100, len(data['questions']))
        samples = random.sample(data['questions'], sample_size)
        
        for i, sample in enumerate(samples):
            if 'question' not in sample:
                logger.warning(f"Sample {i} missing 'question' field")
            if 'response' not in sample:
                logger.warning(f"Sample {i} missing 'response' field")
            if 'question' in sample and len(sample['question'].strip()) < 5:
                logger.warning(f"Sample {i} has very short question: {sample['question']}")
            if 'response' in sample and len(sample['response'].strip()) < 10:
                logger.warning(f"Sample {i} has very short response: {sample['response']}")
    
    def _process_data(self, data):
        """Process raw data into training format"""
        processed_data = []
        
        for item in data['questions']:
            if 'question' not in item or 'response' not in item:
                continue
                
            question = item['question'].strip()
            response = item['response'].strip()
            
            if not question or not response:
                continue
            
            # Format data for chat template if specified
            if self.config.model.use_chat_template:
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
                processed_data.append({
                    "messages": messages,
                    # Additional metadata that might be useful
                    'q_length': len(question),
                    'r_length': len(response),
                })
            else:
                # Standard format if not using chat template
                processed_data.append({
                    'question': question,
                    'response': response,
                    # Additional metadata that might be useful
                    'q_length': len(question),
                    'r_length': len(response),
                })
        
        logger.info(f"Processed {len(processed_data)} valid examples")
        return processed_data
    
    def _split_data(self, data):
        """Split data into train/val/test sets"""
        # Shuffle data deterministically
        random.seed(self.config.integration.seed)
        shuffled_data = deepcopy(data)
        random.shuffle(shuffled_data)
        
        # Calculate reduction based on data_percentage if it exists
        total_original_size = len(shuffled_data)
        data_percentage = getattr(self.config.data, 'data_percentage', 1.0)  # Default to 100% if not specified
        reduced_size = int(total_original_size * data_percentage)
        reduced_data = shuffled_data[:reduced_size]
        
        # Calculate split sizes on the reduced data
        total_size = len(reduced_data)
        val_size = int(total_size * self.config.data.val_ratio)
        test_size = int(total_size * self.config.data.test_ratio)
        train_size = total_size - val_size - test_size
        
        # Split the data
        train_data = reduced_data[:train_size]
        val_data = reduced_data[train_size:train_size + val_size]
        test_data = reduced_data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def _tokenize_datasets(self, train_dataset, val_dataset, test_dataset):
        """Tokenize datasets for training"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer first.")
        
        def tokenize_function(examples):
            """Tokenize examples for supervised fine-tuning"""
            if self.config.model.use_chat_template:
                # Use the model's chat template for formatting
                formatted_inputs = []
                for messages in examples["messages"]:
                    formatted_inputs.append(self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=False
                    ))
                
                # Tokenize formatted inputs
                tokenized = self.tokenizer(
                    formatted_inputs,
                    truncation=True,
                    max_length=self.config.training.max_seq_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # Return tokenized inputs with labels
                tokenized["labels"] = tokenized["input_ids"].clone()
                
                return tokenized
            else:
                # Standard approach (non-chat template)
                model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
                
                for question, response in zip(examples["question"], examples["response"]):
                    # Prepare the input
                    prompt = f"Question: {question}\nAnswer: "
                    prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
                    
                    # Prepare the full sequence
                    full_text = prompt + response
                    tokenized_full = self.tokenizer(
                        full_text, 
                        truncation=True,
                        max_length=self.config.training.max_seq_length,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                    input_ids = tokenized_full.input_ids[0]
                    attention_mask = tokenized_full.attention_mask[0]
                    
                    # Create labels: -100 for prompt, actual IDs for response
                    prompt_len = len(prompt_ids)
                    labels = input_ids.clone()
                    labels[:prompt_len] = -100  # Mask prompt tokens
                    
                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["attention_mask"].append(attention_mask)
                    model_inputs["labels"].append(labels)
                    
                # Convert lists to tensors
                for key in model_inputs:
                    model_inputs[key] = torch.stack(model_inputs[key])
                    
                return model_inputs
        
        # Apply tokenization
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        
        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
        )
        
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=test_dataset.column_names,
        )
        
        return {
            "train": tokenized_train,
            "validation": tokenized_val,
            "test": tokenized_test
        }

class LLMFineTuner:
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.setup_directories()
        self.setup_tracking()
        self.setup_model_and_tokenizer()
        
        # Setup dataset processor
        self.dataset_processor = DatasetProcessor(config)
        self.dataset_processor.set_tokenizer(self.tokenizer)
        
    def setup_directories(self):
        """Setup training directories"""
        # Create output directory
        os.makedirs(self.config.logging.output_dir, exist_ok=True)
        
        # Create logs directory
        logs_dir = os.path.join(self.config.logging.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create backup checkpoint directory
        backup_dir = os.path.join(self.config.logging.output_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create tensorboard directory
        tensorboard_dir = os.path.join(self.config.logging.output_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        
    def setup_tracking(self):
        """Initialize tracking platforms (W&B, HuggingFace)"""
        # Setup Weights & Biases
        if "wandb" in self.config.logging.report_to:
            try:
                # Set API key if provided
                if self.config.integration.wandb_api_key:
                    os.environ["WANDB_API_KEY"] = self.config.integration.wandb_api_key
                    wandb.init(
                    project=self.config.integration.wandb_project,
                    entity=self.config.integration.wandb_entity,
                    config=vars(self.config),
                    name=f"{self.config.model.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    tags=["llm", "finetuning"],
                )
                logger.info("Successfully initialized Weights & Biases tracking")
            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")
                if "wandb" in self.config.logging.report_to:
                    self.config.logging.report_to.remove("wandb")
        
        # Setup HuggingFace Hub
        if self.config.integration.push_to_hub:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=self.config.integration.hf_token)
                # Validate token
                api.whoami()
                logger.info("Successfully authenticated with Hugging Face Hub")
            except Exception as e:
                logger.warning(f"Failed to authenticate with Hugging Face Hub: {e}")
                self.config.integration.push_to_hub = False
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading tokenizer from {self.config.model.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right",
            )

            logger.info(f"Config chat_template exists: {hasattr(self.config.model, 'chat_template')}")
            if hasattr(self.config.model, 'chat_template'):
                logger.info(f"Config chat_template value: {self.config.model.chat_template}")
            logger.info(f"==Tokenizer has chat_template: {hasattr(self.tokenizer, 'chat_template')}")
            if hasattr(self.tokenizer, 'chat_template'):
                logger.info(f"==Tokenizer chat_template value: {self.tokenizer.chat_template}")

            
            # Ensure tokenizer has chat template if specified
            if self.config.model.use_chat_template:
                    # Force set the template from config if available
                if hasattr(self.config.model, 'chat_template') and self.config.model.chat_template:
                    logger.info("Setting custom chat template from config")
                    self.tokenizer.chat_template = self.config.model.chat_template
                else:
                    logger.warning("No chat template in config. Setting default template.")
                    # Default Gemma-3 template
                    self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<start_of_turn>user\n{{ message['content'] }}\n<end_of_turn>\n{% elif message['role'] == 'assistant' %}\n<start_of_turn>model\n{{ message['content'] }}\n<end_of_turn>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<start_of_turn>model\n{% endif %}"            
            
            if self.tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loading model from {self.config.model.model_name}")
            torch_dtype = torch.bfloat16 if self.config.training.bf16 else (torch.float16 if self.config.training.fp16 else torch.float32)
            
            # Check available GPU memory before loading model
            if torch.cuda.is_available():
                free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
                logger.info(f"Available GPU memory: {free_memory:.2f} GB")
                if free_memory < 24 and not (self.config.training.use_4bit or self.config.training.use_peft):
                    logger.warning("Less than 24GB GPU memory available. Consider using 4-bit quantization or LoRA.")
            
            # Setup model loading arguments
            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                # "use_cache": not self.config.training.gradient_checkpointing,
            }
            
            # Add quantization if configured
            if self.config.training.use_4bit:
                try:
                    import bitsandbytes as bnb
                    from transformers import BitsAndBytesConfig
                    
                    logger.info("Using 4-bit quantization")
                    compute_dtype = getattr(torch, self.config.training.bnb_4bit_compute_dtype)
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    
                    model_kwargs["quantization_config"] = quantization_config
                    
                except ImportError:
                    logger.error("Failed to import bitsandbytes. Cannot use 4-bit quantization.")
                    self.config.training.use_4bit = False
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.model_name,
                **model_kwargs
            )

            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = not self.config.training.gradient_checkpointing
                logger.info(f"Setting model config use_cache={self.model.config.use_cache}")
            
            # Apply LoRA if configured
            if self.config.training.use_peft:
                try:
                    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

                    logger.info("Applying LoRA adapter")
                    
                    # Prepare model for k-bit training if using quantization
                    if self.config.training.use_4bit:
                        self.model = prepare_model_for_kbit_training(self.model)
                    
                    # Configure LoRA
                    lora_config = LoraConfig(
                        r=self.config.training.lora_r,
                        lora_alpha=self.config.training.lora_alpha,
                        lora_dropout=self.config.training.lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=self._get_target_modules(),
                    )
                    
                    # Apply LoRA
                    self.model = get_peft_model(self.model, lora_config)
                    self.model.print_trainable_parameters()
                    
                except ImportError:
                    logger.error("Failed to import PEFT. Cannot use LoRA.")
                    self.config.training.use_peft = False
            
            # Enable gradient checkpointing if configured
            if self.config.training.gradient_checkpointing:
                logger.info("Enabling gradient checkpointing")
                self.model.gradient_checkpointing_enable()
                
            # Calculate parameters
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e9
            
            logger.info(f"Model loaded successfully with {total_params:.2f}B parameters")
            logger.info(f"Trainable parameters: {trainable_params:.2f}B ({100 * trainable_params / total_params:.2f}%)")
            
            # Log model configuration to W&B
            if "wandb" in self.config.logging.report_to and wandb.run is not None:
                wandb.config.update({
                    "model_size_b": total_params,
                    "trainable_params_b": trainable_params,
                    "model_name": self.config.model.model_name,
                })
                
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise
    
    def _get_target_modules(self):
        """Get target modules for LoRA based on model architecture"""
        model_name = self.config.model.model_name.lower()
        
        if "llama" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gemma" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "mistral" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "mpt" in model_name:
            return ["Wqkv", "out_proj"]
        elif "gpt-neox" in model_name or "pythia" in model_name:
            return ["query_key_value", "dense"]
        elif "falcon" in model_name:
            return ["query_key_value", "dense"]
        else:
            # Generic target for transformer models
            logger.warning(f"No specific target modules defined for {model_name}. Using generic targets.")
            return ["query", "key", "value", "dense"]
        
    def prepare_dataset(self):
        """Prepare dataset for training"""
        return self.dataset_processor.load_and_process_data()
        
    def backup_checkpoint(self, checkpoint_dir):
        """Backup checkpoint to prevent corruption"""
        try:
            backup_dir = os.path.join(self.config.logging.output_dir, "backups", os.path.basename(checkpoint_dir))
            logger.info(f"Backing up checkpoint {checkpoint_dir} to {backup_dir}")
            
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
                
            shutil.copytree(checkpoint_dir, backup_dir)
            logger.info("Checkpoint backup completed")
        except Exception as e:
            logger.warning(f"Failed to backup checkpoint: {str(e)}")
    
    def log_training_progress(self, metrics, step, phase="train"):
        """Log training progress to various platforms"""
        if "wandb" in self.config.logging.report_to and wandb.run is not None:
            wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=step)
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"{phase.capitalize()} step {step}: {metrics_str}")
    
    def train(self):
        """Main training loop"""
        start_time = time.time()
        
        try:
            # Prepare dataset
            logger.info("Preparing dataset...")
            dataset = self.prepare_dataset()
            
            # Log dataset statistics
            train_size = len(dataset["train"])
            val_size = len(dataset["validation"])
            test_size = len(dataset["test"])
            
            logger.info(f"Dataset statistics:")
            logger.info(f"  Train size: {train_size}")
            logger.info(f"  Validation size: {val_size}")
            logger.info(f"  Test size: {test_size}")
            
            if "wandb" in self.config.logging.report_to and wandb.run is not None:
                wandb.config.update({
                    "train_examples": train_size,
                    "val_examples": val_size,
                    "test_examples": test_size,
                })
            
            # Setup training arguments
            training_args = self.config.get_training_args()
            
            # Setup callbacks
            callbacks = [
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience
                )
            ]
            
            # Initialize trainer
            logger.info("Initializing trainer...")
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                tokenizer=self.tokenizer,
                callbacks=callbacks,
                data_collator=default_data_collator,
            )
            
            # Add custom logging
            original_log = trainer.log
            def custom_log(metrics, step=None, **kwargs):
                # Call the original log method
                original_log(metrics, step, **kwargs)
                # Additional custom logging
                self.log_training_progress(metrics, step)
            trainer.log = custom_log
            
            # Check for checkpoint
            last_checkpoint = get_last_checkpoint(self.config.logging.output_dir)
            if last_checkpoint is not None:
                logger.info(f"Found checkpoint: {last_checkpoint}")
                
                # Check if checkpoint is complete
                checkpoint_files = os.listdir(last_checkpoint)
                expected_files = ['config.json', 'pytorch_model.bin', 'trainer_state.json']
                
                if all(f in checkpoint_files for f in expected_files):
                    logger.info(f"Resuming from checkpoint: {last_checkpoint}")
                    # Create a backup of the checkpoint
                    self.backup_checkpoint(last_checkpoint)
                else:
                    logger.warning(f"Checkpoint at {last_checkpoint} seems incomplete. Starting from scratch.")
                    last_checkpoint = None
            
            # Start training
            logger.info("Starting training...")
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            
            # Save final model and metrics
            logger.info("Saving final model...")
            trainer.save_model()
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            
            # Save tokenizer
            self.tokenizer.save_pretrained(self.config.logging.output_dir)
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
            
            # Calculate training time
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"Training completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            
            if "wandb" in self.config.logging.report_to and wandb.run is not None:
                wandb.log({"training_time_hours": total_time / 3600})
            
            # Push to hub if configured
            if self.config.integration.push_to_hub:
                logger.info(f"Pushing model to Hugging Face Hub as {self.config.integration.hub_model_id}")
                trainer.push_to_hub()
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise
        finally:
            if "wandb" in self.config.logging.report_to and wandb.run is not None:
                wandb.finish()
            
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune LLM using the Hugging Face Transformers")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = UnifiedConfig.from_file(args.config)
    
    # Initialize trainer with configuration
    trainer = LLMFineTuner(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()