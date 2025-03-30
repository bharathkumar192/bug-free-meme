import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2SeqLanguageModeling,
    EarlyStoppingCallback,
    default_data_collator,
)
from datasets import load_dataset, Dataset
import wandb
from typing import Dict, Sequence, List, Any
import logging
import numpy as np
from dataclasses import dataclass, field
import math
from transformers.trainer_utils import get_last_checkpoint
import json
from pathlib import Path
import random
import glob
from copy import deepcopy
import shutil
import re

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

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"  # Base model
    dataset_path: str = "telugu_results.json"  # Path to dataset
    output_dir: str = "finetuned_model"  # Output directory
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Scheduler settings
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"  # Cosine scheduler with restarts
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 5  # Keep only the last 5 checkpoints
    eval_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # Sequence length
    max_seq_length: int = 2048
    
    # Optimization settings
    fp16: bool = True
    bf16: bool = False  # Set to True if only GPU supports it (A100, H100, etc.)
    gradient_checkpointing: bool = True
    
    # DeepSpeed configuration (optional)
    deepspeed_config: str = None
    
    # Dataset processing
    val_size: float = 0.05
    test_size: float = 0.05
    seed: int = 42
    prompt_template: str = "Question: {question}\nAnswer: "
    
    # Tracking
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    # Function to get the training arguments
    def get_training_args(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to=self.report_to,
            fp16=self.fp16,
            bf16=self.bf16,
            deepspeed=self.deepspeed_config,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_torch", 
            ddp_find_unused_parameters=False,
            group_by_length=True,  # Improves efficiency by grouping similar lengths
        )

class DatasetProcessor:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_and_process_data(self):
        """Load and process the dataset"""
        logger.info(f"Loading dataset from {self.config.dataset_path}")
        
        try:
            # Load the JSON file
            with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
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
                
            # Create input-output pairs for training
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
        random.seed(self.config.seed)
        shuffled_data = deepcopy(data)
        random.shuffle(shuffled_data)
        
        # Calculate split sizes
        total_size = len(shuffled_data)
        val_size = int(total_size * self.config.val_size)
        test_size = int(total_size * self.config.test_size)
        train_size = total_size - val_size - test_size
        
        # Split the data
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def _tokenize_datasets(self, train_dataset, val_dataset, test_dataset):
        """Tokenize datasets for training"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer first.")
        
        def tokenize_function(examples):
            """Tokenize examples for supervised fine-tuning"""
            # Format as instruction following
            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            
            for question, response in zip(examples["question"], examples["response"]):
                # Prepare the input (instruction/prompt format)
                prompt = self.config.prompt_template.format(question=question)
                prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
                
                # Prepare the full sequence (input + response)
                full_text = prompt + response
                tokenized_full = self.tokenizer(
                    full_text, 
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                input_ids = tokenized_full.input_ids[0]
                attention_mask = tokenized_full.attention_mask[0]
                
                # Create label mask: -100 for prompt (ignored in loss), actual token IDs for response
                # Find the prompt length in tokenized form
                prompt_len = len(prompt_ids)
                
                # Create labels: -100 for prompt, actual IDs for response
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
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.setup_directories()
        self.setup_wandb()
        self.setup_model_and_tokenizer()
        
        # Setup dataset processor
        self.dataset_processor = DatasetProcessor(config)
        self.dataset_processor.set_tokenizer(self.tokenizer)
        
    def setup_directories(self):
        """Setup training directories"""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create logs directory
        logs_dir = os.path.join(self.config.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create backup checkpoint directory
        backup_dir = os.path.join(self.config.output_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        if "wandb" in self.config.report_to:
            wandb.init(
                project="telugu-llm-finetuning",
                config=vars(self.config),
                name=f"{self.config.model_name.split('/')[-1]}-sft",
            )
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading tokenizer from {self.config.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right",
            )
            
            if self.tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loading model from {self.config.model_name}")
            torch_dtype = torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else torch.float32)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                use_cache=not self.config.gradient_checkpointing,  # Disable KV cache when using gradient checkpointing
            )
            
            if self.config.gradient_checkpointing:
                logger.info("Enabling gradient checkpointing")
                self.model.gradient_checkpointing_enable()
                
            logger.info(f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
                
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise
        
    def prepare_dataset(self):
        """Prepare dataset for training"""
        return self.dataset_processor.load_and_process_data()
        
    def backup_checkpoint(self, checkpoint_dir):
        """Backup checkpoint to prevent corruption"""
        try:
            backup_dir = os.path.join(self.config.output_dir, "backups", os.path.basename(checkpoint_dir))
            logger.info(f"Backing up checkpoint {checkpoint_dir} to {backup_dir}")
            
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
                
            shutil.copytree(checkpoint_dir, backup_dir)
            logger.info("Checkpoint backup completed")
        except Exception as e:
            logger.warning(f"Failed to backup checkpoint: {str(e)}")
        
    def train(self):
        """Main training loop"""
        try:
            # Prepare dataset
            logger.info("Preparing dataset...")
            dataset = self.prepare_dataset()
            
            # Setup training arguments
            training_args = self.config.get_training_args()
            
            # Setup callbacks
            callbacks = [
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
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
            
            # Check for checkpoint
            last_checkpoint = get_last_checkpoint(self.config.output_dir)
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
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(dataset["test"])
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise
        finally:
            if "wandb" in self.config.report_to:
                wandb.finish()
            
def main():
    # Load configuration
    config = TrainingConfig()
    
    # Initialize and run training
    trainer = LLMFineTuner(config)
    trainer.train()

if __name__ == "__main__":
    main() 