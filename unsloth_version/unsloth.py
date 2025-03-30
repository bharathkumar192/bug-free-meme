import os
import torch
import json
import logging
import wandb
import random
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_data_formats

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
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e9
            
            logger.info(f"Model loaded successfully with {trainable_params:.2f}B trainable parameters")
            
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise
    
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
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise
    
    def train(self):
        """Execute the training process"""
        try:
            # Load and process the data
            train_dataset, eval_dataset = self.load_and_process_data()
            
            # Log dataset statistics
            logger.info(f"Train dataset size: {len(train_dataset)}")
            if eval_dataset:
                logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            
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

# Configuration
config = {
    # Model and dataset settings
    "model_name": "google/gemma-3-12b-pt",
    "input_file": "telugu_results.json",
    "output_dir": "finetuned_gemma3_telugu",
    "max_seq_length": 2048,
    
    # Training settings
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "optimizer": "adamw_torch",
    
    # Evaluation settings
    "eval_split": 0.05,  # 5% for evaluation
    "early_stopping_patience": 3,
    
    # Logging and saving settings
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "save_total_limit": 3,
    
    # Integration settings
    "use_wandb": True,
    "wandb_api_key": "YOUR_WANDB_API_KEY",  # Replace with your key
    "wandb_project": "telugu-gemma-sft",
    "wandb_entity": None,
    "push_to_hub": True,
    "hub_model_id": "your-username/gemma-3-12b-telugu",  # Replace with your username
    "hf_token": "YOUR_HF_TOKEN",  # Replace with your token
    
    # Additional settings
    "train_on_responses_only": True,
    "seed": 42,
    "run_name": "full-finetune",
}

# Main execution
if __name__ == "__main__":
    fine_tuner = TeluguFineTuner(config)
    fine_tuner.train()