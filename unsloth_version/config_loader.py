import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TeluguFineTuningConfig:
    # Model and dataset settings
    model_name: str = "google/gemma-3-12b-pt"
    input_file: str = "telugu_results.json"
    output_dir: str = "finetuned_gemma3_telugu"
    max_seq_length: int = 2048
    
    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_torch"
    
    # Evaluation settings
    eval_split: float = 0.05
    early_stopping_patience: int = 3
    
    # Logging and saving settings
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Integration settings
    use_wandb: bool = True
    wandb_api_key: str = "YOUR_WANDB_API_KEY"
    wandb_project: str = "telugu-gemma-sft"
    wandb_entity: Optional[str] = None
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None
    hf_token: Optional[str] = None
    
    # Additional settings
    train_on_responses_only: bool = True
    seed: int = 42
    run_name: str = "full-finetune"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            # Model and dataset settings
            "model_name": self.model_name,
            "input_file": self.input_file,
            "output_dir": self.output_dir,
            "max_seq_length": self.max_seq_length,
            
            # Training settings
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "optimizer": self.optimizer,
            
            # Evaluation settings
            "eval_split": self.eval_split,
            "early_stopping_patience": self.early_stopping_patience,
            
            # Logging and saving settings
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            
            # Integration settings
            "use_wandb": self.use_wandb,
            "wandb_api_key": self.wandb_api_key,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "hf_token": self.hf_token,
            
            # Additional settings
            "train_on_responses_only": self.train_on_responses_only,
            "seed": self.seed,
            "run_name": self.run_name,
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> "TeluguFineTuningConfig":
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load the YAML file
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Create config with defaults
        config = cls()
        
        # Update config from file
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from file and return as dictionary"""
    config = TeluguFineTuningConfig.from_file(config_path)
    return config.to_dict()

if __name__ == "__main__":
    # When run directly, print the loaded configuration
    import json
    config = load_config()
    print(json.dumps(config, indent=2))
