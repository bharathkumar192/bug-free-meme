import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import yaml

@dataclass
class DataConfig:
    """Configuration for dataset preparation and processing"""
    input_file: str = "telugu_results.json"  # Path to raw dataset
    output_dir: str = "processed_data"  # Directory for processed data
    min_length: int = 50  # Minimum text length for validation
    max_length: int = 2048  # Maximum text length for validation
    train_ratio: float = 0.9  # Ratio for training split
    val_ratio: float = 0.05  # Ratio for validation split
    test_ratio: float = 0.05  # Ratio for test split

@dataclass
class ModelConfig:
    """Configuration for model selection and parameters"""
    model_name: str = "google/gemma-3-12b-pt"  # Base model to fine-tune
    use_chat_template: bool = True  # Whether to use chat formatting
    chat_template: Optional[str] = None  # Custom chat template (if specified)

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    num_train_epochs: int = 3  # Number of training epochs
    per_device_train_batch_size: int = 2  # Batch size for training
    per_device_eval_batch_size: int = 2  # Batch size for evaluation
    gradient_accumulation_steps: int = 16  # Steps for gradient accumulation
    learning_rate: float = 2e-6  # Training learning rate
    weight_decay: float = 0.01  # Weight decay for regularization
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    
    # Scheduler settings
    warmup_ratio: float = 0.05  # Ratio of warmup steps
    lr_scheduler_type: str = "cosine"  # Learning rate scheduler type
    
    # Sequence length
    max_seq_length: int = 2048  # Maximum sequence length for tokenization
    
    # Optimization settings
    fp16: bool = True  # Use FP16 precision
    bf16: bool = False  # Use BF16 precision (A100, H100 GPUs)
    gradient_checkpointing: bool = True  # Use gradient checkpointing
    
    # Early stopping
    early_stopping_patience: int = 3  # Patience for early stopping
    
    # DeepSpeed integration
    use_deepspeed: bool = False  # Whether to use DeepSpeed
    deepspeed_config_path: Optional[str] = None  # Path to DeepSpeed config
    
    # PEFT/LoRA parameters (for parameter-efficient fine-tuning)
    use_peft: bool = False  # Whether to use PEFT
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.05  # LoRA dropout
    
    # Quantization settings
    use_4bit: bool = False  # Use 4-bit quantization
    bnb_4bit_compute_dtype: str = "float16"  # Compute dtype for 4-bit quantization

@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing"""
    output_dir: str = "finetuned_model"  # Directory for model output
    logging_steps: int = 10  # Steps between logging
    save_steps: int = 100  # Steps between saving checkpoints
    eval_steps: int = 100  # Steps between evaluations
    save_total_limit: int = 5  # Maximum number of checkpoints to keep
    report_to: List[str] = field(default_factory=lambda: ["wandb", "tensorboard"])  # Tracking platforms

@dataclass
class IntegrationConfig:
    """Configuration for third-party integrations"""
    # Weights & Biases
    wandb_api_key: Optional[str] = None  # Weights & Biases API key
    wandb_project: str = "telugu-gemma-sft"  # W&B project name
    wandb_entity: Optional[str] = None  # W&B entity/organization
    
    # HuggingFace Hub
    hf_token: Optional[str] = None  # HuggingFace API token
    push_to_hub: bool = False  # Whether to push to HuggingFace Hub
    hub_model_id: Optional[str] = None  # HF Hub model ID (username/model-name)
    
    # Seeds
    seed: int = 42  # Random seed for reproducibility

@dataclass
class UnifiedConfig:
    """Unified configuration for the entire fine-tuning pipeline"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "UnifiedConfig":
        """Load configuration from file (YAML or JSON)"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        # Create configuration instance with defaults
        config = cls()
        
        # Update with values from file
        if "data" in config_dict:
            for k, v in config_dict["data"].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)
        
        if "model" in config_dict:
            for k, v in config_dict["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        
        if "training" in config_dict:
            for k, v in config_dict["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        
        if "logging" in config_dict:
            for k, v in config_dict["logging"].items():
                if hasattr(config.logging, k):
                    setattr(config.logging, k, v)
        
        if "integration" in config_dict:
            for k, v in config_dict["integration"].items():
                if hasattr(config.integration, k):
                    setattr(config.integration, k, v)
        
        return config
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to file (YAML or JSON)"""
        config_path = Path(config_path)
        
        # Convert dataclasses to dictionaries
        config_dict = {
            "data": {k: v for k, v in vars(self.data).items()},
            "model": {k: v for k, v in vars(self.model).items()},
            "training": {k: v for k, v in vars(self.training).items()},
            "logging": {k: v for k, v in vars(self.logging).items()},
            "integration": {k: v for k, v in vars(self.integration).items()}
        }
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    def get_training_args(self):
        """Get HuggingFace TrainingArguments from config"""
        from transformers import TrainingArguments
        
        # Prepare DeepSpeed config if needed
        deepspeed = None
        if self.training.use_deepspeed and self.training.deepspeed_config_path:
            if os.path.exists(self.training.deepspeed_config_path):
                deepspeed = self.training.deepspeed_config_path
        
        return TrainingArguments(
            output_dir=self.logging.output_dir,
            num_train_epochs=self.training.num_train_epochs,
            per_device_train_batch_size=self.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training.gradient_accumulation_steps,
            learning_rate=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
            max_grad_norm=self.training.max_grad_norm,
            warmup_ratio=self.training.warmup_ratio,
            lr_scheduler_type=self.training.lr_scheduler_type,
            logging_steps=self.logging.logging_steps,
            save_steps=self.logging.save_steps,
            eval_steps=self.logging.eval_steps,
            save_total_limit=self.logging.save_total_limit,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to=self.logging.report_to,
            fp16=self.training.fp16,
            bf16=self.training.bf16,
            deepspeed=deepspeed,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_torch", 
            ddp_find_unused_parameters=False,
            group_by_length=True,
            hub_model_id=self.integration.hub_model_id if self.integration.push_to_hub else None,
            push_to_hub=self.integration.push_to_hub,
            hub_token=self.integration.hf_token,
            seed=self.integration.seed,
        )