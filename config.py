import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import json
from pathlib import Path

@dataclass
class DataConfig:
    input_file: str = "telugu_results.json"
    output_dir: str = "processed_data"
    min_length: int = 50
    max_length: int = 2048
    data_percentage: float = 0.3  # Use 60% of data
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

@dataclass
class ModelConfig:
    model_name: str = "google/gemma-3-12b-pt"
    use_chat_template: bool = True
    chat_template: str = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<start_of_turn>user\n{{ message['content'] }}\n<end_of_turn>\n{% elif message['role'] == 'assistant' %}\n<start_of_turn>model\n{{ message['content'] }}\n<end_of_turn>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<start_of_turn>model\n{% endif %}"

@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    
    # Memory optimization
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    # deepspeed: str = "ds_config_zero3.json"
    deepspeed: str = None
    
    # PEFT/LoRA settings
    use_peft: bool = False  # Enabled PEFT for memory efficiency
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Quantization
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # Additional
    early_stopping_patience: int = 3
    max_grad_norm: float = 1.0

@dataclass
class LoggingConfig:
    output_dir: str = "finetuned_model"
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 5
    report_to: List[str] = field(default_factory=lambda: ["wandb", "tensorboard"])

@dataclass
class IntegrationConfig:
    wandb_api_key: str = "bc746fafa585f61730cdfd3c8226492c777f875a"  # Replace with your key
    wandb_project: str = "telugu-gemma-sft"
    wandb_entity: Optional[str] = None
    
    hf_token: str = "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"  # Replace with your token
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None
    
    seed: int = 42

@dataclass
class UnifiedConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "UnifiedConfig":
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load the file based on extension
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Create config with defaults
        config = cls()
        
        # Update sections from file
        sections = ["data", "model", "training", "logging", "integration"]
        for section in sections:
            if section in config_dict:
                section_obj = getattr(config, section)
                for key, value in config_dict[section].items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        return config
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to file"""
        config_path = Path(config_path)
        
        # Convert to dict
        config_dict = {
            "data": {k: v for k, v in vars(self.data).items()},
            "model": {k: v for k, v in vars(self.model).items()},
            "training": {k: v for k, v in vars(self.training).items()},
            "logging": {k: v for k, v in vars(self.logging).items()},
            "integration": {k: v for k, v in vars(self.integration).items()}
        }
        
        # Create directories if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on extension
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def get_training_args(self):
        """Get HuggingFace TrainingArguments from config"""
        from transformers import TrainingArguments
        
        # Check deepspeed config exists
        deepspeed = None
        if hasattr(self.training, "deepspeed") and self.training.deepspeed:
            if os.path.exists(self.training.deepspeed):
                deepspeed = self.training.deepspeed
        
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


# Generate default config file if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate default configuration file")
    parser.add_argument("--output", type=str, default="config.yaml", help="Output config file path")
    args = parser.parse_args()
    
    # Create default config
    config = UnifiedConfig()
    
    # Set recommended values for training Gemma-3-12B
    config.training.per_device_train_batch_size = 2
    config.training.gradient_accumulation_steps = 32
    # config.training.use_peft = True
    # config.training.use_4bit = True
    config.training.max_seq_length = 2048
    
    # Save to file
    config.to_file(args.output)
    print(f"Default configuration saved to {args.output}")