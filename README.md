# LLM Fine-tuning Pipeline

This repository contains a comprehensive pipeline for fine-tuning Large Language Models (LLMs) using the Hugging Face Transformers library. The pipeline includes data preparation, model setup, and fine-tuning with best practices.

## Features

- Data preparation and validation
- Support for various input formats (JSON, CSV)
- Automatic dataset splitting (train/validation/test)
- Dataset statistics generation
- Efficient fine-tuning using LoRA (Low-Rank Adaptation)
- 4-bit quantization support for memory efficiency
- Weights & Biases integration for experiment tracking
- Checkpoint saving and resuming
- Comprehensive logging

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB GPU memory (for 7B models)
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Weights & Biases (optional but recommended):
```bash
wandb login
```

## Usage

### 1. Data Preparation

First, prepare your dataset using the `prepare_data.py` script:

```bash
python prepare_data.py
```

Edit the `main()` function in `prepare_data.py` to specify:
- Your input data file
- Output directory
- Text length constraints
- Dataset split ratios

The script will:
- Clean and validate your data
- Create train/validation/test splits
- Generate dataset statistics
- Save processed datasets in JSON format

### 2. Fine-tuning

After preparing your data, run the fine-tuning script:

```bash
python finetune_llm.py
```

Edit the `TrainingConfig` class in `finetune_llm.py` to specify:
- Model name (e.g., "meta-llama/Llama-2-7b-hf")
- Dataset name (from your processed data)
- Training parameters (epochs, batch size, learning rate, etc.)
- LoRA configuration
- Quantization settings

## Configuration

### Data Preparation Parameters

- `min_length`: Minimum text length (default: 50)
- `max_length`: Maximum text length (default: 2048)
- `train_ratio`: Training set ratio (default: 0.9)
- `val_ratio`: Validation set ratio (default: 0.05)
- `test_ratio`: Test set ratio (default: 0.05)

### Fine-tuning Parameters

- `model_name`: Base model to fine-tune
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Steps for gradient accumulation
- `learning_rate`: Learning rate for training
- `max_grad_norm`: Maximum gradient norm
- `warmup_ratio`: Warmup ratio for learning rate scheduler
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `lora_dropout`: LoRA dropout rate
- `use_4bit`: Whether to use 4-bit quantization

## Output

The pipeline generates:
1. Processed datasets in JSON format
2. Dataset statistics
3. Fine-tuned model checkpoints
4. Training logs and metrics (via Weights & Biases)

## Best Practices

1. **Data Quality**:
   - Ensure your dataset is clean and well-formatted
   - Remove duplicates and low-quality samples
   - Validate text length and content

2. **Model Selection**:
   - Choose a base model appropriate for your task
   - Consider model size vs. available computational resources
   - Use quantization when memory is limited

3. **Training**:
   - Start with a small learning rate (e.g., 2e-4)
   - Use gradient accumulation for larger effective batch sizes
   - Monitor training metrics to prevent overfitting
   - Save checkpoints regularly

4. **Evaluation**:
   - Use validation set to monitor training progress
   - Evaluate on test set after training
   - Consider using multiple evaluation metrics

## Troubleshooting

1. **Out of Memory (OOM) Errors**:
   - Reduce batch size
   - Enable 4-bit quantization
   - Use gradient accumulation
   - Consider using a smaller model

2. **Training Instability**:
   - Adjust learning rate
   - Modify warmup ratio
   - Check gradient clipping
   - Verify data quality

3. **Poor Performance**:
   - Check data quality and preprocessing
   - Adjust model hyperparameters
   - Consider using a different base model
   - Verify training data distribution

## Contributing

Feel free to submit issues and enhancement requests! 