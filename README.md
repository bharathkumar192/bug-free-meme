# Telugu Gemma-3. **Training crashes or hangs**:
   - Check your GPU driver compatibility
   - Ensure you have enough disk space for checkpoints
   - Look at the log files for specific error messages

4. **Poor quality responses**:
   - Increase the number of training epochs
   - Check dataset quality and ensure proper preprocessing
   - Adjust learning rate or use a learning rate finder
   - Verify the chat template is correctly applied

## Additional Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Gemma-3 Model Card](https://huggingface.co/google/gemma-3-12b-pt)
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [HuggingFace SFT Tutorial](https://huggingface.co/docs/trl/sft_trainer)

## Performance Optimization Tips

1. **Flash Attention**: Enable Flash Attention for faster training with these environment variables:
   ```bash
   export TRANSFORMERS_ATTN_IMPLEMENTATION="flash_attention_2"
   ```

2. **Memory Management**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

3. **Multi-GPU Training**:
   - Enable DeepSpeed for multi-GPU training if needed
   - Adjust parameters for distributed training

## Acknowledgements

- Google for releasing the Gemma-3 model family
- Unsloth for optimization techniques
- HuggingFace for their transformers and TRL libraries

## License

This code is provided under [your license of choice]. The underlying Gemma-3 model is subject to Google's model license. Fine-tuning

This repository contains scripts for fine-tuning Google's Gemma-3-12b model on Telugu question-answer pairs to get natural-toned responses. The implementation uses Unsloth for efficient full parameter fine-tuning, avoiding memory issues even with large models like Gemma-3-12b.

## Features

- Full parameter fine-tuning of Gemma-3-12b (no PEFT/LoRA quantization)
- Optimized for high-memory GPUs (40GB, 48GB, 64GB, 80GB)
- Unsloth integration for memory and speed optimization
- Chat template formatting for question-answer pairs
- Weights & Biases integration for tracking experiments
- HuggingFace Hub integration for saving models
- Interactive testing interface

## Requirements

- Python 3.8+
- CUDA-capable GPU with at least 40GB VRAM (80GB recommended for Gemma-3-12b)
- Internet connection for downloading the base model

## Project Structure

```
.
├── config.yaml                 # Configuration file
├── config_loader.py            # Configuration loader utility
├── finetune_gemma3_telugu.py   # Main fine-tuning script
├── inference.py                # Script for testing the fine-tuned model
├── run.sh                      # Setup and execution script
└── telugu_results.json         # Your Telugu question-answer dataset
```

## Dataset Format

The dataset should be in the following JSON format:

```json
{
  "questions": [
    {
      "question": "ఇండియాలో గ్రోసరీస్ మీద డబ్బులు సేవ్ చేయడానికి బెస్ట్ వేస్ ఏంటి?",
      "response": "ఇండియాలో గ్రోసరీస్ ..."
    },
    ...
  ]
}
```

## Setup and Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/telugu-gemma3-finetuning.git
cd telugu-gemma3-finetuning
```

2. Run the setup script to install dependencies and prepare the environment:

```bash
chmod +x run.sh
./run.sh
```

3. Update the configuration in `config.yaml` to match your needs.

## Configuration

Key configuration parameters in `config.yaml`:

```yaml
# Model and dataset settings
model_name: "google/gemma-3-12b-pt"
input_file: "telugu_results.json"
output_dir: "finetuned_gemma3_telugu"

# Training settings
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1e-5

# Integration settings
use_wandb: true
wandb_api_key: "YOUR_WANDB_API_KEY"
push_to_hub: true
hub_model_id: "your-username/gemma-3-12b-telugu"
hf_token: "YOUR_HF_TOKEN"
```

## Fine-tuning

To start the fine-tuning process:

```bash
python finetune_gemma3_telugu.py --config config.yaml
```

This will:
1. Load your Telugu dataset
2. Initialize the Gemma-3-12b model
3. Setup full parameter fine-tuning 
4. Train for the specified number of epochs
5. Save the model locally and (optionally) push to HuggingFace Hub

### Memory Optimization

The default configuration uses:
- BFloat16 precision for memory efficiency and training stability
- Gradient checkpointing to reduce memory usage
- Optimized batch sizes and gradient accumulation steps
- Unsloth for faster training and more efficient memory usage

## Testing the Fine-tuned Model

Once fine-tuning is complete, you can evaluate the model:

```bash
# Test with sample questions from a file
python inference.py --model_path finetuned_gemma3_telugu --questions_file test_questions.json

# Interactive mode
python inference.py --model_path finetuned_gemma3_telugu --interactive
```

## Memory Requirements

- **Gemma-3-12b**: 80GB VRAM recommended for full parameter fine-tuning
- For smaller GPUs (40-48GB), consider:
  - Reducing batch size and increasing gradient_accumulation_steps
  - Enabling 8-bit quantization
  - Using PEFT/LoRA (not covered in this implementation, as per requirements)

## Troubleshooting

Common issues and solutions:

1. **Out of Memory errors**:
   - Reduce batch size
   - Increase gradient accumulation steps
   - Enable gradient checkpointing (already enabled by default)

2. **Slow training**:
   - Ensure you're using a GPU with CUDA support
   - Check that BFloat16 precision is enabled
   - Verify Unsloth is properly installed and configured

3
