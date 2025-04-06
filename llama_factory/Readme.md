# Gemma-3-12B Telugu Fine-tuning Guide

This repository contains scripts and configuration files for fine-tuning the Gemma-3-12B model on Telugu language data. The fine-tuning is performed using [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), a unified and efficient framework for fine-tuning large language models.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Tips and Best Practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)

## Requirements

### Hardware Requirements

For fine-tuning Gemma-3-12B model:

- **LoRA/QLoRA**: At least 1x GPU with 48GB+ VRAM (e.g., RTX 3090, RTX 4090, A6000)
- **Full parameter fine-tuning**: Multiple GPUs with 80GB+ total VRAM (e.g., 2-4x A100 40GB GPUs)

### Software Requirements

- Python 3.9+
- PyTorch 2.0.0+
- Transformers 4.35.0+
- LLaMA Factory (install instructions below)
- For QLoRA: bitsandbytes 0.40.0+
- For faster training: Flash Attention 2

## Setup

1. **Clone LLaMA Factory repository**:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

2. **Install dependencies**:

```bash
pip install -e ".[torch,metrics]"
```

For QLoRA, add bitsandbytes:

```bash
pip install -e ".[torch,metrics,bitsandbytes]"
```

For Flash Attention 2 (highly recommended):

```bash
pip install flash-attn --no-build-isolation
```

3. **Copy the scripts and config files** from this repository to your LLaMA Factory directory.

4. **Install evaluation metrics**:

```bash
pip install rouge-score sacrebleu
```

## Data Preparation

The data preparation script converts your Telugu question-answer pairs from the input JSON format to the format required by LLaMA Factory.

1. **Format your data**:

Your input JSON file should have the format:
```json
{
  "questions": [
    {
      "question": "Telugu question here",
      "response": "Telugu answer here"
    },
    ...
  ]
}
```

2. **Run the data preparation script**:

```bash
python data-preparation.py --input_file your_telugu_data.json --output_dir ./data/telugu --val_split 0.1
```

This will create:
- `./data/telugu/telugu_train.json`: Training dataset
- `./data/telugu/telugu_val.json`: Validation dataset
- `./data/telugu/dataset_info.json`: Dataset configuration for LLaMA Factory

## Fine-tuning

This repository provides several fine-tuning approaches:

### Option 1: LoRA Fine-tuning (Recommended for most cases)

LoRA (Low-Rank Adaptation) fine-tunes only a small set of adapter parameters, which is memory-efficient and faster to train.

```bash
# Customize parameters in finetune.sh and then run:
bash finetune.sh
```

### Option 2: QLoRA Fine-tuning (For limited VRAM)

QLoRA combines quantization with LoRA, allowing fine-tuning on consumer GPUs with as little as 16GB VRAM.

```bash
# Edit finetune.sh to set:
# FINETUNING_TYPE="qlora"
# QUANT_BIT=4
bash finetune.sh
```

### Option 3: Full Parameter Fine-tuning (For best performance)

Full fine-tuning updates all model parameters, which generally yields the best performance but requires much more computational resources.

```bash
# Edit finetune.sh to set:
# FINETUNING_TYPE="full"
# USE_DEEPSPEED=true
# DEEPSPEED_STAGE=3
bash finetune.sh
```

## Evaluation

After fine-tuning, evaluate your model on a test set:

```bash
python eval.py \
  --base_model_path google/gemma-3-12b \
  --adapter_path ./gemma3-telugu-output \
  --test_file ./data/telugu/telugu_test.json \
  --output_dir ./evaluation_results \
  --batch_size 4 \
  --use_bf16
```

The script computes BLEU and ROUGE metrics and saves the results to `./evaluation_results/eval_results.json`.

## Inference

### Interactive Chat

You can chat with your fine-tuned model using the LLaMA Factory CLI:

```bash
# For LoRA models:
llamafactory-cli chat \
  --model_name_or_path google/gemma-3-12b \
  --adapter_name_or_path ./gemma3-telugu-output \
  --template gemma3

# For full fine-tuned models:
llamafactory-cli chat \
  --model_name_or_path ./gemma3-telugu-output \
  --template gemma3
```

### Web UI

LLaMA Factory also provides a web interface:

```bash
# For LoRA models:
llamafactory-cli webchat \
  --model_name_or_path google/gemma-3-12b \
  --adapter_name_or_path ./gemma3-telugu-output \
  --template gemma3

# For full fine-tuned models:
llamafactory-cli webchat \
  --model_name_or_path ./gemma3-telugu-output \
  --template gemma3
```

### Merging LoRA Adapters (Optional)

If you used LoRA or QLoRA, you can merge the adapter weights with the base model for faster inference:

```bash
llamafactory-cli export \
  --model_name_or_path google/gemma-3-12b \
  --adapter_name_or_path ./gemma3-telugu-output \
  --template gemma3 \
  --export_dir ./gemma3-telugu-merged
```

## Tips and Best Practices

1. **Start with LoRA**: Begin with LoRA fine-tuning to find good hyperparameters before moving to full fine-tuning.

2. **Optimal LoRA rank**: 
   - For small datasets (< 10k samples): Try ranks 8-16
   - For larger datasets: Try ranks 16-64

3. **Learning rate**:
   - LoRA/QLoRA: 1e-4 to 3e-4 works well
   - Full fine-tuning: 5e-6 to 2e-5 is recommended

4. **Batch size**: Use the largest batch size that fits in your GPU memory. If needed, decrease batch size and increase gradient accumulation steps to maintain an effective batch size of 32-64.

5. **Flash Attention 2**: Always enable Flash Attention 2 when possible, as it significantly speeds up training and reduces memory usage.

6. **BF16 precision**: Use BF16 instead of FP16 for Gemma model training, as it provides better numerical stability.

7. **Monitoring**: Enable W&B logging by uncommenting the relevant lines in the configuration file to monitor training progress.

## Troubleshooting

1. **Out of memory errors**: 
   - Try reducing batch size and increasing gradient accumulation steps
   - Switch to QLoRA with 4-bit quantization
   - Enable CPU offloading with DeepSpeed ZeRO-3

2. **Slow training**:
   - Enable Flash Attention 2
   - Use BF16 precision
   - Increase batch size and decrease gradient accumulation steps
   - Enable more GPUs if available

3. **Poor model quality**:
   - Increase training epochs
   - Try different learning rates
   - For LoRA, increase rank or try different target modules
   - For small datasets, use more prompt examples or improve data quality
   - Check if the model is overfitting (validation loss increases)

4. **Model generates English instead of Telugu**:
   - Add more Telugu examples
   - Add explicit instruction in system prompt to respond in Telugu
   - Set temperature lower (0.3-0.5) for more deterministic responses