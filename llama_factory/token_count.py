import datasets
from transformers import AutoTokenizer
import os
import math
from tqdm.auto import tqdm # For progress bar

# --- Configuration ---
# !! IMPORTANT: Replace with the EXACT Gemma model identifier you are fine-tuning !!
model_name = "google/gemma-3-12b-pt" # Or gemma-3-?b-it, etc.

# !! IMPORTANT: Set the correct paths to your dataset files !!
train_file = "./data/telugu/telugu_train.json"
val_file = "./data/telugu/telugu_val.json"

# Optional: specify a cache directory for the tokenizer
cache_dir = "./tokenizer_cache"

# --- Configuration End ---

# Ensure cache directory exists
os.makedirs(cache_dir, exist_ok=True)

print(f"Loading tokenizer for '{model_name}'...")
try:
    # Trust remote code if necessary for some tokenizers (like newer Gemma ones)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    print("Make sure you have transformers installed, are logged in if necessary (huggingface-cli login), and the model name is correct.")
    exit()

# Add EOS token if tokenizer doesn't add it automatically in encode
# Most tokenizers handle this correctly, but good to be aware
eos_token = tokenizer.eos_token if tokenizer.eos_token else ""

def format_example(example):
    """Formats the instruction, input, and output fields into a single string."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')

    # Simple formatting - adjust if your training script uses a specific prompt template
    if input_text and input_text.strip():
        # If input exists, include it clearly separated
         combined_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
    else:
        # If no input, format slightly differently
         combined_text = f"Instruction: {instruction}\nOutput: {output_text}"

    # Add EOS token - crucial for training the model to stop generating
    return combined_text + eos_token


def calculate_tokens(file_path, tokenizer, formatter_func):
    """Loads a JSON dataset, formats, tokenizes, and counts tokens."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return 0, 0

    print(f"\nProcessing file: {file_path}")
    try:
        # Load dataset assumes a list of json objects [{"instruction":...}, ...]
        dataset = datasets.load_dataset("json", data_files=file_path, split="train")
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        return 0, 0

    total_tokens = 0
    num_examples = 0

    # Use tqdm for progress bar
    for example in tqdm(dataset, desc=f"Tokenizing {os.path.basename(file_path)}"):
        # Format the text fields into one string
        formatted_text = formatter_func(example)

        # Tokenize without truncation to count all tokens
        tokens = tokenizer(formatted_text, truncation=False)["input_ids"]
        total_tokens += len(tokens)
        num_examples += 1

    return total_tokens, num_examples

# --- Run Calculation ---
train_total_tokens, train_num_examples = calculate_tokens(train_file, tokenizer, format_example)
val_total_tokens, val_num_examples = calculate_tokens(val_file, tokenizer, format_example)

# --- Report Results ---
print("\n" + "="*30)
print("Dataset Token Count Results")
print("="*30)
print(f"Training Set ({train_file}):")
print(f"  Number of examples: {train_num_examples:,}")
print(f"  Total tokens:       {train_total_tokens:,}")
if train_num_examples > 0:
    print(f"  Average tokens/example: {train_total_tokens / train_num_examples:.2f}")

print("\nValidation Set ({val_file}):")
print(f"  Number of examples: {val_num_examples:,}")
print(f"  Total tokens:       {val_total_tokens:,}")
if val_num_examples > 0:
    print(f"  Average tokens/example: {val_total_tokens / val_num_examples:.2f}")
print("="*30)

# --- How this relates to Hyperparameters ---

print("\nConnecting to Hyperparameters (using your previous config values as example):")

# Example config values (replace with your actual chosen values)
cfg = {
    "per_device_train_batch_size": 4, # As per your config (VERIFY VRAM!)
    "gradient_accumulation_steps": 4, # As per your config
    "num_train_epochs": 3.0,          # As per your config
    "cutoff_len": 4096,               # As per your config (max_seq_length)
    "warmup_ratio": 0.03              # As per your config
    # Assume number of GPUs, e.g., 1 for calculation. Adjust if using more.
}
num_gpus = 1 # !! IMPORTANT: Set the actual number of GPUs you will use !!

if train_num_examples > 0:
    # Calculate effective batch size
    effective_batch_size = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] * num_gpus
    print(f"\nEffective Batch Size = {cfg['per_device_train_batch_size']} (per_device) * {cfg['gradient_accumulation_steps']} (accum) * {num_gpus} (gpus) = {effective_batch_size}")

    # Calculate steps per epoch
    # Use ceil to ensure the last partial batch is processed
    steps_per_epoch = math.ceil(train_num_examples / effective_batch_size)
    print(f"Steps per Epoch ≈ {train_num_examples} (examples) / {effective_batch_size} (eff_batch_size) = {steps_per_epoch}")

    # Calculate total training steps
    total_training_steps = int(steps_per_epoch * cfg["num_train_epochs"])
    print(f"Total Training Steps ≈ {steps_per_epoch} (steps/epoch) * {cfg['num_train_epochs']} (epochs) = {total_training_steps}")

    # Calculate warmup steps
    warmup_steps = int(cfg["warmup_ratio"] * total_training_steps)
    print(f"Warmup Steps ≈ {cfg['warmup_ratio']} (ratio) * {total_training_steps} (total_steps) = {warmup_steps}")

    print("\nHow to use this information:")
    print(f"1. Total Steps ({total_training_steps}): Your LR scheduler needs this number to decay correctly over the training duration.")
    print(f"2. Warmup Steps ({warmup_steps}): Set this in your trainer configuration.")
    print(f"3. Logging/Saving/Eval Steps: Choose values that make sense relative to {total_training_steps}.")
    print(f"   - e.g., Log every 10-100 steps.")
    print(f"   - e.g., Save/Evaluate every {max(1, total_training_steps // 10)} steps (e.g., 10 times per training run), or based on time.")
    print(f"4. Sanity Check Effective Batch Size: Is {effective_batch_size} reasonable? If too small, consider increasing gradient_accumulation_steps (if VRAM allows) or using more GPUs.")
    print(f"5. VRAM Check: Remember to **verify** that `per_device_train_batch_size={cfg['per_device_train_batch_size']}` actually fits in your GPU memory with `cutoff_len={cfg['cutoff_len']}`!")
else:
    print("\nCould not calculate steps as train_num_examples is zero. Please check dataset loading.")