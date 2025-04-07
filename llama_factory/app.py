import modal
import os
import subprocess
from pathlib import Path
import shutil
import json  # Added missing import

# Define the Modal image with all necessary dependencies
# (Including 'packaging' before 'flash-attn', and other necessary libs)
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install("llamafactory")  # Install LLaMA Factory directly
    # Add HF_TOKEN secret if you need private repo access or plan to push
    .env({"HF_TOKEN": "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"})
    # Add WANDB_API_KEY secret if using W&B
    .env({"WANDB_API_KEY": "bc746fafa585f61730cdfd3c8226492c777f875a"})
)

# Create volumes for caching models and storing training artifacts
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
training_vol = modal.Volume.from_name("gemma-telugu-training", create_if_missing=True)

# Create the Modal app
app = modal.App("gemma-telugu-finetuning", image=image)

@app.function(
    gpu="A100-80GB:2",  # Request 2 A100 80GB GPUs
    timeout=60*60*24,   # 24-hour timeout for long training
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/training": training_vol  # Mount the training volume at /training
    },
    memory=32384, # Increased memory
)
def run_finetune_script():
    """Checks files, sets up, and runs the finetune.sh script directly on Modal."""

    # Define the base directory where files are mounted/expected
    base_training_dir = Path("/training")

    # Define relative paths of required files within the base_training_dir
    required_files = [
        "config.py",
        "finetune.py",
        "finetune.sh",
        "eval.py",
        "data/telugu_train.json", # Adjusted path assuming data is directly in /training/data
        "data/telugu_val.json",   # Adjusted path
        "data/dataset_info.json"  # Adjusted path
    ]

    # --- Corrected File Checking Logic ---
    print(f"Checking for required files within {base_training_dir}...")
    missing_files = []
    for relative_file_path in required_files:
        # Construct the absolute path within the container's volume mount
        absolute_file_path = base_training_dir / relative_file_path
        if not absolute_file_path.exists():
            missing_files.append(str(absolute_file_path)) # Log the expected absolute path

    if missing_files:
        print(f"Error: Missing required files in the {base_training_dir} volume mount:")
        for mf in missing_files:
            print(f"- {mf}")
        print("\nPlease ensure these files were uploaded correctly using the --upload-files flag in 'modal run'.")
        return {"status": "error", "message": "Missing required files"}
    else:
        print("All required files found.")
    print(f"Changing working directory to {base_training_dir}")
    try:
        os.chdir(base_training_dir)
    except FileNotFoundError:
        print(f"Error: Could not change directory to {base_training_dir}. Check volume mount.")
        return {"status": "error", "message": f"Failed to change directory to {base_training_dir}"}

    # Now that we are in /training, we can use relative paths for subprocess calls
    finetune_script_relative = "./finetune.sh" # Path relative to the new CWD (/training)

    # Make the shell script executable
    print(f"Making {finetune_script_relative} executable...")
    try:
        # Use check=True to raise error if chmod fails
        subprocess.run(["chmod", "+x", finetune_script_relative], check=True, capture_output=True, text=True)
        print(f"Successfully ran chmod on {finetune_script_relative}")
    except subprocess.CalledProcessError as e:
        print(f"Error running chmod on {finetune_script_relative}:")
        print(f"Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        return {"status": "error", "message": f"chmod failed: {e.stderr}"}
    except FileNotFoundError:
         print(f"Error: Cannot find {finetune_script_relative} in current directory ({os.getcwd()}) for chmod.")
         return {"status": "error", "message": f"chmod failed: {finetune_script_relative} not found."}


    # Set environment variables for distributed training (if needed)
    # Check if FORCE_TORCHRUN is needed by llamafactory/your setup when using deepspeed
    os.environ["FORCE_TORCHRUN"] = "1"
    print("Set FORCE_TORCHRUN=1 (Note: Verify if needed for your LlamaFactory version)")

    # Run the finetune.sh script
    # IMPORTANT: Ensure finetune.sh does NOT have interactive prompts (read -p)
    print(f"Starting fine-tuning process by running {finetune_script_relative}...")
    try:
        # Using subprocess.run to execute the shell script
        # capture_output=True helps debug by saving stdout/stderr
        process = subprocess.run([finetune_script_relative], check=True, capture_output=True, text=True, env=os.environ.copy())
        print("Fine-tuning script finished successfully.")
        print("--- Script stdout: ---")
        print(process.stdout)
        # Check stderr even on success for warnings
        if process.stderr:
             print("--- Script stderr (Warnings/Info): ---")
             print(process.stderr)
        # Define expected output directory relative to the current dir (/training)
        output_dir = base_training_dir / "gemma3-telugu-output"
        return {"status": "success", "output_dir": str(output_dir)}
    except subprocess.CalledProcessError as e:
        print(f"Error: Fine-tuning script {finetune_script_relative} failed with exit code {e.returncode}.")
        print("--- Script stderr: ---")
        print(e.stderr)
        print("--- Script stdout: ---")
        print(e.stdout)
        output_dir = base_training_dir / "gemma3-telugu-output"
        return {"status": "error", "message": f"Fine-tuning script failed. Stderr: {e.stderr}", "output_dir_attempted": str(output_dir)}
    except FileNotFoundError:
         print(f"Error: Cannot find {finetune_script_relative} in current directory ({os.getcwd()}) to execute.")
         return {"status": "error", "message": f"Execution failed: {finetune_script_relative} not found."}


# Added the evaluate_model function back (it was missing in your last snippet)
@app.function(
    gpu="A100-80GB:1", # Use 1 GPU for evaluation
    timeout=60*15, # 15 min timeout for evaluation
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/training": training_vol
    },
    image=image # Use the same image
)
def evaluate_model(
    model_output_dir: str = "/training/gemma3-telugu-output", # Pass the model dir
    test_sample: str = "తెలుగులో ఏదైనా ప్రశ్న అడగండి."
    ):
    """Loads the fine-tuned model and evaluates it on a sample prompt."""

    # Define base directory within the container
    base_training_dir = Path("/training")
    # Construct absolute path to the model directory
    absolute_model_dir = Path(model_output_dir) # Already absolute if passed as /training/...

    # Check if the required eval script exists
    eval_script_path = base_training_dir / "eval.py"
    if not eval_script_path.exists():
         print(f"Error: Evaluation script not found at {eval_script_path}")
         return {"status": "error", "message": "eval.py not found"}

    # Check if the specified model directory exists
    if not absolute_model_dir.exists() or not absolute_model_dir.is_dir():
        print(f"Error: Model directory not found at {absolute_model_dir}")
        # Attempt to list contents of /training for debugging
        try:
            print(f"Contents of {base_training_dir}: {os.listdir(base_training_dir)}")
            parent_dir = absolute_model_dir.parent
            if parent_dir.exists():
                 print(f"Contents of {parent_dir}: {os.listdir(parent_dir)}")
        except Exception as list_e:
            print(f"Could not list directory contents: {list_e}")
        return {"status": "error", "message": f"Model directory {absolute_model_dir} not found"}

    # Dynamically import the evaluate functions from eval.py within the container
    # Need to add the /training directory to Python's path
    sys.path.append(str(base_training_dir))
    try:
        from eval import load_model_and_tokenizer, format_prompt, generate_responses
        print("Successfully imported functions from eval.py")
    except ImportError as import_err:
        print(f"Error importing from eval.py: {import_err}")
        print(f"Python sys.path: {sys.path}")
        return {"status": "error", "message": f"Failed to import from eval.py: {import_err}"}

    # Simulate command-line arguments for the eval script functions
    class EvalArgs:
        def __init__(self):
            # Assume full fine-tuning, so we load the merged path
            self.merged_model_path = str(absolute_model_dir)
            # If you used LoRA and didn't merge, you'd need base_model_path and adapter_path
            self.base_model_path = "google/gemma-3-12b-pt" # Base model used for tokenizer primarily if merged
            self.adapter_path = None # Set to adapter path if using LoRA without merge
            self.max_new_tokens = 512
            self.temperature = 0.7
            self.top_p = 0.9
            self.top_k = 50
            self.num_beams = 1
            self.batch_size = 1 # Keep batch size 1 for single sample eval
            self.use_bf16 = True # Match training precision if possible
            self.use_fp16 = False
            self.use_4bit = False # Quantization not used here, adjust if needed
            self.use_8bit = False
            self.seed = 42

    args = EvalArgs()

    # Load model and tokenizer
    print(f"Loading model and tokenizer from {args.merged_model_path}...")
    try:
        model, tokenizer = load_model_and_tokenizer(args)
        print("Model and tokenizer loaded.")
    except Exception as load_err:
        print(f"Error loading model/tokenizer: {load_err}")
        return {"status": "error", "message": f"Failed to load model/tokenizer: {load_err}"}

    # Prepare test sample (assuming eval.py expects a list of dicts)
    # Adjust if your eval script has different input format
    test_data = [{"question": test_sample, "response": ""}] # Reference response is empty for generation

    # Generate response
    print("Generating response...")
    try:
        results = generate_responses(model, tokenizer, test_data, args)
        print("Response generation complete.")
        if not results or "generated" not in results[0]:
             print("Error: Generation result format unexpected.")
             return {"status": "error", "message": "Unexpected generation result format"}

        return {
            "status": "success",
            "prompt": test_sample,
            "response": results[0]["generated"]
        }
    except Exception as gen_err:
        print(f"Error during response generation: {gen_err}")
        return {"status": "error", "message": f"Failed during generation: {gen_err}"}



@app.function(
    volumes={"/training": training_vol}, # Mount the volume to this function too
    timeout=600 # 10 min timeout for uploads
)
def upload_to_volume(local_path_str: str):
    """
    Uploads contents of a local directory to the root of the training_vol.
    This function runs inside the Modal container environment.
    """
    print(f"Inside upload_to_volume: Attempting to upload from {local_path_str}")
    local_path = Path(local_path_str)
    remote_target_path = "/" # Target root of the volume mount

    if not local_path.exists() or not local_path.is_dir():
        print(f"Error: Local path '{local_path}' does not exist or is not a directory.")
        raise ValueError(f"Invalid local path provided for upload: {local_path}")

    try:
        # This should work when run inside the Modal function context
        training_vol.add_local_dir(local_path=local_path, remote_path=remote_target_path)
        print(f"Successfully uploaded contents of {local_path} to volume path {remote_target_path}")
        # Force a commit/save of the volume changes
        training_vol.commit()
        print("Volume commit successful.")
    except AttributeError as e:
         print(f"AttributeError during upload: {e}. This suggests 'add_local_dir' is still not available.")
         # Fallback attempt or re-raise
         raise e
    except Exception as e:
        print(f"An error occurred during upload_to_volume: {e}")
        # Re-raise the exception to signal failure
        raise e




@app.local_entrypoint()
def main(upload_files: bool = False, files_dir: str = None):
    """Main entry point for the Modal application. Handles file uploads and triggers training."""

    # Option to upload files to the volume before starting the run
    if upload_files:
        if not files_dir:
            print("Error: --files-dir must be specified when --upload-files is used.")
            return
        local_files_path = Path(files_dir).resolve() # Resolve to absolute path
        if not local_files_path.exists() or not local_files_path.is_dir():
            print(f"Error: Local directory specified by --files-dir does not exist or is not a directory: {local_files_path}")
            return

        print(f"Uploading contents of local directory '{local_files_path}' to root of Modal Volume 'gemma-telugu-training'...")

        try:
            # Use add_local_dir to upload the directory contents
            # The second argument is the target path *inside* the volume.
            # "/" means upload contents into the root of the volume (/training when mounted)
            upload_to_volume.remote(local_path_str=str(local_files_path))
            print("Remote upload function executed.")
            print("-" * 30)
        except Exception as upload_err:
            print(f"Error during file upload using add_local_dir: {upload_err}")
            print("Aborting.")
            return

    # Trigger the remote fine-tuning function
    print("Starting remote fine-tuning function...")
    # Assuming run_finetune_script is defined above with the @app.function decorator
    result = run_finetune_script.remote() # Call the remote function

    print("\n--- Fine-tuning Result ---")
    print(f"Status: {result.get('status', 'Unknown')}")
    if result.get('status') == "success":
        print(f"Fine-tuning completed successfully!")
        output_dir = result.get('output_dir', 'N/A')
        print(f"Model output directory in volume: {output_dir}")

        # Optional: Automatically trigger evaluation after successful training
        # print("\n--- Starting Evaluation ---")
        # # Assuming evaluate_model is defined above with the @app.function decorator
        # eval_result = evaluate_model.remote(model_output_dir=output_dir)
        # print(f"Evaluation Status: {eval_result.get('status', 'Unknown')}")
        # if eval_result.get('status') == 'success':
        #      print(f"Prompt: {eval_result.get('prompt')}")
        #      print(f"Response: {eval_result.get('response')}")
        # else:
        #      print(f"Evaluation Failed: {eval_result.get('message', 'No details')}")

    else:
        # Handle cases where run_finetune_script returned an error status or failed unexpectedly
        error_message = result.get('message', 'No error message provided.')
        print(f"Fine-tuning failed: {error_message}")
        output_dir_attempted = result.get('output_dir_attempted', 'N/A')
        if output_dir_attempted != 'N/A':
            print(f"(Attempted output directory was: {output_dir_attempted})")

    print("--- Run Finished ---")