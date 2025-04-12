import modal
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define CUDA tag for the Modal image
tag = "12.4.0-devel-ubuntu20.04"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.9")
    .apt_install(
        "python3", 
        "python3-pip", 
        "python3-dev",
        "build-essential",
        "wget",
        # "git"
    )
    # Install PyTorch and transformers
    .run_commands("pip install torch torchvision torchaudio packaging ninja wheel")
    .run_commands("pip install transformers accelerate peft datasets tqdm huggingface_hub")
    # Add HF_TOKEN
    .env({"HF_TOKEN": "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"})
    .add_local_dir("/Users/yuvasaiveeravalli/Downloads/bug-free-meme/llama_factory/", remote_path="/training", copy=True)
)

# Create volumes for caching models and storing inference results
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
inference_vol = modal.Volume.from_name("gemma-telugu-inference-more", create_if_missing=True)

# Create the Modal app
app = modal.App("gemma-telugu-test", image=image)

@app.function(
    gpu="H100",  
    timeout=60*60*24,   # 24-hour timeout for long training
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/output": inference_vol,
    },
    memory=64384,
)
def run_inference(
    model_path: str,
    prompts_file: str,
    output_dir: str = "",
    use_bf16: bool = True,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    push_to_hf: bool = True,
    repo_id: str = None,
    subfolder: str = "inference_results",
    append_results: bool = True
):
    """Run inference on the fine-tuned model with Telugu prompts and optionally push to HF."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    import time
    from pathlib import Path
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load prompts
    logger.info(f"Loading prompts from {prompts_file}")
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    logger.info(f"Loaded {len(questions)} questions from prompts file")
    
    # Determine data type
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Ensure tokenizer has padding settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager" 
    )
    
    # Set model to evaluation mode
    model.eval()
    logger.info("Model loaded successfully")
    
    # Format prompt according to Gemma's template used during fine-tuning
    def format_prompt(question: str) -> str:
        # Following the Gemma3 template format used during training
        prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        return prompt
    
    # Process all questions and generate responses
    results = []
    start_time = time.time()
    
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
        
        # Format the prompt
        prompt = format_prompt(question)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        question_start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=(temperature > 0),
                eos_token_id=tokenizer.convert_tokens_to_ids(["<end_of_turn>"])[0]  # Use appropriate end token
            )
        question_time = time.time() - question_start_time
        
        # Decode the generated response
        generated_text = tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clean up response (remove trailing model tokens if any)
        if "<end_of_turn>" in generated_text:
            generated_text = generated_text.split("<end_of_turn>")[0].strip()
        
        # Store the result
        results.append({
            "question": question,
            "response": generated_text,
            "generation_time_seconds": question_time
        })
        
        logger.info(f"Response: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
        logger.info(f"Generation time: {question_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Prepare the output data
    output_data = {
        "model": model_path,
        "total_questions": len(questions),
        "total_time_seconds": total_time,
        "average_time_per_question": total_time / len(questions),
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "data_type": "bfloat16" if use_bf16 else "float16"
        },
        "results": results
    }
    
    # Save results to file - ensure this is using the correct path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"telugu_responses.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_file}")


    # After saving results, if push_to_hf is True, push to Hugging Face
    hf_result = None
    if push_to_hf and repo_id:
        logger.info("\n--- Pushing Results to Hugging Face ---")
        try:
            hf_result = push_results_to_huggingface_internal(
                repo_id=repo_id,
                json_file=output_file,
                # readable_file=readable_output_file,
                subfolder=subfolder,
                append_results=append_results
            )
            
            # Log push results
            logger.info("\n--- HuggingFace Push Results ---")
            logger.info(f"Push result status: {hf_result.get('status', 'unknown')}")
            if hf_result.get("status") == "success":
                logger.info(f"Results successfully pushed to: {hf_result['repo_url']}")
                logger.info(f"Files pushed: {', '.join(hf_result['files_pushed'])}")
            else:
                logger.error(f"Failed to push results: {hf_result.get('error', 'unknown error')}")
        except Exception as e:
            logger.error(f"Error pushing to Hugging Face: {str(e)}")
            hf_result = {"status": "error", "error": str(e)}
    
    
    # Format and display a summary of the results in a table format
    print_output_summary(results)
    
    # Also save a more readable format
    # readable_output_file = os.path.join(output_dir, f"telugu_responses_readable_{timestamp}.txt")
    # save_readable_output(results, readable_output_file)
    
    # logger.info(f"Readable results saved to {readable_output_file}")
    
    # Verify files exist before returning
    if not os.path.exists(output_file):
        logger.error(f"ERROR: Output file not created at {output_file}")
    else:
        logger.info(f"Verified output file exists at {output_file}")
    
    # if not os.path.exists(readable_output_file):
    #     logger.error(f"ERROR: Readable output file not created at {readable_output_file}")
    # else:
    #     logger.info(f"Verified readable output file exists at {readable_output_file}")
    
    # Return the results along with the file paths
    return {
        "data": output_data,
        "json_file": "telugu_responses.json",
        # "readable_file": readable_output_file,
        "timestamp": timestamp
    }

def print_output_summary(results):
    """Print a nicely formatted summary of the results."""
    print("\n" + "="*100)
    print(f"{'QUESTION':<50} | {'RESPONSE (PREVIEW)':<50}")
    print("-"*100)
    
    for item in results:
        # Get first 50 chars of question and response for preview
        question_preview = item["question"][:47] + "..." if len(item["question"]) > 50 else item["question"]
        response_preview = item["response"][:47] + "..." if len(item["response"]) > 50 else item["response"]
        
        print(f"{question_preview:<50} | {response_preview:<50}")
    
    print("="*100)

def save_readable_output(results, output_file):
    """Save results in a more human-readable format."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("TELUGU MODEL INFERENCE RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, item in enumerate(results, 1):
            f.write(f"QUESTION {i}: {item['question']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"RESPONSE: {item['response']}\n")
            f.write(f"GENERATION TIME: {item['generation_time_seconds']:.2f} seconds\n")
            f.write("\n" + "=" * 80 + "\n\n")

def push_results_to_huggingface_internal(
    repo_id: str,
    json_file: str,
    # readable_file: str,
    subfolder: str = "inference_results",
    commit_message: str = "Add model inference results",
    append_results: bool = True
):
    """Internal function to push inference results to a Hugging Face repository."""
    from huggingface_hub import HfApi, hf_hub_download
    import os
    import json
    from pathlib import Path
    import shutil
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Check if files exist
    # logger.info(f"Checking if files exist: {json_file} and {readable_file}")
    # if not os.path.exists(json_file):
    #     logger.error(f"JSON file not found: {json_file}")
    #     raise FileNotFoundError(f"Cannot find JSON file: {json_file}")
    
    # if not os.path.exists(readable_file):
    #     logger.error(f"Readable file not found: {readable_file}")
    #     raise FileNotFoundError(f"Cannot find readable file: {readable_file}")
    
    # Prepare the paths
    json_path = Path(json_file)
    # readable_path = Path(readable_file)
    
    logger.info(f"Pushing results to Hugging Face repository: {repo_id}")
    # logger.info(f"Files to push: {json_path.name} and {readable_path.name}")
    
    # If we're appending results, try to download existing files first
    all_results_path = None
    if append_results:
        try:
            # Try to download the existing results file using hf_hub_download
            logger.info(f"Attempting to download existing results file from {repo_id}/{subfolder}")
            existing_json_file = f"{subfolder}/all_results.json" if subfolder else "all_results.json"
            
            try:
                # Use hf_hub_download instead of api.download_file
                local_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=existing_json_file,
                    repo_type="model",
                    local_dir="/output"
                )
                
                logger.info(f"Downloaded existing file to: {local_file}")
                
                # Copy to a consistent location if needed
                all_results_path = os.path.join(os.path.dirname(json_file), "all_results.json")
                shutil.copy(local_file, all_results_path)
                
                # Load the existing results
                with open(all_results_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
                
                logger.info(f"Successfully downloaded and loaded existing results file with {len(all_results) if isinstance(all_results, list) else 1} entries")
            except Exception as e:
                logger.info(f"No existing results file found or error downloading: {str(e)}")
                all_results = []
                all_results_path = os.path.join(os.path.dirname(json_file), "all_results.json")
            
            # Load the new results
            with open(json_path, "r", encoding="utf-8") as f:
                new_results = json.load(f)
            
            # Add timestamp to the new results for identification
            timestamp = json_path.name.split('_')[-1].split('.')[0]
            new_results["timestamp"] = timestamp
            
            # Append the new results
            if isinstance(all_results, list):
                all_results.append(new_results)
            else:
                # If all_results is not a list (e.g., it's a dictionary), convert to a list
                all_results = [all_results, new_results]
            
            # Save the combined results
            with open(all_results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Combined results saved to {all_results_path}")
            
        except Exception as e:
            logger.error(f"Error appending results: {str(e)}")
            logger.info("Continuing with pushing individual results file")
            all_results_path = None
    
    # Create a README file with information about the results
    with open(json_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    
    # If we're appending results and have a combined file, use that for the README
    if append_results and all_results_path and os.path.exists(all_results_path):
        try:
            with open(all_results_path, "r", encoding="utf-8") as f:
                all_results_data = json.load(f)
            
            # Handle the case where results_data is the all_results list
            latest_result = all_results_data[-1] if isinstance(all_results_data, list) else results_data
            model_info = latest_result.get('model', 'Unknown model')
            total_questions = latest_result.get('total_questions', 0)
            avg_time = latest_result.get('average_time_per_question', 0)
            generation_config = latest_result.get('generation_config', {})
            results_to_sample = latest_result.get('results', [])
        except Exception as e:
            logger.error(f"Error reading combined results for README: {str(e)}")
            model_info = results_data.get('model', 'Unknown model')
            total_questions = results_data.get('total_questions', 0)
            avg_time = results_data.get('average_time_per_question', 0)
            generation_config = results_data.get('generation_config', {})
            results_to_sample = results_data.get('results', [])
    else:
        model_info = results_data.get('model', 'Unknown model')
        total_questions = results_data.get('total_questions', 0)
        avg_time = results_data.get('average_time_per_question', 0)
        generation_config = results_data.get('generation_config', {})
        results_to_sample = results_data.get('results', [])
    
    readme_content = f"""# Telugu Inference Results

## Model Information
- **Model**: {model_info}
- **Date**: {json_path.name.split('_')[-1].split('.')[0]}
- **Number of questions**: {total_questions}
- **Average time per question**: {avg_time:.2f} seconds

## Generation Configuration
- Max new tokens: {generation_config.get('max_new_tokens', 'N/A')}
- Temperature: {generation_config.get('temperature', 'N/A')}
- Top-p: {generation_config.get('top_p', 'N/A')}
- Top-k: {generation_config.get('top_k', 'N/A')}
- Data type: {generation_config.get('data_type', 'N/A')}

## Files
- `{json_path.name}`: JSON format results (individual run)
{f"- `all_results.json`: Combined results from all runs" if all_results_path else ""}
- `{readable_path.name}`: Human-readable format results (latest run)

## Sample Questions and Responses
"""
    
    # Add some sample Q&A pairs to the README
    for i, result in enumerate(results_to_sample[:5], 1):
        readme_content += f"""
### Question {i}: {result.get('question', 'N/A')}
**Response**: {result.get('response', 'N/A')[:300]}{'...' if len(result.get('response', '')) > 300 else ''}
**Generation time**: {result.get('generation_time_seconds', 0):.2f} seconds
"""
    
    readme_path = os.path.join(os.path.dirname(json_file), "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Get the repository path on Hugging Face
    repo_path = f"{repo_id}/{subfolder}" if subfolder else repo_id
    
    # Create or update the folder in the repository
    try:
        # Push the files to Hugging Face
        files_to_push = [
            (json_path, f"{subfolder}/{json_path.name}" if subfolder else json_path.name),
            (readable_path, f"{subfolder}/{readable_path.name}" if subfolder else readable_path.name),
            (Path(readme_path), f"{subfolder}/README.md" if subfolder else "README.md")
        ]
        
        # If we appended results, also push the combined file
        if all_results_path and os.path.exists(all_results_path):
            files_to_push.append(
                (Path(all_results_path), f"{subfolder}/all_results.json" if subfolder else "all_results.json")
            )
        
        for file_path, repo_file_path in files_to_push:
            logger.info(f"Uploading {file_path} to {repo_file_path}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=repo_file_path,
                repo_id=repo_id,
                repo_type="model"
            )
            logger.info(f"Uploaded {file_path.name} to {repo_file_path}")
        
        logger.info(f"Successfully pushed files to {repo_path}")
        return {
            "status": "success",
            "repo_url": f"https://huggingface.co/{repo_path}",
            "files_pushed": [file_path.name for file_path, _ in files_to_push]
        }
    
    except Exception as e:
        logger.error(f"Error pushing files to Hugging Face: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.local_entrypoint()
def main():
    """Main entry point for the Modal application."""
    
    # Define the model path and prompts
    model_path = "bharathkumar1922001/gemma-3-12b-pt-telugu-sft"  # Your fine-tuned model
    huggingface_repo = "bharathkumar1922001/gemma-3-12b-pt-telugu-sft"  # Repository to push results to
    
    # Define the prompts JSON content
    # prompts = {
    #     "questions": [
    #         # Original questions
    #         "రోగనిరోధక శక్తిని సహజంగా ఎలా పెంచుకోవచ్చు?",
    #         "ఆహారం గుండె ఆరోగ్యాన్ని ఎలా ప్రభావితం చేస్తుంది?",
    #         "రాజకీయ అవినీతి వల్ల దేశానికి ఎలాంటి నష్టాలు కలుగుతాయి?",
    #         "ఆర్థిక మాంద్యాన్ని ప్రభుత్వాలు ఎలా పరిష్కరించగలవు?",
    #         "టెలిఫోన్ ను ఎవరు ఆవిష్కరించారు?",
    #         "సరఫరా మరియు డిమాండ్ ధరలను ఎలా ప్రభావితం చేస్తాయి?",
    #         "సోషల్ మీడియా మార్కెటింగ్ ఎలా పని చేస్తుంది?",
    #         "ఇంటర్మిటెంట్ ఫాస్టింగ్ వల్ల ఏమైనా బెనిఫిట్స్?",
    #         "రోజూ జిమ్ముకు వెళ్తే బాడీ పెరుగుతుందా?",
    #         "పొట్టలో గ్యాస్ తగ్గాలంటే ఏం చేయాలి?",
    #         "ఆంధ్రప్రదేశ్ లో ప్రస్తుత రాజకీయ పరిస్థితులు ఎలా ఉన్నాయి?",
    #         "ఢిల్లీలో పొల్యూషన్ ఎక్కువ ఉంది.ఎందుకు?",
    #         "ముంబై వర్సెస్ ఢిల్లీ బెస్ట్ సిటీ ఏది?",
    #         "ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ రావడం వల్ల జాబ్స్ నిజంగానే పోతాయా?",
    #         "ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ లో బాగా షైన్ అవ్వాలంటే ఎలాంటి కోర్సులు నేర్చుకోవాలి?",
    #         "స్టాక్ మార్కెట్ అంటే ఏంటి?",
    #         "డే ట్రేడింగ్ చేయడం వల్ల ఎక్కువ డబ్బులు లాస్ అవుతామా?",
    #         "మోనాలిసా పెయింటింగ్ నిజంగానే అంత ఫేమస్సా లేక జస్ట్ అదొక హైపా?",
            
    #         # New questions
    #         "ప్రాబయాటిక్స్ తినడం వల్ల నిజంగా ఆరోగ్యానికి మేలు జరుగుతుందా?",
    #         "కెటో డైట్ చేస్తే అసలు వైట్ లాస్ అవుతుందా లేక ట్రెండ్ మాత్రమేనా?",
    #         "యోగా చేయడం వల్ల మనకి నిజంగా స్ట్రెస్ తగ్గుతుందా?",
    #         "నిద్ర లేకపోతే బ్రెయిన్ డ్యామేజ్ అవుతుందా?",
    #         "కృత్రిమ మేధ ఎందుకు ఇప్పుడే ఇంత ఫేమస్ అయింది?",
    #         "రోజు 8 గంటలు స్క్రీన్ చూస్తుంటే ఐస్ కూడా డ్యామేజ్ అవుతుందా?",
    #         "క్రిప్టో కరెన్సీ అంటే సింపుల్ గా ఏంటి? అసలు మన మనీకి సేఫా?",
    #         "మెటావర్స్ ఏంటి బ్రో? అది నిజంగా ఫ్యూచర్ ఆఫ్ ఇంటర్నెట్ ఆ?",
    #         "సాఫ్ట్‌వేర్ డెవలపర్ అవ్వాలంటే డిగ్రీ అవసరమా? ఇప్పుడున్న ట్రెండ్ ఏంటి?",
    #         "రిమోట్ వర్క్ మన ఆరోగ్యాన్ని, మెంటల్ హెల్త్ ని ఎలా ఎఫెక్ట్ చేస్తుంది?",
    #         "కాలేజీకి వెళ్లకపోయినా సక్సెస్ అవొచ్చా? ఎలా?",
    #         "రీసెంట్ గా వర్క్‌ప్లేస్‌లో ఏ స్కిల్స్ కి డిమాండ్ ఎక్కువగా ఉంది?"
    #     ]
    # }
    
    # Save prompts to temporary file in /output
    prompts_file = "/training/prompts.json"
    # os.makedirs(os.path.dirname(prompts_file), exist_ok=True)
    
    # with open(prompts_file, "w", encoding="utf-8") as f:
    #     json.dump(prompts, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved prompts to {prompts_file}")
    logger.info("Starting inference with the Telugu fine-tuned Gemma-3 model...")
    
    # Run the inference with explicit output directory
    results = run_inference.remote(
        model_path=model_path,
        prompts_file=prompts_file,
        output_dir="/output",
        use_bf16=True,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        push_to_hf=True,
        repo_id=huggingface_repo,
        subfolder="inference_results",
        append_results=True
    )
    
    # Validate results contain the expected keys
    logger.info("Validating inference results...")
    required_keys = ["data", "json_file", "timestamp"]
    missing_keys = [key for key in required_keys if key not in results]
    
    if missing_keys:
        logger.error(f"Missing keys in results: {missing_keys}")
        raise ValueError(f"Inference results missing required keys: {missing_keys}")
    
    # Verify files exist
    logger.info(f"Verifying existence of files:")
    logger.info(f"JSON file: {results['json_file']}")
    # logger.info(f"Readable file: {results['readable_file']}")
    
    if not os.path.exists(results['json_file']):
        logger.error(f"JSON file does not exist: {results['json_file']}")
        raise FileNotFoundError(f"JSON file not found: {results['json_file']}")
    
    # if not os.path.exists(results['readable_file']):
    #     logger.error(f"Readable file does not exist: {results['readable_file']}")
    #     raise FileNotFoundError(f"Readable file not found: {results['readable_file']}")
    
    # Display summary
    logger.info("\n--- Inference Complete ---")
    logger.info(f"Model: {results['data']['model']}")
    logger.info(f"Processed {results['data']['total_questions']} questions")
    logger.info(f"Total time: {results['data']['total_time_seconds']:.2f} seconds")
    logger.info(f"Average time per question: {results['data']['average_time_per_question']:.2f} seconds")
    
    # Display a few example Q&A pairs
    logger.info("\n--- Sample Q&A Pairs ---")
    for i, result in enumerate(results['data']['results'][:5]):
        logger.info(f"\nQ{i+1}: {result['question']}")
        logger.info(f"A{i+1}: {result['response']}")
        logger.info(f"Time: {result['generation_time_seconds']:.2f}s")
    
    # Push results to Hugging Face with clear file paths
    logger.info("\n--- Pushing Results to Hugging Face ---")
    push_result = push_results_to_huggingface_internal.remote(
        repo_id=huggingface_repo,
        json_file=results['json_file'],
        # readable_file=results['readable_file'],
        subfolder="inference_results",
        commit_message=f"Add Telugu inference results - {results['timestamp']}",
        append_results=True  # Enable appending results
    )
    
    # Add explicit logging to verify the push operation
    logger.info("\n--- HuggingFace Push Results ---")
    logger.info(f"Push result status: {push_result.get('status', 'unknown')}")
    if push_result.get("status") == "success":
        logger.info(f"Results successfully pushed to: {push_result['repo_url']}")
        logger.info(f"Files pushed: {', '.join(push_result['files_pushed'])}")
    else:
        logger.error(f"Failed to push results: {push_result.get('error', 'unknown error')}")
    
    logger.info("\n--- All operations completed ---")
    return {
        "inference_results": results,
        "push_results": push_result
    }