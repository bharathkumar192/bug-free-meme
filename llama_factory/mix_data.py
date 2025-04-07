import json
import os
import random # Import the random module

# --- Configuration ---
RESULTS_FILE = 'telugu_results.json'
TELUGU_FILE = 'telugu.json'
# Changed output filename slightly to reflect randomness
OUTPUT_FILE = 'combined_random_samples.json' 

# Set how many random samples you want from each file
x_random_samples_from_results = 40000 - 4962  # Number of random samples from results.json
y_random_samples_from_telugu = 4962   # Number of random samples from telugu.json
# --- End Configuration ---

def load_json_file(filepath):
    """Loads data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'questions' not in data or not isinstance(data['questions'], list):
                print(f"Error: Expected 'questions' key with a list value in {filepath}")
                return None
            return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return None

# Modified function to select random samples
def process_random_samples(data, key_map, num_samples_to_take, filename):
    """Extracts and standardizes a random subset of samples."""
    processed = []
    source_list = data.get('questions', [])
    
    # Ensure we don't try to take more samples than available
    actual_samples_to_take = min(num_samples_to_take, len(source_list))
    
    # Print warnings/info based on availability vs requested number
    if num_samples_to_take > len(source_list) and len(source_list) > 0:
        print(f"Warning: Requested {num_samples_to_take} random samples from {filename}, but only {len(source_list)} are available. Taking all {len(source_list)} randomly.")
    elif num_samples_to_take == 0:
         print(f"Info: Requested 0 samples from {filename}.")
    elif len(source_list) == 0:
         print(f"Warning: No samples found in {filename} under the 'questions' key. Cannot select any.")
         return [], 0 # Return empty list and 0 count

    selected_items = []
    if actual_samples_to_take > 0:
        try:
            # Use random.sample to get unique random items from the source list
            selected_items = random.sample(source_list, actual_samples_to_take)
        except ValueError as e:
             # This should theoretically not happen due to the min() check above, but good practice
             print(f"Error during random sampling for {filename}: {e}")
             return [], 0
             
    # Process the randomly selected items
    for item in selected_items:
        # Use .get() to safely access keys, providing default empty strings
        prompt = item.get(key_map['prompt'], '')
        response = item.get(key_map['response'], '')
        
        if not prompt:
             print(f"Warning: Empty prompt found in a randomly selected item from {filename}")
        if not response:
             print(f"Warning: Empty response found in a randomly selected item from {filename}")

        processed.append({"prompt": prompt, "response": response})
        
    return processed, actual_samples_to_take


# --- Main Execution ---

# Load data from files
results_data = load_json_file(RESULTS_FILE)
telugu_data = load_json_file(TELUGU_FILE)

# Proceed only if both files were loaded successfully
if results_data and telugu_data:
    
    # Define key mappings for standardization
    results_key_map = {"prompt": "question", "response": "response"}
    telugu_key_map = {"prompt": "prompt", "response": "response"}

    # Process RANDOM samples from results.json
    processed_results, num_results_taken = process_random_samples(
        results_data, results_key_map, x_random_samples_from_results, RESULTS_FILE
    )

    # Process RANDOM samples from telugu.json
    processed_telugu, num_telugu_taken = process_random_samples(
        telugu_data, telugu_key_map, y_random_samples_from_telugu, TELUGU_FILE
    )

    # Combine the processed samples 
    # The order will be: random samples from results first, then random samples from telugu
    combined_questions = processed_results + processed_telugu

    # --- Optional: Shuffle the final combined list ---
    # If you want the final list itself to be shuffled (mixing results and telugu samples together randomly):
    random.shuffle(combined_questions) 
    print("\nNote: The final combined list of samples has also been shuffled.")
    # --- End Optional Shuffle ---


    # Structure the final output
    final_data = {"questions": combined_questions}

    # Write the combined data to the output file
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2) 
        
        print(f"\nSuccessfully created '{OUTPUT_FILE}' with:")
        print(f"- {num_results_taken} random samples from '{RESULTS_FILE}'") # Updated message
        print(f"- {num_telugu_taken} random samples from '{TELUGU_FILE}'") # Updated message
        print(f"Total samples in output: {len(combined_questions)}")

    except IOError as e:
        print(f"Error writing output file '{OUTPUT_FILE}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output file: {e}")

else:
    print("\nAborting script due to errors loading input files.")

# --- End of Script ---