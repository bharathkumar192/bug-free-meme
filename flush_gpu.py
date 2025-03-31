# # Example in Python (after you're done with model/tensors)
# import torch
# import gc


# # For PyTorch - release cached memory (doesn't release actively used memory)
# if torch.cuda.is_available():
#     torch.cuda.empty_cache() 

# # For TensorFlow/Keras
# # from tensorflow.keras import backend as K
# # K.clear_session() # This often helps release graph-related memory

# # Run Python's garbage collector
# gc.collect() 

# # Optional: double check cache after GC
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# Load model directly
# from transformers import AutoProcessor, AutoModelForImageTextToText

# processor = AutoProcessor.from_pretrained("bharathkumar1922001/gemma-3-12b-telugu")
# model = AutoModelForImageTextToText.from_pretrained("bharathkumar1922001/gemma-3-12b-telugu")

from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="bharathkumar1922001/gemma-3-12b-telugu", 
    filename="model-00003-of-00005.safetensors",
    token="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"
)