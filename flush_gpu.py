# Example in Python (after you're done with model/tensors)
import torch
import gc


# For PyTorch - release cached memory (doesn't release actively used memory)
if torch.cuda.is_available():
    torch.cuda.empty_cache() 

# For TensorFlow/Keras
# from tensorflow.keras import backend as K
# K.clear_session() # This often helps release graph-related memory

# Run Python's garbage collector
gc.collect() 

# Optional: double check cache after GC
if torch.cuda.is_available():
    torch.cuda.empty_cache()