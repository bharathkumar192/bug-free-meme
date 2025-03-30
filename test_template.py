import json
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-pt")

# Set the chat template
tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
<start_of_turn>user
{{ message['content'] }}
<end_of_turn>
{% elif message['role'] == 'assistant' %}
<start_of_turn>model
{{ message['content'] }}
<end_of_turn>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<start_of_turn>model
{% endif %}"""

# Load a sample from your dataset
with open("telugu_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    sample = data["questions"][0]

# Create a messages array in the format expected by the tokenizer
messages = [
    {"role": "user", "content": sample["question"]},
    {"role": "assistant", "content": sample["response"]}
]

# Apply the chat template
formatted = tokenizer.apply_chat_template(messages, tokenize=False)

# Print the formatted text to see the result
print("Formatted conversation:")
print(formatted)
print("\n---\n")

# You can also tokenize it to check if tokenization works
tokens = tokenizer.encode(formatted)
print(f"Tokenized successfully: {len(tokens)} tokens")