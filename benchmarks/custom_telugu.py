#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Register custom Telugu benchmark tasks with the LM Evaluation Harness.
Place this file in lm-evaluation-harness/lm_eval/tasks/custom/ directory.
"""

import os
import yaml
from pathlib import Path
from lm_eval.api.task import Task, ConfigurableTask
from lm_eval.api.registry import register_task

# Load tasks from the YAML file
def load_tasks_from_yaml(yaml_path):
    """Load task definitions from a YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Path to your custom task YAML
TASK_YAML = os.path.join(os.path.dirname(__file__), "telugu_tasks.yaml")

# Register IndxSentiment for Telugu
@register_task("indicsentiment_te")
class IndicSentimentTelugu(ConfigurableTask):
    """Telugu sentiment analysis task from IndicSentiment dataset."""
    
    DATASET_PATH = "ai4bharat/IndicSentiment"
    DATASET_NAME = "te"  # Telugu language code
    
    def __init__(self, seed=42, num_fewshot=0):
        super().__init__(seed=seed, num_fewshot=num_fewshot)
        
    def get_config(self):
        return {
            "description": "Telugu sentiment analysis from IndicSentiment dataset",
            "group": "indic_tasks",
            "task": "multiple_choice",
            "dataset_path": self.DATASET_PATH,
            "dataset_name": self.DATASET_NAME,
            "output_type": "multiple_choice",
            "doc_to_text": "Classify the sentiment of the following Telugu text as Positive, Negative, or Neutral: {text}",
            "doc_to_target": lambda x: x["label"],
            "doc_to_choice": lambda x: ["Positive", "Negative", "Neutral"],
            "metrics": ["accuracy", "f1_macro"],
        }

# Register IndicQA for Telugu
@register_task("indicqa_te")
class IndicQATelugu(ConfigurableTask):
    """Telugu question answering task from IndicQA dataset."""
    
    DATASET_PATH = "ai4bharat/IndicQA"
    DATASET_NAME = "te"  # Telugu language code
    
    def get_config(self):
        return {
            "description": "Telugu question answering from IndicQA dataset",
            "group": "indic_tasks",
            "task": "qa",
            "dataset_path": self.DATASET_PATH,
            "dataset_name": self.DATASET_NAME,
            "output_type": "generate_until",
            "doc_to_text": lambda x: f"Context: {x.get('context', '')}\n\nQuestion: {x['question']}\n\nAnswer:",
            "doc_to_target": lambda x: x["answers"][0] if isinstance(x["answers"], list) else x["answers"],
            "metrics": ["rouge", "bleu", "exact_match"],
            "generation_kwargs": {
                "until": ["\n\n"],
                "max_gen_toks": 512
            }
        }

# Register Flores Translation for Telugu-English
@register_task("flores_te_en")
class FloresTeluguEnglish(ConfigurableTask):
    """Telugu to English translation from FLORES dataset."""
    
    DATASET_PATH = "facebook/flores"
    
    def get_config(self):
        return {
            "description": "Telugu to English translation from FLORES dataset",
            "group": "indic_translation",
            "task": "generate_until",
            "dataset_path": self.DATASET_PATH,
            "dataset_name": "tel_eng",  # Telugu-English language pair
            "output_type": "generate_until",
            "doc_to_text": lambda x: f"Translate the following Telugu text to English:\n\nTelugu: {x['sentence_tel']}\n\nEnglish:",
            "doc_to_target": lambda x: x["sentence_eng"],
            "metrics": ["sacrebleu", "rouge"],
            "generation_kwargs": {
                "until": ["\n\n"],
                "max_gen_toks": 256
            },
            "fewshot_config": {
                "num_fewshot": 1,
                "pool_size": 5
            }
        }

# Register Flores Translation for English-Telugu
@register_task("flores_en_te")
class FloresEnglishTelugu(ConfigurableTask):
    """English to Telugu translation from FLORES dataset."""
    
    DATASET_PATH = "facebook/flores"
    
    def get_config(self):
        return {
            "description": "English to Telugu translation from FLORES dataset",
            "group": "indic_translation",
            "task": "generate_until",
            "dataset_path": self.DATASET_PATH,
            "dataset_name": "eng_tel",  # English-Telugu language pair
            "output_type": "generate_until",
            "doc_to_text": lambda x: f"Translate the following English text to Telugu:\n\nEnglish: {x['sentence_eng']}\n\nTelugu:",
            "doc_to_target": lambda x: x["sentence_tel"],
            "metrics": ["sacrebleu", "rouge"],
            "generation_kwargs": {
                "until": ["\n\n"],
                "max_gen_toks": 256
            },
            "fewshot_config": {
                "num_fewshot": 1,
                "pool_size": 5
            }
        }

# Register MILU Telugu General
@register_task("milu_te_general")
class MILUTeluguGeneral(ConfigurableTask):
    """Telugu general language understanding from MILU dataset."""
    
    DATASET_PATH = "ai4bharat/MILU"
    DATASET_NAME = "te_general"
    
    def get_config(self):
        return {
            "description": "Telugu general language understanding from MILU dataset",
            "group": "indic_tasks",
            "task": "generate_until",
            "dataset_path": self.DATASET_PATH,
            "dataset_name": self.DATASET_NAME,
            "output_type": "generate_until",
            "doc_to_text": lambda x: x["prompt"],
            "doc_to_target": lambda x: x["response"],
            "metrics": ["rouge", "bleu"],
            "generation_kwargs": {
                "until": ["\n\n"],
                "max_gen_toks": 512
            }
        }

# Load and register tasks from the YAML file if it exists
if os.path.exists(TASK_YAML):
    try:
        yaml_tasks = load_tasks_from_yaml(TASK_YAML)
        
        for task_name, task_config in yaml_tasks.items():
            @register_task(task_name)
            class DynamicTask(ConfigurableTask):
                def get_config(self):
                    return task_config
                    
        print(f"Registered {len(yaml_tasks)} tasks from {TASK_YAML}")
    except Exception as e:
        print(f"Error loading tasks from {TASK_YAML}: {e}")