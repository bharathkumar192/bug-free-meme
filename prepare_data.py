import json
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UnifiedConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, config: UnifiedConfig):
        """Initialize with unified configuration"""
        self.config = config
        self.input_file = config.data.input_file
        self.output_dir = Path(config.data.output_dir)
        self.min_length = config.data.min_length
        self.max_length = config.data.max_length
        self.train_ratio = config.data.train_ratio
        self.val_ratio = config.data.val_ratio
        self.test_ratio = config.data.test_ratio
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from input file"""
        logger.info(f"Loading data from {self.input_file}")
        
        if self.input_file.endswith('.json'):
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if data has the 'questions' key
                if 'questions' in data:
                    questions = data['questions']
                    logger.info(f"Found {len(questions)} question-answer pairs")
                    return questions
                else:
                    logger.warning("Data does not have 'questions' key")
                    return []
        elif self.input_file.endswith('.csv'):
            data = pd.read_csv(self.input_file).to_dict('records')
            return data
        else:
            raise ValueError(f"Unsupported file format: {self.input_file}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()
    
    def validate_text(self, text: str) -> bool:
        """Validate text meets quality criteria"""
        if not isinstance(text, str):
            return False
            
        if len(text) < self.min_length:
            return False
        if len(text) > self.max_length:
            return False
        # Check for minimum word count
        if len(text.split()) < 5:  # Reduced to 5 from 10
            return False
        return True
    
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw data into training format for question-answer pairs"""
        processed_data = []
        
        for item in tqdm(data, desc="Processing data"):
            # Skip if required fields are missing
            if 'question' not in item or 'response' not in item:
                continue
                
            question = self.clean_text(item['question'])
            response = self.clean_text(item['response'])
            
            # Skip if either question or response doesn't pass validation
            if not self.validate_text(question) or not self.validate_text(response):
                continue
            
            # Use chat template format
            if self.config.model.use_chat_template:
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
                processed_data.append({
                    "messages": messages,
                    "q_length": len(question),
                    "r_length": len(response),
                })
            else:
                # Standard format if not using chat template
                processed_data.append({
                    'question': question,
                    'response': response,
                    'q_length': len(question),
                    'r_length': len(response),
                })
        
        logger.info(f"Processed {len(processed_data)} valid examples")
        return processed_data
    
    def create_splits(self, data: List[Dict[str, Any]]) -> Dict[str, Dataset]:
        """Create train/validation/test splits"""
        # Return empty datasets if no data
        if not data:
            empty_dataset = Dataset.from_list([])
            return {
                'train': empty_dataset,
                'validation': empty_dataset,
                'test': empty_dataset
            }
            
        # Ensure deterministic results
        np.random.seed(self.config.integration.seed)
        
        # Shuffle data
        shuffled_data = data.copy()
        np.random.shuffle(shuffled_data)
        
        # Calculate split sizes
        total_size = len(shuffled_data)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        
        # Split the data
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        # Convert to HuggingFace datasets
        datasets = {}
        datasets['train'] = Dataset.from_list(train_data)
        datasets['validation'] = Dataset.from_list(val_data) 
        datasets['test'] = Dataset.from_list(test_data)
            
        return datasets
    
    def save_datasets(self, datasets: Dict[str, Dataset]):
        """Save processed datasets"""
        for split_name, dataset in datasets.items():
            if len(dataset) > 0:  # Only save non-empty datasets
                output_file = self.output_dir / f"{split_name}.json"
                dataset.to_json(output_file)
                logger.info(f"Saved {split_name} split with {len(dataset)} examples to {output_file}")
    
    def generate_statistics(self, datasets: Dict[str, Dataset]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        stats = {}
        
        for split_name, dataset in datasets.items():
            if len(dataset) == 0:
                stats[split_name] = {
                    'num_examples': 0,
                    'note': 'Empty dataset'
                }
                continue
                
            # Check if we're using chat template
            if 'messages' in dataset[0]:
                # For chat template format
                q_lengths = [item['q_length'] for item in dataset]
                r_lengths = [item['r_length'] for item in dataset]
                
                stats[split_name] = {
                    'num_examples': len(dataset),
                    'avg_question_length': float(np.mean(q_lengths)),
                    'avg_response_length': float(np.mean(r_lengths)),
                    'min_question_length': int(np.min(q_lengths)),
                    'max_question_length': int(np.max(q_lengths)),
                    'min_response_length': int(np.min(r_lengths)),
                    'max_response_length': int(np.max(r_lengths)),
                }
            else:
                # For standard format
                q_lengths = [len(item['question']) for item in dataset]
                r_lengths = [len(item['response']) for item in dataset]
                
                stats[split_name] = {
                    'num_examples': len(dataset),
                    'avg_question_length': float(np.mean(q_lengths)),
                    'avg_response_length': float(np.mean(r_lengths)),
                    'min_question_length': int(np.min(q_lengths)),
                    'max_question_length': int(np.max(q_lengths)),
                    'min_response_length': int(np.min(r_lengths)),
                    'max_response_length': int(np.max(r_lengths)),
                }
            
        return stats
    
    def save_statistics(self, stats: Dict[str, Any]):
        """Save dataset statistics"""
        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_file}")
    
    def process(self):
        """Main processing pipeline"""
        try:
            # Load data
            data = self.load_data()
            if not data:
                logger.error("No data loaded or empty dataset")
                return
                
            # Process data
            processed_data = self.process_data(data)
            if not processed_data:
                logger.error("No examples passed validation")
                return
                
            # Create splits
            datasets = self.create_splits(processed_data)
            
            # Generate statistics
            stats = self.generate_statistics(datasets)
            
            # Save everything
            self.save_datasets(datasets)
            self.save_statistics(stats)
            
            logger.info("Data preparation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during data preparation: {str(e)}")
            raise

def main():
    # Load unified configuration
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for LLM fine-tuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = UnifiedConfig.from_file(args.config)
    
    # Initialize data preparator with configuration
    preparator = DataPreparator(config)
    
    # Process data
    preparator.process()

if __name__ == "__main__":
    main()