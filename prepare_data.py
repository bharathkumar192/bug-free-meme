import json
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(
        self,
        input_file: str,
        output_dir: str,
        min_length: int = 50,
        max_length: int = 2048,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from input file"""
        logger.info(f"Loading data from {self.input_file}")
        
        if self.input_file.endswith('.json'):
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif self.input_file.endswith('.csv'):
            data = pd.read_csv(self.input_file).to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {self.input_file}")
            
        return data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()
    
    def validate_text(self, text: str) -> bool:
        """Validate text meets quality criteria"""
        if len(text) < self.min_length:
            return False
        if len(text) > self.max_length:
            return False
        # Check for minimum word count
        if len(text.split()) < 10:
            return False
        # Check for basic sentence structure
        if not re.search(r'[.!?]', text):
            return False
        return True
    
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean the data"""
        processed_data = []
        
        for item in tqdm(data, desc="Processing data"):
            if isinstance(item, dict):
                text = item.get('text', '')
            else:
                text = str(item)
                
            text = self.clean_text(text)
            
            if self.validate_text(text):
                processed_data.append({
                    'text': text,
                    'length': len(text),
                    'word_count': len(text.split())
                })
                
        return processed_data
    
    def create_splits(self, data: List[Dict[str, Any]]) -> Dict[str, Dataset]:
        """Create train/validation/test splits"""
        # Shuffle data
        np.random.shuffle(data)
        
        # Calculate split sizes
        total_size = len(data)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        
        # Create splits
        splits = {
            'train': data[:train_size],
            'validation': data[train_size:train_size + val_size],
            'test': data[train_size + val_size:]
        }
        
        # Convert to HuggingFace datasets
        datasets = {}
        for split_name, split_data in splits.items():
            datasets[split_name] = Dataset.from_list(split_data)
            
        return datasets
    
    def save_datasets(self, datasets: Dict[str, Dataset]):
        """Save processed datasets"""
        for split_name, dataset in datasets.items():
            output_file = self.output_dir / f"{split_name}.json"
            dataset.to_json(output_file)
            logger.info(f"Saved {split_name} split to {output_file}")
            
    def generate_statistics(self, datasets: Dict[str, Dataset]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        stats = {}
        
        for split_name, dataset in datasets.items():
            lengths = [item['length'] for item in dataset]
            word_counts = [item['word_count'] for item in dataset]
            
            stats[split_name] = {
                'num_examples': len(dataset),
                'avg_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'min_length': np.min(lengths),
                'max_length': np.max(lengths),
                'avg_word_count': np.mean(word_counts),
                'std_word_count': np.std(word_counts),
                'min_word_count': np.min(word_counts),
                'max_word_count': np.max(word_counts)
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
            logger.info(f"Loaded {len(data)} examples")
            
            # Process data
            processed_data = self.process_data(data)
            logger.info(f"Processed {len(processed_data)} examples")
            
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
    # Example usage
    preparator = DataPreparator(
        input_file="telugu_results.json", 
        output_dir="processed_data",
        min_length=50,
        max_length=2048,
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05
    )
    
    preparator.process()

if __name__ == "__main__":
    main() 