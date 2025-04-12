# prepare_indic_sentiment_telugu.py

import os
import pandas as pd
from datasets import load_dataset

def download_and_filter_indic_sentiment():
    """
    Download and filter the IndicSentiment dataset for Telugu.
    """
    print("Downloading IndicSentiment dataset...")
    dataset = load_dataset("ai4bharat/IndicSentiment")
    
    # Extract Telugu reviews
    print("Filtering for Telugu data...")
    telugu_data = []
    
    for split in dataset:
        df = dataset[split].to_pandas()
        
        # Only select rows with Telugu reviews
        telugu_rows = df[df['INDIC REVIEW'].str.contains('telugu', case=False, na=False)]
        
        # Keep only necessary columns
        telugu_df = telugu_rows[['INDIC REVIEW', 'LABEL']].rename(
            columns={'INDIC REVIEW': 'text', 'LABEL': 'label'}
        )
        
        # Map labels to integers for easier evaluation
        label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        telugu_df['label'] = telugu_df['label'].map(label_map)
        
        telugu_data.append((split, telugu_df))
    
    # Save to disk
    os.makedirs("tasks_data/telugu_sentiment", exist_ok=True)
    
    for split_name, split_data in telugu_data:
        output_file = f"tasks_data/telugu_sentiment/{split_name}.csv"
        split_data.to_csv(output_file, index=False)
        print(f"Saved {len(split_data)} Telugu examples to {output_file}")

if __name__ == "__main__":
    download_and_filter_indic_sentiment()