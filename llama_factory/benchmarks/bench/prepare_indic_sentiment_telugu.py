# prepare_indic_sentiment_telugu.py

import os
import pandas as pd
from datasets import load_dataset

def download_and_filter_indic_sentiment():
    """
    Download and filter the IndicSentiment dataset for Telugu.
    """
    print("Downloading IndicSentiment dataset...")
    dataset = load_dataset("ai4bharat/IndicSentiment", "translation-te", trust_remote_code=True)
    
    print("Filtering for Telugu data...")
    telugu_data = []
    
    for split in dataset:
        df = dataset[split].to_pandas()
        print(f"Available columns in {split} split: {df.columns.tolist()}")
        
        # Check the first row to understand the data structure
        if len(df) > 0:
            print(f"First row sample: {df.iloc[0].to_dict()}")
        
        # Rename columns based on the actual dataset structure
        # For IndicSentiment 'translation-te' config, the sentiment column might be different
        try:
            if 'sentiment' in df.columns:
                sentiment_col = 'sentiment'
            elif 'Sentiment' in df.columns:
                sentiment_col = 'Sentiment'
            else:
                # Find any column that might contain sentiment information
                possible_cols = [col for col in df.columns if 'sent' in col.lower()]
                sentiment_col = possible_cols[0] if possible_cols else None
                
            if sentiment_col:
                telugu_df = df.rename(columns={sentiment_col: 'label'})
                if 'review' in df.columns:
                    telugu_df = telugu_df.rename(columns={'review': 'text'})
                elif 'Review' in df.columns:
                    telugu_df = telugu_df.rename(columns={'Review': 'text'})
                else:
                    # Find review column based on likely names
                    text_cols = [col for col in df.columns if any(term in col.lower() for term in ['review', 'text', 'content'])]
                    if text_cols:
                        telugu_df = telugu_df.rename(columns={text_cols[0]: 'text'})
            else:
                print(f"Warning: Could not find sentiment column in {split} split")
                continue
                
            # Map labels to integers for easier evaluation
            # We'll inspect the data to ensure we have the right mapping
            print(f"Unique values in sentiment column: {telugu_df['label'].unique()}")
            
            label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
            # Adjust mapping based on actual values if needed
            telugu_df['label'] = telugu_df['label'].map(label_map)
            
            telugu_data.append((split, telugu_df))
            
        except Exception as e:
            print(f"Error processing {split} split: {str(e)}")
            print(f"DataFrame info: {df.info()}")
    
    # Save to disk
    os.makedirs("tasks_data/telugu_sentiment", exist_ok=True)
    
    for split_name, split_data in telugu_data:
        output_file = f"tasks_data/telugu_sentiment/{split_name}.csv"
        split_data.to_csv(output_file, index=False)
        print(f"Saved {len(split_data)} Telugu examples to {output_file}")

if __name__ == "__main__":
    download_and_filter_indic_sentiment()