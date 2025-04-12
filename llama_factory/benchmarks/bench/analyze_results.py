# analyze_results.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_results(results_path):
    """Load results from a benchmark run"""
    with open(results_path, 'r') as f:
        return json.load(f)

def analyze_sentiment_results(results_dir):
    """Analyze results from the Telugu sentiment task"""
    results_file = Path(results_dir) / 'results.json'
    samples_files = list(Path(results_dir).glob('*telugu_sentiment.jsonl'))
    
    if not results_file.exists():
        print(f"Results file not found at {results_file}")
        return
    
    if not samples_files:
        print("No samples file found for sentiment analysis")
        return
    
    samples_file = samples_files[0]
    
    # Load results and samples
    results = load_results(results_file)
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Create output directory
    analysis_dir = Path(results_dir) / 'analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Analyze per-class accuracy if possible
    if 'target' in df.columns and 'filtered_resps' in df.columns:
        sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        df['target_label'] = df['target'].apply(lambda x: sentiment_map.get(x, str(x)))
        df['correct'] = df.apply(lambda row: row['filtered_resps'][0][0] == row['target'] 
                              if len(row['filtered_resps']) > 0 and len(row['filtered_resps'][0]) > 0 
                              else False, axis=1)
        
        class_accuracy = df.groupby('target_label')['correct'].mean()
        
        # Plot class accuracy
        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_accuracy.index, y=class_accuracy.values)
        plt.title('Accuracy by Sentiment Class')
        plt.ylabel('Accuracy')
        plt.xlabel('Sentiment Class')
        plt.savefig(analysis_dir / 'sentiment_class_accuracy.png')
        
        # Create confusion matrix if possible
        try:
            confusion = pd.crosstab(
                df['target_label'], 
                df.apply(lambda row: sentiment_map.get(row['filtered_resps'][0][0], 'unknown') 
                      if len(row['filtered_resps']) > 0 and len(row['filtered_resps'][0]) > 0 
                      else 'unknown', axis=1),
                normalize='index'
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion, annot=True, fmt='.2%', cmap='Blues')
            plt.title('Confusion Matrix: Telugu Sentiment Analysis')
            plt.tight_layout()
            plt.savefig(analysis_dir / 'sentiment_confusion_matrix.png')
        except Exception as e:
            print(f"Could not create confusion matrix: {e}")
    
    # Save analysis results
    analysis = {
        'overall_accuracy': results['results'].get('telugu_sentiment', {}).get('acc,none', 'N/A')
    }
    
    if 'target' in df.columns and 'filtered_resps' in df.columns:
        analysis['class_accuracy'] = class_accuracy.to_dict() if 'class_accuracy' in locals() else {}
    
    with open(analysis_dir / 'sentiment_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Sentiment analysis completed and saved to {analysis_dir}")

def analyze_mmlu_results(results_dir):
    """Analyze results from the Telugu MMLU task"""
    results_file = Path(results_dir) / 'results.json'
    samples_files = list(Path(results_dir).glob('*mmlu_telugu.jsonl'))
    
    if not results_file.exists():
        print(f"Results file not found at {results_file}")
        return
    
    if not samples_files:
        print("No samples file found for MMLU analysis")
        return
    
    samples_file = samples_files[0]
    
    # Load results and samples
    results = load_results(results_file)
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Create output directory
    analysis_dir = Path(results_dir) / 'analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Calculate overall accuracy
    overall_acc = results['results'].get('mmlu_telugu', {}).get('acc,none', 'N/A')
    norm_acc = results['results'].get('mmlu_telugu', {}).get('acc_norm,none', 'N/A')
    
    # Create analysis summary
    analysis = {
        'overall_accuracy': overall_acc,
        'normalized_accuracy': norm_acc,
        'sample_count': len(df)
    }
    
    with open(analysis_dir / 'mmlu_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"MMLU analysis completed and saved to {analysis_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results for Telugu models")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing benchmark results")
    args = parser.parse_args()
    
    # Run analyses
    analyze_sentiment_results(args.results_dir)
    analyze_mmlu_results(args.results_dir)
    
    print("Analysis completed!")

if __name__ == "__main__":
    main()