# analyze_results.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(results_path):
    """Load results from a benchmark run"""
    with open(results_path, 'r') as f:
        return json.load(f)

def analyze_sentiment_results(results_dir):
    """Analyze results from the Telugu sentiment task"""
    results_file = Path(results_dir) / 'results.json'
    samples_file = None
    
    # Find the samples file
    for file in Path(results_dir).glob('*_telugu_sentiment.jsonl'):
        samples_file = file
        break
    
    if not samples_file:
        print("No samples file found for sentiment analysis")
        return
    
    # Load results and samples
    results = load_results(results_file)
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Analyze per-class accuracy
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    df['target_label'] = df['target'].map(sentiment_map)
    df['correct'] = df.apply(lambda row: row['filtered_resps'][0][0] == row['target'], axis=1)
    
    class_accuracy = df.groupby('target_label')['correct'].mean()
    
    # Plot class accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_accuracy.index, y=class_accuracy.values)
    plt.title('Accuracy by Sentiment Class')
    plt.ylabel('Accuracy')
    plt.xlabel('Sentiment Class')
    plt.savefig(os.path.join(results_dir, 'sentiment_class_accuracy.png'))
    
    # Create a confusion matrix
    confusion = pd.crosstab(
        df['target_label'], 
        df.apply(lambda row: sentiment_map[row['filtered_resps'][0][0]], axis=1),
        normalize='index'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='.2%', cmap='Blues')
    plt.title('Confusion Matrix: Telugu Sentiment Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sentiment_confusion_matrix.png'))
    
    # Save analysis results
    analysis = {
        'class_accuracy': class_accuracy.to_dict(),
        'confusion_matrix': confusion.to_dict(),
        'overall_accuracy': results['results']['telugu_sentiment']['acc,none']
    }
    
    with open(os.path.join(results_dir, 'sentiment_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Sentiment analysis completed and saved to {results_dir}")

def analyze_mmlu_results(results_dir):
    """Analyze results from the Telugu MMLU task"""
    results_file = Path(results_dir) / 'results.json'
    samples_file = None
    
    # Find the samples file
    for file in Path(results_dir).glob('*_mmlu_telugu.jsonl'):
        samples_file = file
        break
    
    if not samples_file:
        print("No samples file found for MMLU analysis")
        return
    
    # Load results and samples
    results = load_results(results_file)
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Extract topics from questions (if available)
    # This assumes there's some structure we can use to identify topics
    # We'll use a placeholder approach here
    
    # Calculate overall accuracy
    overall_acc = results['results']['mmlu_telugu']['acc,none']
    norm_acc = results['results']['mmlu_telugu'].get('acc_norm,none', 'N/A')
    
    # Create analysis summary
    analysis = {
        'overall_accuracy': overall_acc,
        'normalized_accuracy': norm_acc,
        'sample_count': len(df)
    }
    
    with open(os.path.join(results_dir, 'mmlu_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"MMLU analysis completed and saved to {results_dir}")

def main():
    import argparse
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