#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Telugu Benchmark Results Analysis
--------------------------------
This script analyzes and visualizes benchmark results for Telugu language models.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class BenchmarkAnalyzer:
    """Analyzes benchmark results from the LM Evaluation Harness."""
    
    def __init__(self, results_dir="benchmark_results"):
        """Initialize the analyzer with the directory containing benchmark results."""
        self.results_dir = Path(results_dir)
        self.results = {}
        self.models = set()
        self.tasks = set()
        
    def load_results(self):
        """Load all result files from the results directory."""
        result_files = list(self.results_dir.glob("*_results.json"))
        
        if not result_files:
            print(f"No result files found in {self.results_dir}")
            return
            
        print(f"Found {len(result_files)} result files")
        
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                model_name = data.get('config', {}).get('model', file_path.stem.split('_results')[0])
                
                # Store results
                self.results[file_path.stem] = data
                self.models.add(model_name)
                
                # Extract tasks
                for task in data.get('results', {}).keys():
                    self.tasks.add(task)
                    
                print(f"Loaded results for model: {model_name}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def create_comparison_table(self):
        """Create a comparison table of all models and tasks."""
        if not self.results:
            print("No results loaded. Call load_results() first.")
            return None
            
        # Create a multi-level index for the metrics
        metrics = {}
        
        for result_key, result in self.results.items():
            model_name = result.get('config', {}).get('model', result_key)
            
            for task, task_results in result.get('results', {}).items():
                if task not in metrics:
                    metrics[task] = {}
                
                metrics[task][model_name] = task_results
        
        # Create DataFrames for each task
        task_dfs = {}
        for task, model_results in metrics.items():
            task_dfs[task] = pd.DataFrame(model_results)
        
        return task_dfs
    
    def plot_comparison(self, metric='accuracy', output_dir=None):
        """Plot comparison of models across tasks for a given metric."""
        task_dfs = self.create_comparison_table()
        
        if not task_dfs:
            return
            
        if output_dir is None:
            output_dir = self.results_dir / "plots"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison plots for each task
        for task, df in task_dfs.items():
            if metric in df.index:
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x=df.loc[metric].index, y=df.loc[metric].values)
                plt.title(f"{task} - {metric}")
                plt.ylabel(metric)
                plt.xlabel("Model")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save plot
                plot_path = output_dir / f"{task}_{metric}.png"
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved plot to {plot_path}")
    
    def plot_radar_chart(self, output_dir=None):
        """Create radar charts to compare models across multiple metrics."""
        task_dfs = self.create_comparison_table()
        
        if not task_dfs:
            return
            
        if output_dir is None:
            output_dir = self.results_dir / "plots"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Find common metrics across tasks
        common_metrics = set()
        for task, df in task_dfs.items():
            if len(common_metrics) == 0:
                common_metrics = set(df.index)
            else:
                common_metrics &= set(df.index)
        
        # Filter only numerical metrics
        numeric_metrics = []
        for metric in common_metrics:
            is_numeric = all(isinstance(task_dfs[task].loc[metric].iloc[0], (int, float)) 
                            for task in task_dfs.keys())
            if is_numeric:
                numeric_metrics.append(metric)
        
        if not numeric_metrics:
            print("No common numeric metrics found across tasks")
            return
            
        # Create a radar chart for each model across tasks
        for model in self.models:
            # Extract data for this model
            model_data = {}
            for task, df in task_dfs.items():
                if model in df.columns:
                    model_data[task] = {metric: df.loc[metric, model] for metric in numeric_metrics}
            
            if not model_data:
                continue
                
            # Convert to DataFrame for plotting
            plot_df = pd.DataFrame(model_data).T
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of variables
            N = len(plot_df)
            angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Draw the polygon for each metric
            for metric in numeric_metrics:
                values = plot_df[metric].values.tolist()
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, label=metric)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels and title
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(plot_df.index)
            plt.title(f"Performance of {model} across tasks")
            plt.legend(loc='upper right')
            
            # Save plot
            plot_path = output_dir / f"{model}_radar.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved radar chart to {plot_path}")
    
    def generate_report(self, output_path=None):
        """Generate a comprehensive report of benchmark results."""
        if not self.results:
            print("No results loaded. Call load_results() first.")
            return
            
        if output_path is None:
            output_path = self.results_dir / "benchmark_report.md"
        
        task_dfs = self.create_comparison_table()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Telugu Language Model Benchmark Report\n\n")
            
            f.write("## Models Evaluated\n\n")
            for model in sorted(self.models):
                f.write(f"- {model}\n")
            
            f.write("\n## Tasks Evaluated\n\n")
            for task in sorted(self.tasks):
                f.write(f"- {task}\n")
            
            f.write("\n## Results by Task\n\n")
            
            for task, df in task_dfs.items():
                f.write(f"### {task}\n\n")
                f.write(df.to_markdown())
                f.write("\n\n")
            
            f.write("\n## Conclusion\n\n")
            f.write("This report compares the performance of different Telugu language models across various benchmarks.\n")
            f.write("The results highlight the strengths and weaknesses of each model on tasks relevant to Telugu language processing.\n")
        
        print(f"Report generated at {output_path}")
        
        # Also export CSV files for each task
        csv_dir = self.results_dir / "csv"
        os.makedirs(csv_dir, exist_ok=True)
        
        for task, df in task_dfs.items():
            csv_path = csv_dir / f"{task}_comparison.csv"
            df.to_csv(csv_path)
            print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    analyzer = BenchmarkAnalyzer("benchmark_results")
    analyzer.load_results()
    analyzer.plot_comparison(metric='accuracy')
    analyzer.plot_comparison(metric='bleu')
    analyzer.plot_comparison(metric='rouge1')
    analyzer.plot_radar_chart()
    analyzer.generate_report()