import os
import re
import pandas as pd
import argparse
from typing import Dict, List, Optional
from pathlib import Path

class ModelLogParser:
    """Parser for extracting model metrics from log files."""
    
    def __init__(self, log_dir: str):
        """
        Initialize the parser with the directory containing log files.
        
        Args:
            log_dir (str): Path to directory containing log files
        """
        self.log_dir = log_dir
        self.results = []

    def extract_hyperparameters(self, filename: str) -> Dict[str, float]:
        """
        Extract hyperparameters from filename.
        
        Args:
            filename (str): Name of the log file
            
        Returns:
            Dict[str, float]: Dictionary of hyperparameters
        """
        # Example filename: lr_0.0005_n_heads_1_attn_drop_0.1_drop_0_decay_0.000_g_node_True.txt
        params = {}
        
        # Extract learning rate
        lr_match = re.search(r'lr_(\d+\.\d+)', filename)
        if lr_match:
            params['learning_rate'] = float(lr_match.group(1))
            
        # Extract number of attention heads
        heads_match = re.search(r'n_heads_(\d+)', filename)
        if heads_match:
            params['attention_heads'] = int(heads_match.group(1))
            
        # Extract attention dropout
        attn_drop_match = re.search(r'attn_drop_(\d+\.\d+)', filename)
        if attn_drop_match:
            params['attention_dropout'] = float(attn_drop_match.group(1))
            
        # Extract dropout
        drop_match = re.search(r'drop_(\d+(?:\.\d+)?)', filename)
        if drop_match:
            params['dropout'] = float(drop_match.group(1))
            
        # Extract weight decay
        decay_match = re.search(r'decay_(\d+\.\d+)', filename)
        if decay_match:
            params['weight_decay'] = float(decay_match.group(1))
            
        return params

    def extract_metrics(self, content: str) -> Dict[str, float]:
        """
        Extract model metrics and domain information from log content.
        
        Args:
            content (str): Content of the log file
            
        Returns:
            Dict[str, float]: Dictionary of metrics and domain info
        """
        metrics = {}
        
        # Extract domain name
        domain_match = re.search(r'Domain:\s*(\w+(?:_\w+)*)', content)
        if domain_match:
            metrics['domain'] = domain_match.group(1)
        
        # Extract success rates
        success_monitor_match = re.search(r'Success with monitor\s+(\d+\.\d+)%', content)
        if success_monitor_match:
            metrics['success_rate_with_monitor'] = float(success_monitor_match.group(1))
            
        success_no_monitor_match = re.search(r'Success without monitor\s+(\d+\.\d+)%', content)
        if success_no_monitor_match:
            metrics['success_rate_without_monitor'] = float(success_no_monitor_match.group(1))
            
        # Extract plan metrics
        plan_length_match = re.search(r'Plan Length\s+(\d+\.\d+)', content)
        if plan_length_match:
            metrics['avg_plan_length'] = float(plan_length_match.group(1))
            
        time_match = re.search(r'Time \(sec\)\s+(\d+\.\d+)', content)
        if time_match:
            metrics['avg_time'] = float(time_match.group(1))
            
        max_plan_match = re.search(r'MAx Plan Length\s+(\d+)', content)
        if max_plan_match:
            metrics['max_plan_length'] = int(max_plan_match.group(1))
            
        max_time_match = re.search(r'Max Time \(sec\)\s+(\d+\.\d+)', content)
        if max_time_match:
            metrics['max_time'] = float(max_time_match.group(1))
            
        median_plan_match = re.search(r'Median Plan Length\s+(\d+\.\d+)', content)
        if median_plan_match:
            metrics['median_plan_length'] = float(median_plan_match.group(1))
            
        quality_match = re.search(r'Plan Quality :\s+(\d+\.\d+)', content)
        if quality_match:
            metrics['plan_quality'] = float(quality_match.group(1))
            
        return metrics

    def parse_files(self) -> None:
        """Parse all log files in the directory and collect results."""
        # Check if directory exists
        if not os.path.exists(self.log_dir):
            raise FileNotFoundError(f"Directory not found: {self.log_dir}")
            
        # Get list of text files
        files = [f for f in os.listdir(self.log_dir) if f.endswith('.txt')]
        
        if not files:
            print(f"No .txt files found in {self.log_dir}")
            return
            
        print(f"Found {len(files)} log files to process...")
        
        for filename in files:
            try:
                filepath = os.path.join(self.log_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Extract information
                hyperparams = self.extract_hyperparameters(filename)
                metrics = self.extract_metrics(content)
                
                # Combine results
                result = {
                    'filename': filename,
                    **hyperparams,
                    **metrics
                }
                
                self.results.append(result)
                print(f"Successfully processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    def save_results(self, output_path: str) -> None:
        """
        Save results to CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
        """
        if not self.results:
            print("No results to save. Run parse_files() first.")
            return
            
        df = pd.DataFrame(self.results)
        
        # Reorder columns for better readability
        column_order = [
            'filename',
            'domain',
            'learning_rate',
            'attention_heads',
            'attention_dropout',
            'dropout',
            'weight_decay',
            'success_rate_with_monitor',
            'success_rate_without_monitor',
            'avg_plan_length',
            'avg_time',
            'max_plan_length',
            'max_time',
            'median_plan_length',
            'plan_quality'
        ]
        
        # Only include columns that exist in the DataFrame
        final_columns = [col for col in column_order if col in df.columns]
        df = df[final_columns]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Parse model log files and extract metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'log_dir',
        type=str,
        help='Directory containing the log files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='model_results.csv',
        help='Output CSV file path'
    )
    
    return parser

def main():
    """Main function to run the parser."""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Convert to absolute paths
    log_dir = os.path.abspath(args.log_dir)
    output_path = os.path.abspath(args.output)
    
    try:
        # Initialize parser
        parser = ModelLogParser(log_dir)
        
        # Parse files
        parser.parse_files()
        
        # Save results
        parser.save_results(output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())