from typing import List, Dict, Any, Optional
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
from pathlib import Path
import logging

'''
python gen_figures1.py

# 9. Verify outputs
ls -l figures/comparison_*.png
ls -l results/evolution_results_*.json
'''

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlottingError(Exception):
    """Custom exception for plotting errors."""
    pass

class ResultsLoadError(Exception):
    """Custom exception for results loading errors."""
    pass

def load_results(results_dir: str = 'results') -> pd.DataFrame:
    """
    Load all results files and organize data into a DataFrame.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        DataFrame containing organized results data
        
    Raises:
        ResultsLoadError: If there's an error loading the results
    """
    try:
        data: List[Dict[str, Any]] = []
        
        # Define primary metric for each benchmark and split
        primary_metrics = {
            'chartqa': {
                'test': 'relaxed_overall'
            },
            'vqav2': {
                'validation': 'exact_match',
                'test': 'submission'
            },
            'mmmu': {
                'validation': 'mmmu_acc',
                'test': 'submission'
            }
        }
        
        # Find all result files
        result_files = glob(os.path.join(results_dir, 'evolution_results_*.json'))
        
        if not result_files:
            raise ResultsLoadError(f"No result files found in {results_dir}")
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    
                    # Extract benchmark and split from filename
                    filename = os.path.basename(file_path)
                    parts = filename.replace('.json', '').split('_')
                    if len(parts) < 4:  # evolution_results_benchmark_split.json
                        logger.warning(f"Unexpected filename format: {filename}")
                        continue
                        
                    benchmark = parts[2]
                    split = parts[3]
                    
                    # Get metric name from the results file (new)
                    metric_name = result.get('metric_name')
                    if not metric_name:
                        # Fallback to primary_metrics if not found in results
                        if benchmark not in primary_metrics or split not in primary_metrics[benchmark]:
                            logger.warning(f"No primary metric defined for {benchmark} {split}")
                            continue
                        metric_name = primary_metrics[benchmark][split]
                    
                    # Extract data using correct metric name
                    data.append({
                        'model': result['model'],
                        'benchmark': benchmark,
                        'evolve_type': result['evolve_type'],
                        'baseline_score': result['baseline_validation_metrics'][metric_name],
                        'evolved_score': result['validation_metrics'][metric_name],
                        'metric': metric_name,
                        'generation_best_scores': result['generation_best_scores'],
                        'seed_scores': result['seed_scores']
                    })
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding {file_path}: {str(e)}")
                continue
            except KeyError as e:
                logger.error(f"Missing key in {file_path}: {str(e)}")
                continue
        
        if not data:
            raise ResultsLoadError("No valid results data found")
        
        return pd.DataFrame(data)
    
    except Exception as e:
        raise ResultsLoadError(f"Error loading results: {str(e)}")

def create_barplot(
    data: pd.DataFrame,
    benchmark: str,
    evolve_type: str,
    output_dir: str = 'figures'
) -> None:
    """
    Create and save a barplot for a specific benchmark and evolution type.
    
    Args:
        data: DataFrame containing results data
        benchmark: Name of the benchmark
        evolve_type: Type of evolution strategy
        output_dir: Directory to save figures
        
    Raises:
        PlottingError: If there's an error creating or saving the plot
    """
    try:
        # Filter data for this benchmark and evolve_type
        plot_data = data[
            (data['benchmark'] == benchmark) & 
            (data['evolve_type'] == evolve_type)
        ]
        
        if plot_data.empty:
            raise PlottingError(f"No data found for benchmark={benchmark}, evolve_type={evolve_type}")
        
        # Get metric name from the data
        metric_name = plot_data['metric'].iloc[0]
        
        # Prepare data for plotting
        plot_df = pd.melt(
            plot_data,
            id_vars=['model'],
            value_vars=['baseline_score', 'evolved_score'],
            var_name='Type',
            value_name=metric_name  # Use actual metric name
        )
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create grouped barplot
        sns.barplot(
            data=plot_df,
            x='model',
            y=metric_name,  # Use actual metric name
            hue='Type',
            palette=['lightgray', 'darkblue']
        )
        
        # Customize plot
        plt.title(f'{benchmark.upper()} - {evolve_type.capitalize()} Evolution\n{metric_name}')
        plt.xlabel('Model')
        plt.ylabel(metric_name)  # Use actual metric name
        plt.xticks(rotation=45)
        plt.legend(title='', labels=['Baseline', 'Evolved'])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_dir) / f'comparison_{benchmark}_{evolve_type}.png'
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"Saved plot to {output_path}")
        
    except Exception as e:
        raise PlottingError(f"Error creating plot: {str(e)}")

def main() -> None:
    """
    Main function for generating comparison plots.
    
    Raises:
        PlottingError: If there's an error in the plotting process
    """
    try:
        # Create output directory
        os.makedirs('figures', exist_ok=True)
        
        # Load all results
        logger.info("Loading results...")
        data = load_results()
        
        # Get unique benchmarks and evolve_types
        benchmarks = data['benchmark'].unique()
        evolve_types = data['evolve_type'].unique()
        
        if len(benchmarks) == 0 or len(evolve_types) == 0:
            raise PlottingError("No benchmarks or evolution types found in data")
        
        # Generate plots for each combination
        logger.info("Generating plots...")
        for benchmark in benchmarks:
            for evolve_type in evolve_types:
                logger.info(f"Creating plot for {benchmark} - {evolve_type}")
                create_barplot(data, benchmark, evolve_type)
        
        logger.info("Done! Figures saved in 'figures' directory.")
        
    except Exception as e:
        logger.error(f"Error generating figures: {str(e)}")
        raise PlottingError(f"Error generating figures: {str(e)}")

if __name__ == "__main__":
    main() 