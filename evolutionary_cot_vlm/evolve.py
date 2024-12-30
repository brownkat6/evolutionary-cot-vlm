from typing import List, Dict, Any, Optional, Tuple, Union
import json
import random
import numpy as np
import argparse
import os
from pathlib import Path
from datetime import datetime
from models import load_model
from evals import evaluate_model, LMMWrapper
from evolve_generations import (
    EVOLUTION_STRATEGIES,
    EvolutionParams,
    mutate_prefix
)

def get_timestamp() -> str:
    """Get current timestamp for printing."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Constants
N_GENERATIONS = 2
N_TRAIN_SAMPLES = 10
N_SEED_PREFIXES = 20  # New constant for number of seed prefixes
N_VAL_SAMPLES = 30

# Evolution parameters
EVOLUTION_PARAMS = EvolutionParams(
    population_size=10,
    elite_size=10,
    mutation_rate=0.3,
    tournament_size=5,
    crossover_points=1,
    selection_pressure=2.0,
    temperature=0.1
)

def validate_evolution_params(params: EvolutionParams) -> None:
    """Validate evolution parameters."""
    if not 0 <= params.mutation_rate <= 1:
        raise ValueError(f"Mutation rate must be between 0 and 1, got {params.mutation_rate}")
    if params.population_size < params.elite_size:
        raise ValueError(f"Population size ({params.population_size}) must be greater than elite size ({params.elite_size})")
    if params.tournament_size > params.population_size:
        raise ValueError(f"Tournament size ({params.tournament_size}) must not exceed population size ({params.population_size})")
    if params.temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {params.temperature}")

class EvolutionError(Exception):
    """Custom exception for evolution process errors."""
    pass

class PrefixLoadError(Exception):
    """Custom exception for prefix loading errors."""
    pass

def load_seed_prefixes(file_path: str) -> List[str]:
    """
    Load prefixes from JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing prefixes
        
    Returns:
        List of prefix strings limited to N_SEED_PREFIXES
        
    Raises:
        PrefixLoadError: If there's an error loading the prefixes
    """
    try:
        prefixes = []
        with open(file_path, 'r') as f:
            for line in f:
                prefix = json.loads(line)['prefix']
                prefixes.append(prefix)
                if len(prefixes) >= N_SEED_PREFIXES:  # Limit to N_SEED_PREFIXES
                    break
        return prefixes
    except Exception as e:
        raise PrefixLoadError(f"Error loading prefixes from {file_path}: {str(e)}")

def fitness_function(
    model: Any,
    processor: Any,
    prefix: str,
    benchmark: str,
    split: str = "validation",
    num_samples: Optional[int] = None
) -> float:
    """Evaluate fitness of a prefix."""
    try:
        # Force test split for ChartQA
        eval_split = "test" if benchmark == "chartqa" else split
        
        # Evaluate model with prefix
        metrics = evaluate_model(
            model=model,
            processor=processor,
            benchmark=benchmark,
            split=eval_split,
            num_samples=num_samples,
            prefix=prefix
        )
        
        # Return appropriate metric based on benchmark
        if benchmark == 'mmmu':
            # Use weighted average of short-form and long-form scores
            return metrics['accuracy']
        elif benchmark == 'chartqa':
            return metrics['accuracy']
        elif benchmark == 'vqav2':
            return metrics['accuracy']
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
            
    except Exception as e:
        print(f"Error in fitness calculation: {str(e)}")
        return 0.0  # Return minimum score on error


def main() -> None:
    """Main function for running the evolution process."""
    try:
        print(f"\n[{get_timestamp()}] 🚀 Starting evolution process...\n")
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, required=True,
                          choices=['blip2', 'llava', 'minigpt4', 'otter', 'molmo'])
        parser.add_argument('--benchmark', type=str, required=True,
                          choices=['chartqa', 'vqav2', 'mmmu'])
        parser.add_argument('--seed_file', type=str, required=True,
                          help='Path to seed_prefixes.jsonl')
        parser.add_argument('--evolve_type', type=str, default='default',
                          choices=list(EVOLUTION_STRATEGIES.keys()),
                          help='Type of evolution strategy to use')
        parser.add_argument('--data_dir', type=str, default=None,
                           help='Path to dataset directory')
        args = parser.parse_args()
        
        print(f"📋 Configuration:")
        print(f"   Model: {args.model}")
        print(f"   Benchmark: {args.benchmark}")
        print(f"   Evolution type: {args.evolve_type}")
        print(f"   Seed file: {args.seed_file}")
        print(f"   Using {N_SEED_PREFIXES} seed prefixes\n")

        # Get evolution strategy
        evolution_strategy = EVOLUTION_STRATEGIES[args.evolve_type]
        print(f"🔄 Using {args.evolve_type} evolution strategy")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        print("📁 Created results directory")

        # Load model
        print(f"\n[{get_timestamp()}] 🤖 Loading {args.model} model...")
        model, processor = load_model(args.model)
        
        # Only wrap non-Llava models (where processor is not None)
        if processor is not None:
            model = LMMWrapper(model, processor)
        print("✅ Model loaded successfully")
        
        # Load seed prefixes
        print(f"\n[{get_timestamp()}] 📥 Loading seed prefixes from {args.seed_file}...")
        prefixes = load_seed_prefixes(args.seed_file)
        print(f"✅ Loaded {len(prefixes)} seed prefixes")
        
        # Get baseline validation score
        print(f"\n[{get_timestamp()}] 📊 Computing baseline validation score...")
        eval_split = "test" if args.benchmark == "chartqa" else "validation"
        baseline_metrics = evaluate_model(
            model=model,  # Use wrapped model
            processor=None,  # Processor not needed since it's in wrapper
            split=eval_split,
            benchmark=args.benchmark,
            prefix="",
            num_samples=N_VAL_SAMPLES,
        )
        
        # Define metric names for each benchmark and split
        metric_map = {
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
        
        # Get appropriate split and metric name
        eval_split = "test" if args.benchmark == "chartqa" else "validation"
        metric_name = metric_map[args.benchmark][eval_split]
        
        baseline_validation_score = baseline_metrics[metric_name]
        print(f"📈 Baseline {metric_name}: {baseline_validation_score:.4f}")
        
        # Evolution loop
        current_prefixes = prefixes
        best_prefix: Optional[str] = None
        best_score = float('-inf')
        generation_best_scores: List[float] = []
        
        print(f"\n[{get_timestamp()}] 🧬 Starting evolution process...")
        validate_evolution_params(EVOLUTION_PARAMS)
        
        seed_scores = []
        for generation in range(N_GENERATIONS):
            print(f"\n{'='*60}")
            print(f"🔄 Generation {generation + 1}/{N_GENERATIONS}")
            print(f"{'='*60}")
            
            scores = []
            print("\n📊 Evaluating current generation...")
            for i, prefix in enumerate(current_prefixes, 1):
                print(f"   Evaluating prefix {i}/{len(current_prefixes)}", end='\r')
                metrics = evaluate_model(
                    model=model,
                    processor=None,
                    prefix=prefix,
                    benchmark=args.benchmark,
                    split=eval_split,
                    num_samples=N_TRAIN_SAMPLES
                )
                scores.append(metrics[metric_name])
                if i == 0:
                    seed_scores.append(metrics[metric_name])
                
                if scores[-1] > best_score:
                    best_score = scores[-1]
                    best_prefix = prefix
                    print(f"\n🌟 New best {metric_name}: {best_score:.4f}")
            
            generation_best = max(scores)
            generation_best_scores.append(float(generation_best))
            
            print(f"\n📈 Generation {generation + 1} Statistics:")
            print(f"   Best {metric_name}:  {generation_best:.4f}")
            print(f"   Mean {metric_name}:  {np.mean(scores):.4f}")
            print(f"   Std {metric_name}:   {np.std(scores):.4f}")
            
            # Evolve new generation
            print("\n🧬 Evolving new generation...")
            current_prefixes = evolution_strategy(
                prefixes=current_prefixes,
                fitness_scores=scores,
                params=EVOLUTION_PARAMS,
                mutate_fn=mutate_prefix
            )
            print("✅ New generation created")
        
        # Evaluate best prefix on validation set
        print(f"\n[{get_timestamp()}] 🎯 Evaluating best prefix...")
        if best_prefix is None:
            raise EvolutionError("No best prefix found during evolution")
            
        print(f"\n🏆 Best prefix found: {best_prefix}")
        val_metrics = evaluate_model(
            model=model,
            processor=None,
            split=eval_split,
            benchmark=args.benchmark,
            prefix=best_prefix,
            num_samples=N_VAL_SAMPLES,
        )
        
        final_score = val_metrics[metric_name]
        improvement = final_score - baseline_validation_score
        print(f"\n📊 Final Results:")
        print(f"   {metric_name}: {final_score:.4f}")
        print(f"   Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
        
        # Save results with correct metric names
        results = {
            'benchmark': args.benchmark,
            'model': args.model,
            'evolve_type': args.evolve_type,
            'best_prefix': best_prefix,
            'best_train_score': best_score,
            'seed_mean_score': float(np.mean(seed_scores)),
            'validation_metrics': val_metrics,
            'baseline_validation_metrics': baseline_metrics,
            'seed_scores': [float(s) for s in seed_scores],
            'generation_best_scores': generation_best_scores,
            'evolution_params': EVOLUTION_PARAMS.__dict__,
            'metric_name': metric_name
        }
        
        results_path = Path('results') / f'evolution_results_{args.benchmark}_{args.model}_{args.evolve_type}.json'
        print(f"\n💾 Saving results to {results_path}")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n[{get_timestamp()}] ✨ Evolution process completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: Evolution process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 