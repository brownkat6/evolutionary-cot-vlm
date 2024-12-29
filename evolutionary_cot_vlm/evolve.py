from typing import List, Dict, Any, Optional, Tuple
import json
import random
import numpy as np
import argparse
import os
from pathlib import Path
from datetime import datetime
from models import load_model
from evals import evaluate_model
from evolve_generations import (
    EVOLUTION_STRATEGIES,
    EvolutionParams,
    mutate_prefix
)

def get_timestamp() -> str:
    """Get current timestamp for printing."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Constants
N_GENERATIONS = 5
N_TRAIN_SAMPLES = 100

# Evolution parameters
EVOLUTION_PARAMS = EvolutionParams(
    population_size=100,
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
        List of prefix strings
        
    Raises:
        PrefixLoadError: If there's an error loading the prefixes
    """
    try:
        prefixes = []
        with open(file_path, 'r') as f:
            for line in f:
                prefix = json.loads(line)['prefix']
                prefixes.append(prefix)
        return prefixes
    except Exception as e:
        raise PrefixLoadError(f"Error loading prefixes from {file_path}: {str(e)}")

def fitness_function(
    model: Any,
    processor: Any,
    prefix: str,
    benchmark: str
) -> float:
    """
    Evaluate a prefix using the benchmark-specific metric on train set samples.
    
    Args:
        model: The model to evaluate
        processor: The model's processor/tokenizer
        prefix: Prefix to evaluate
        benchmark: Name of the benchmark
        
    Returns:
        Fitness score between 0 and 1
    """
    metrics = evaluate_model(
        model=model,
        processor=processor,
        benchmark=benchmark,
        split='train',
        num_samples=N_TRAIN_SAMPLES,
        prefix=prefix
    )
    
    if benchmark == 'mmmu':
        if 'short_form_accuracy' in metrics and 'long_form_rouge_l' in metrics:
            short_weight = metrics['short_form_count'] / metrics['total_samples']
            long_weight = metrics['long_form_count'] / metrics['total_samples']
            return (metrics['short_form_accuracy'] * short_weight + 
                   metrics['long_form_rouge_l'] * long_weight)
    return metrics['accuracy']

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
        print(f"   Seed file: {args.seed_file}\n")

        # Get evolution strategy
        evolution_strategy = EVOLUTION_STRATEGIES[args.evolve_type]
        print(f"🔄 Using {args.evolve_type} evolution strategy")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        print("📁 Created results directory")

        # Load model
        print(f"\n[{get_timestamp()}] 🤖 Loading {args.model} model...")
        model, processor = load_model(args.model)
        print("✅ Model loaded successfully")
        
        # Load seed prefixes
        print(f"\n[{get_timestamp()}] 📥 Loading seed prefixes from {args.seed_file}...")
        prefixes = load_seed_prefixes(args.seed_file)
        print(f"✅ Loaded {len(prefixes)} seed prefixes")
        
        # Get baseline validation score
        print(f"\n[{get_timestamp()}] 📊 Computing baseline validation score...")
        baseline_metrics = evaluate_model(
            model=model,
            processor=processor,
            benchmark=args.benchmark,
            split='validation',
            prefix="",
            data_dir=args.data_dir
        )
        baseline_validation_score = baseline_metrics['accuracy']
        print(f"📈 Baseline validation score: {baseline_validation_score:.4f}")
        
        # Evaluate seed prefixes
        print(f"\n[{get_timestamp()}] 🔍 Evaluating seed prefixes...")
        seed_scores = []
        for i, prefix in enumerate(prefixes, 1):
            print(f"   Evaluating seed prefix {i}/{len(prefixes)}", end='\r')
            score = fitness_function(model, processor, prefix, args.benchmark)
            seed_scores.append(score)
        print(f"\n📊 Mean seed score: {np.mean(seed_scores):.4f}")
        
        # Evolution loop
        current_prefixes = prefixes
        best_prefix: Optional[str] = None
        best_score = float('-inf')
        generation_best_scores: List[float] = []
        
        print(f"\n[{get_timestamp()}] 🧬 Starting evolution process...")
        validate_evolution_params(EVOLUTION_PARAMS)
        
        for generation in range(N_GENERATIONS):
            print(f"\n{'='*60}")
            print(f"🔄 Generation {generation + 1}/{N_GENERATIONS}")
            print(f"{'='*60}")
            
            scores = []
            print("\n📊 Evaluating current generation...")
            for i, prefix in enumerate(current_prefixes, 1):
                print(f"   Evaluating prefix {i}/{len(current_prefixes)}", end='\r')
                score = fitness_function(model, processor, prefix, args.benchmark)
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_prefix = prefix
                    print(f"\n🌟 New best score: {best_score:.4f}")
            
            generation_best = max(scores)
            generation_best_scores.append(float(generation_best))
            
            print(f"\n📈 Generation {generation + 1} Statistics:")
            print(f"   Best score:  {generation_best:.4f}")
            print(f"   Mean score:  {np.mean(scores):.4f}")
            print(f"   Std score:   {np.std(scores):.4f}")
            
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
        print(f"\n[{get_timestamp()}] 🎯 Evaluating best prefix on validation set...")
        if best_prefix is None:
            raise EvolutionError("No best prefix found during evolution")
            
        print(f"\n🏆 Best prefix found: {best_prefix}")
        val_metrics = evaluate_model(
            model=model,
            processor=processor,
            benchmark=args.benchmark,
            split='validation',
            prefix=best_prefix
        )
        
        final_validation_score = val_metrics['accuracy']
        improvement = final_validation_score - baseline_validation_score
        print(f"\n📊 Final Results:")
        print(f"   Validation score: {final_validation_score:.4f}")
        print(f"   Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
        
        # Save results
        results = {
            'benchmark': args.benchmark,
            'model': args.model,
            'evolve_type': args.evolve_type,
            'best_prefix': best_prefix,
            'best_train_score': best_score,
            'seed_mean_train_score': float(np.mean(seed_scores)),
            'validation_score': float(val_metrics['accuracy']),
            'baseline_validation_score': float(baseline_validation_score),
            'seed_scores': [float(s) for s in seed_scores],
            'generation_best_scores': generation_best_scores,
            'evolution_params': EVOLUTION_PARAMS.__dict__
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