"""
This module implements three distinct evolutionary computation strategies for optimizing prompt prefixes:

1. Tournament Selection (default):
   A selection mechanism that randomly samples k individuals from the population and selects
   the fittest among them. This process creates a selection pressure that can be tuned via
   the tournament size parameter. Larger tournaments lead to higher selection pressure and
   faster convergence, while smaller tournaments maintain higher diversity.
   Reference: Miller, B. L., & Goldberg, D. E. (1995). Genetic algorithms, tournament
   selection, and the effects of noise. Complex Systems, 9(3), 193-212.

2. Rank-Based Selection:
   This strategy assigns selection probabilities based on the rank of individuals rather
   than their absolute fitness values. This approach helps prevent premature convergence
   and is particularly useful when fitness scores have non-uniform scaling or when the
   population contains outliers. The selection pressure parameter controls the relative
   advantage of higher-ranked individuals.
   Reference: Baker, J. E. (1985). Adaptive selection methods for genetic algorithms.
   In Proceedings of an International Conference on Genetic Algorithms and Their Applications.

3. Boltzmann Selection:
   Inspired by simulated annealing, this method uses an exponential scaling of fitness
   values controlled by a temperature parameter. Higher temperatures lead to more uniform
   selection probabilities (exploration), while lower temperatures increase the selection
   pressure towards fitter individuals (exploitation). This creates a natural annealing
   schedule for the selection process.
   Reference: Goldberg, D. E. (1990). A note on Boltzmann tournament selection for
   genetic algorithms and population-oriented simulated annealing. Complex Systems, 4(4),
   445-460.

All strategies maintain elitism by preserving the best solutions across generations and
employ multi-point crossover for recombination. The mutation operator applies random
text transformations with a configurable probability to maintain genetic diversity.

The effectiveness of each strategy depends on the fitness landscape characteristics:
- Tournament: Robust and efficient for most applications
- Rank-Based: Better for noisy or non-uniformly scaled fitness functions
- Boltzmann: Useful when gradual convergence is desired or when the fitness landscape
  has many local optima

Parameters for each strategy can be tuned via the EvolutionParams dataclass.
"""

from typing import List, Tuple, Callable
import random
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvolutionParams:
    """Parameters for evolution strategies."""
    population_size: int = 100
    elite_size: int = 10
    mutation_rate: float = 0.3
    tournament_size: int = 5
    crossover_points: int = 1
    selection_pressure: float = 2.0  # For rank-based selection
    temperature: float = 0.1  # For Boltzmann selection

def tournament_selection(
    prefixes: List[str],
    fitness_scores: List[float],
    tournament_size: int
) -> str:
    """
    Select parent using tournament selection.
    
    Args:
        prefixes: List of candidate prefixes
        fitness_scores: Corresponding fitness scores
        tournament_size: Number of candidates in each tournament
        
    Returns:
        Selected parent prefix
    """
    tournament_idx = random.sample(range(len(prefixes)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_idx]
    winner_idx = tournament_idx[np.argmax(tournament_fitness)]
    return prefixes[winner_idx]

def rank_based_selection(
    prefixes: List[str],
    fitness_scores: List[float],
    selection_pressure: float
) -> str:
    """
    Select parent using rank-based selection.
    
    Args:
        prefixes: List of candidate prefixes
        fitness_scores: Corresponding fitness scores
        selection_pressure: Controls selection intensity (1.0 to 2.0)
        
    Returns:
        Selected parent prefix
        
    Raises:
        ValueError: If selection pressure is invalid
    """
    if not 1.0 <= selection_pressure <= 2.0:
        raise ValueError(f"Selection pressure must be between 1.0 and 2.0, got {selection_pressure}")
    if not fitness_scores:
        raise ValueError("Empty fitness scores")
    
    # Sort by fitness
    sorted_pairs = sorted(enumerate(fitness_scores), key=lambda x: x[1])
    ranks = np.arange(1, len(prefixes) + 1)
    
    # Calculate rank-based probabilities
    probabilities = (2 - selection_pressure + (2 * (selection_pressure - 1) * 
                    (ranks - 1) / (len(prefixes) - 1))) / len(prefixes)
    
    # Validate probabilities
    if not np.all(probabilities >= 0):
        raise ValueError("Negative probabilities in rank-based selection")
    if abs(sum(probabilities) - 1.0) > 1e-10:
        raise ValueError("Probabilities do not sum to 1 in rank-based selection")
    
    # Select based on ranks
    selected_idx = random.choices(
        [pair[0] for pair in sorted_pairs],
        weights=probabilities,
        k=1
    )[0]
    
    return prefixes[selected_idx]

def boltzmann_selection(
    prefixes: List[str],
    fitness_scores: List[float],
    temperature: float
) -> str:
    """
    Select parent using Boltzmann selection.
    
    Args:
        prefixes: List of candidate prefixes
        fitness_scores: Corresponding fitness scores
        temperature: Temperature parameter (controls selection pressure)
        
    Returns:
        Selected parent prefix
        
    Raises:
        ValueError: If temperature is too close to zero
        ValueError: If fitness scores are invalid
    """
    if temperature < 1e-10:
        raise ValueError("Temperature too close to zero")
    if not fitness_scores:
        raise ValueError("Empty fitness scores")
    
    # Calculate Boltzmann probabilities
    scaled_fitness = np.array(fitness_scores) / temperature
    max_scaled = max(scaled_fitness)  # Store for numerical stability
    exp_fitness = np.exp(scaled_fitness - max_scaled)  # Subtract max for numerical stability
    sum_exp = sum(exp_fitness)
    
    if sum_exp < 1e-10:
        raise ValueError("Numerical underflow in Boltzmann selection")
        
    probabilities = exp_fitness / sum_exp
    
    # Validate probabilities
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Invalid probabilities in Boltzmann selection")
    
    # Select based on Boltzmann probabilities
    selected_idx = random.choices(range(len(prefixes)), weights=probabilities, k=1)[0]
    return prefixes[selected_idx]

def multi_point_crossover(parent1: str, parent2: str, n_points: int) -> str:
    """
    Perform multi-point crossover between two parents.
    
    Args:
        parent1: First parent string
        parent2: Second parent string
        n_points: Number of crossover points
        
    Returns:
        Child string
        
    Raises:
        ValueError: If inputs are invalid
    """
    if n_points < 1:
        raise ValueError(f"Number of crossover points must be positive, got {n_points}")
    
    words1 = parent1.split()
    words2 = parent2.split()
    
    if not words1 or not words2:
        raise ValueError("Empty parent strings")
    
    # Get crossover points
    max_points = min(len(words1), len(words2)) - 1
    if max_points < 1:
        return parent1  # Return first parent if crossover not possible
        
    n_points = min(n_points, max_points)
    points = sorted(random.sample(range(1, max_points + 1), n_points))
    
    # Perform crossover
    current_parent = words1
    result = []
    last_point = 0
    
    for point in points:
        result.extend(current_parent[last_point:point])
        current_parent = words2 if current_parent is words1 else words1
        last_point = point
    
    result.extend(current_parent[last_point:])
    return " ".join(result)

def default_evolution(
    prefixes: List[str],
    fitness_scores: List[float],
    params: EvolutionParams,
    mutate_fn: Callable[[str, EvolutionParams], str]
) -> List[str]:
    """
    Default evolution strategy using tournament selection.
    
    Args:
        prefixes: Current generation of prefixes
        fitness_scores: Corresponding fitness scores
        params: Evolution parameters
        mutate_fn: Function to mutate prefixes
        
    Returns:
        New generation of prefixes
    """
    # Sort by fitness and keep elite
    sorted_pairs = sorted(zip(prefixes, fitness_scores), 
                         key=lambda x: x[1], 
                         reverse=True)
    sorted_prefixes = [p for p, _ in sorted_pairs]
    
    new_generation = sorted_prefixes[:params.elite_size]
    
    # Generate rest of population using tournament selection
    while len(new_generation) < params.population_size:
        parent1 = tournament_selection(prefixes, fitness_scores, params.tournament_size)
        parent2 = tournament_selection(prefixes, fitness_scores, params.tournament_size)
        
        child = multi_point_crossover(parent1, parent2, params.crossover_points)
        child = mutate_fn(child, params)
        new_generation.append(child)
    
    return new_generation

def rank_based_evolution(
    prefixes: List[str],
    fitness_scores: List[float],
    params: EvolutionParams,
    mutate_fn: Callable[[str, EvolutionParams], str]
) -> List[str]:
    """
    Evolution strategy using rank-based selection.
    
    Args:
        prefixes: Current generation of prefixes
        fitness_scores: Corresponding fitness scores
        params: Evolution parameters
        mutate_fn: Function to mutate prefixes
        
    Returns:
        New generation of prefixes
    """
    # Sort by fitness and keep elite
    sorted_pairs = sorted(zip(prefixes, fitness_scores), 
                         key=lambda x: x[1], 
                         reverse=True)
    sorted_prefixes = [p for p, _ in sorted_pairs]
    
    new_generation = sorted_prefixes[:params.elite_size]
    
    # Generate rest of population using rank-based selection
    while len(new_generation) < params.population_size:
        parent1 = rank_based_selection(prefixes, fitness_scores, params.selection_pressure)
        parent2 = rank_based_selection(prefixes, fitness_scores, params.selection_pressure)
        
        child = multi_point_crossover(parent1, parent2, params.crossover_points)
        child = mutate_fn(child, params)
        new_generation.append(child)
    
    return new_generation

def boltzmann_evolution(
    prefixes: List[str],
    fitness_scores: List[float],
    params: EvolutionParams,
    mutate_fn: Callable[[str, EvolutionParams], str]
) -> List[str]:
    """
    Evolution strategy using Boltzmann selection.
    
    Args:
        prefixes: Current generation of prefixes
        fitness_scores: Corresponding fitness scores
        params: Evolution parameters
        mutate_fn: Function to mutate prefixes
        
    Returns:
        New generation of prefixes
    """
    # Sort by fitness and keep elite
    sorted_pairs = sorted(zip(prefixes, fitness_scores), 
                         key=lambda x: x[1], 
                         reverse=True)
    sorted_prefixes = [p for p, _ in sorted_pairs]
    
    new_generation = sorted_prefixes[:params.elite_size]
    
    # Generate rest of population using Boltzmann selection
    while len(new_generation) < params.population_size:
        parent1 = boltzmann_selection(prefixes, fitness_scores, params.temperature)
        parent2 = boltzmann_selection(prefixes, fitness_scores, params.temperature)
        
        child = multi_point_crossover(parent1, parent2, params.crossover_points)
        child = mutate_fn(child, params)
        new_generation.append(child)
    
    return new_generation

def mutate_prefix(prefix: str, params: EvolutionParams) -> str:
    """
    Apply random mutations to a prefix. Each possible mutation is applied independently
    with probability mutation_rate.
    
    Args:
        prefix: Original prefix string
        params: Evolution parameters including mutation rate
        
    Returns:
        Mutated prefix string
        
    Raises:
        ValueError: If no mutations are defined
    """
    mutations = [
        lambda x: x.replace("step by step", "systematically"),
        lambda x: x.replace("systematically", "methodically"),
        lambda x: x.replace("analyze", "examine"),
        lambda x: x.replace("Let's", "Let me"),
        lambda x: x.replace("carefully", "thoroughly"),
        lambda x: x.replace("logical", "structured"),
        lambda x: x.replace("breakdown", "analysis"),
    ]
    
    if not mutations:
        raise ValueError("No mutations defined")
    
    mutated = prefix
    for mutation in mutations:
        if random.random() < params.mutation_rate:
            mutated = mutation(mutated)
    
    return mutated

# Dictionary mapping evolution types to their functions
EVOLUTION_STRATEGIES = {
    'default': default_evolution,
    'rank': rank_based_evolution,
    'boltzmann': boltzmann_evolution
} 