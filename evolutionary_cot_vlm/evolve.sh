#!/bin/bash
#SBATCH --job-name=evolve_prefixes
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=0-12:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/evolve_%A_%a.out
#SBATCH --error=slurm_logs/evolve_%A_%a.err
## SBATCH --array=0-44  # 5 models × 3 benchmarks × 3 evolution types = 45 total jobs
#SBATCH --array=0-14 # TODO: for now just run subset of jobs instead of all of them

# Set cache directory to lab's netscratch location
export EVAL_CACHE_DIR="/n/netscratch/dwork_lab/Lab/katrina/data/vlm_evolution_cache"

# Create necessary directories
mkdir -p slurm_logs
mkdir -p results
mkdir -p figures
mkdir -p "$EVAL_CACHE_DIR"

# Enable dataset caching and image preloading
export USE_DATASET_CACHE="1" #"1"
export PRELOAD_IMAGES="1" #"1"

# Define arrays of models, benchmarks, and evolution types
models=(blip2 molmo llava minigpt4 claude)
benchmarks=(chartqa vqav2 mmmu)
evolve_types=(default rank boltzmann)
#evolve_types=(default)

# Calculate indices
evolve_type_idx=$((SLURM_ARRAY_TASK_ID / 9))  # 9 = 3 benchmarks × 3 evolve types
remainder=$((SLURM_ARRAY_TASK_ID % 9))
benchmark_idx=$((remainder / 3))
model_idx=$((remainder % 3))

model=${models[$model_idx]}
benchmark=${benchmarks[$benchmark_idx]}
evolve_type=${evolve_types[$evolve_type_idx]}

echo "Running evolution for model=$model benchmark=$benchmark evolve_type=$evolve_type"

# Activate conda environment (adjust as needed)
source ~/.bashrc
conda activate evolve_env

# Run the evolution script
/n/holylabs/LABS/dwork_lab/Lab/katrinabrown/home/conda/envs/evolve_env/bin/python evolve.py \
    --model $model \
    --benchmark $benchmark \
    --seed_file seed_prefixes.jsonl \
    --evolve_type $evolve_type

echo "Finished evolution for model=$model benchmark=$benchmark evolve_type=$evolve_type" 