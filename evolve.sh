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
#SBATCH --array=0-4 # TODO: for now just run 5 jobs instead of all of them

# Create necessary directories
mkdir -p slurm_logs
mkdir -p results
mkdir -p figures

# Define arrays of models, benchmarks, and evolution types
models=(blip2 llava minigpt4 otter molmo)
benchmarks=(chartqa vqav2 mmmu)
evolve_types=(default rank boltzmann)

# Calculate indices
model_idx=$((SLURM_ARRAY_TASK_ID / 9))  # 9 = 3 benchmarks × 3 evolve types
remainder=$((SLURM_ARRAY_TASK_ID % 9))
benchmark_idx=$((remainder / 3))
evolve_type_idx=$((remainder % 3))

model=${models[$model_idx]}
benchmark=${benchmarks[$benchmark_idx]}
evolve_type=${evolve_types[$evolve_type_idx]}

echo "Running evolution for model=$model benchmark=$benchmark evolve_type=$evolve_type"

# Activate conda environment (adjust as needed)
source ~/.bashrc
conda activate evolve_env

# Run the evolution script
python evolve.py \
    --model $model \
    --benchmark $benchmark \
    --seed_file seed_prefixes.jsonl \
    --evolve_type $evolve_type

echo "Finished evolution for model=$model benchmark=$benchmark evolve_type=$evolve_type" 