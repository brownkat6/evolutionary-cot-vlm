#!/bin/bash
#SBATCH --job-name=evolve_single
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=0-12:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/evolve_single_%j.out
#SBATCH --error=slurm_logs/evolve_single_%j.err

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model> <benchmark> <evolve_type>"
    echo "  model: one of [blip2, molmo, llava, minigpt4, claude]"
    echo "  benchmark: one of [chartqa, vqav2, mmmu]"
    echo "  evolve_type: one of [default, rank, boltzmann]"
    exit 1
fi

# Get command line arguments
model=$1
benchmark=$2
evolve_type=$3

# Validate inputs
valid_models=(blip2 molmo llava minigpt4 claude)
valid_benchmarks=(chartqa vqav2 mmmu)
valid_evolve_types=(default rank boltzmann)

# Check model
if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
    echo "Error: Invalid model. Must be one of: ${valid_models[*]}"
    exit 1
fi

# Check benchmark
if [[ ! " ${valid_benchmarks[@]} " =~ " ${benchmark} " ]]; then
    echo "Error: Invalid benchmark. Must be one of: ${valid_benchmarks[*]}"
    exit 1
fi

# Check evolve_type
if [[ ! " ${valid_evolve_types[@]} " =~ " ${evolve_type} " ]]; then
    echo "Error: Invalid evolve_type. Must be one of: ${valid_evolve_types[*]}"
    exit 1
fi

# Set cache directory to lab's netscratch location
export EVAL_CACHE_DIR="/n/netscratch/dwork_lab/Lab/katrina/data/vlm_evolution_cache"

# Create necessary directories
mkdir -p slurm_logs
mkdir -p results
mkdir -p figures
mkdir -p "$EVAL_CACHE_DIR"

# Enable dataset caching and image preloading
export USE_DATASET_CACHE="1"
export PRELOAD_IMAGES="1"

echo "Running evolution for model=$model benchmark=$benchmark evolve_type=$evolve_type"

# Initialize mamba
source ~/.bashrc
conda activate evolve_env

# echo the python command you are about to run
echo "Running command: /n/holylabs/LABS/dwork_lab/Lab/katrinabrown/home/conda/envs/evolve_env/bin/python evolve.py --model $model --benchmark $benchmark --seed_file seed_prefixes.jsonl --evolve_type $evolve_type"

# Run the evolution script
/n/holylabs/LABS/dwork_lab/Lab/katrinabrown/home/conda/envs/evolve_env/bin/python evolve.py \
    --model $model \
    --benchmark $benchmark \
    --seed_file seed_prefixes.jsonl \
    --evolve_type $evolve_type

echo "Finished evolution for model=$model benchmark=$benchmark evolve_type=$evolve_type" 