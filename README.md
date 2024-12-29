# Evolutionary Prompt Engineering for Vision-Language Models
## Overview
his repository implements an evolutionary computation framework for optimizing prompt prefixes in vision-language models (VLMs). The framework employs three theoretically-grounded selection strategies to evolve effective prompts across multiple benchmarks and model architectures.
## Selection Strategies
### 1. Tournament Selection (Default)
mplements a standard tournament selection mechanism where k individuals are randomly sampled from the population, and the fittest individual is selected as a parent. This approach provides adjustable selection pressure through tournament size modification.
### 2. Rank-Based Selection
mploys selection probabilities based on the rank of individuals rather than their absolute fitness values. This method helps prevent premature convergence and is particularly effective when fitness scores have non-uniform scaling.
### 3. Boltzmann Selection
tilizes an exponential scaling of fitness values controlled by a temperature parameter, inspired by simulated annealing principles. This approach enables dynamic adjustment of selection pressure throughout the evolutionary process.
## Supported Models
 BLIP-2
 LLaVA
 MiniGPT-4
 OTTER
 MoLMo
## Benchmarks
 ChartQA
 VQAv2
 MMMU (Multi-Modal Multi-Task Understanding)
## Installation
### Environment Setup
Create conda environment
mamba create --prefix /path/to/env/evolve_env python=3.9
mamba activate /path/to/env/evolve_env
Install dependencies
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c conda-forge numpy pandas matplotlib seaborn transformers accelerate datasets pillow requests tqdm scikit-learn tensorboard jupyter bitsandbytes
pip install rouge-score
bash
Generate seed prefixes
python generate_prefixes.py
Launch evolution jobs
sbatch evolve.sh
Generate analysis figures
python gen_figures1.py

## Implementation Details

### Evolution Parameters
- Population Size: 100
- Elite Size: 10
- Mutation Rate: 0.3
- Tournament Size: 5
- Crossover Points: 1
- Selection Pressure: 2.0 (Rank-based)
- Temperature: 0.1 (Boltzmann)

### Evaluation Metrics
- ChartQA: Accuracy
- VQAv2: Accuracy
- MMMU: Weighted combination of short-form accuracy and long-form ROUGE-L scores

## File Structure
## Citation
If you use this code in your research, please cite:
bibtex
@misc{evolutionary_prompt_engineering,
title={Evolutionary Prompt Engineering for Vision-Language Models},
author={[Authors]},
year={2024},
publisher={GitHub},
journal={GitHub repository},
howpublished={\url{https://github.com/brownkat6/evolutionary-cot-vlm}}
}

## License
[License Type]

## Acknowledgments
This research was conducted at Harvard University no external funding sources.
EOL



