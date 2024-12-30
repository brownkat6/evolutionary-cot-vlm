# Evolutionary Prompt Engineering for Vision-Language Models
## Overview
This repository implements an evolutionary computation framework for optimizing prompt prefixes in vision-language models (VLMs). The framework employs three theoretically-grounded selection strategies to evolve effective prompts across multiple benchmarks and model architectures.
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

```
mamba create --prefix /path/to/env/evolve_env python=3.9
mamba activate /path/to/env/evolve_env
```
Install dependencies
```
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c conda-forge numpy pandas matplotlib seaborn transformers accelerate datasets pillow requests tqdm scikit-learn tensorboard jupyter bitsandbytes
pip install rouge-score
```

Generate seed prefixes
```
python generate_prefixes.py
```
Launch evolution jobs
```
sbatch evolve.sh
```
Generate analysis figures
```
python gen_figures1.py
```

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

## Key Files

- `evolve_generations.py`: Implements three evolution strategies (Tournament, Rank-Based, Boltzmann)
- `evolve.py`: Main script for running evolution experiments
- `evals.py`: Contains evaluation metrics and dataset loading utilities
- `models.py`: Model loading and inference implementations
- `gen_figures1.py`: Generates comparison plots and visualizations
- `generate_prefixes.py`: Creates initial population of prompt prefixes
- `evolve.sh`: SLURM script for distributed execution

## Generated Files

- `results/evolution_results_*.json`: Evolution results for each model-benchmark combination
- `figures/comparison_*.png`: Performance comparison plots

## Citation
If you use this code in your research, please cite:

```
@misc{evolutionary_prompt_engineering,
title="Evolutionary Prompt Engineering for Vision-Language Models",
author="Brown, Katrina and Rho, John and Bhalthuwar, Sid",
year="2024",
publisher={GitHub},
journal={GitHub repository},
howpublished={\url{https://github.com/brownkat6/evolutionary-cot-vlm}}
}
```

## Acknowledgments
This research was conducted at Harvard University no external funding sources.


Data download
```
wget -O chartqa.zip "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip" && \
unzip chartqa.zip -d CHARTQA_DIR && \
rm chartqa.zip
```


# TODO: Debug 0 samples i train/val set for MMMU
✓ MMMU dataset successfully set up in /n/netscratch/dwork_lab/Lab/katrina/data/mmmu with 11550 examples
✅ Loaded validation dataset with 0 samples


# TODO: debug Running evolution for model=llava benchmark=chartqa evolve_type=default
INFO:evals:LLaVa processor configured with:
INFO:evals:  - Image size: {'height': 336, 'width': 336}
INFO:evals:  - Patch size: 14
INFO:evals:  - Feature strategy: full
^M  0%|          | 0/30 [00:00<?, ?it/s]Expanding inputs for image tokens in LLaVa should be done in processing. Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. Using processors without these attributes in the config is deprecated and will throw an error in v4.47.
ERROR:evals:Error processing item: The input provided to the model are wrong. The number of image tokens is 0 while the number of image given to the model is 5. This prevents correct indexing and breaks batch generation.
^M  3%|▎         | 1/30 [00:01<00:40,  1.39s/it]ERROR:evals:Error processing item: The input provided to the model are wrong. The number of image tokens is 0 while the number of image given to the model is 5. This prevents correct indexing and breaks batch generation.
ERROR:evals:Error processing item: The input provided to the model are wrong. The number of image tokens is 0 while the number of image given to the model is 5. This prevents correct indexing and breaks batch generation.
ERROR:evals:Error processing item: The input provided to the model are wrong. The number of image tokens is 0 while the number of image given to the model is 5. This prevents correct indexing and breaks batch generation.
ERROR:evals:Error processing item: The input provided to the model are wrong. The number of image tokens is 0 while the number of image given to the model is 5. This prevents correct indexing and breaks batch generation.

VQA BLIP2
ERROR:evals:Error creating VQA task agent: [Errno 122] Disk quota exceeded: '/n/holylabs/LABS/dwork_lab/Lab/katrinabrown/home/conda/envs/evolve_env/lib/python3.9/site-packages/data/COCO-IMG-2014/train2014/COCO_train2014_000000252702.jpg'

Running evolution for model=llava benchmark=chartqa evolve_type=default
