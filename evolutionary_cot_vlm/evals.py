from typing import Any, Dict, List, Tuple, Union, Optional
import json
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from rouge_score import rouge_scorer
import logging
from pathlib import Path
import os
import shutil
from utils.dataset_loading import get_chartqa_dataset
import parlai.core.build_data as build_data
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.tasks.vqa_v2.agents import OeTeacher
from constants import CHARTQA_DIR, VQA_V2_DIR, MMMU_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass

class DatasetLoadError(Exception):
    """Custom exception for dataset loading errors."""
    pass

class MetricCalculationError(Exception):
    """Custom exception for metric calculation errors."""
    pass

def download_chartqa(output_dir: Path) -> None:
    """Download ChartQA dataset to specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Downloading ChartQA Dataset to {output_dir} ===")
    
    # Download from HuggingFace
    url = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    zip_path = output_dir / "chartqa.zip"
    
    print("1. Downloading zip file...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc="Downloading"
    ) as pbar:
        for data in response.iter_content(8192):
            size = f.write(data)
            pbar.update(size)
    
    print("2. Extracting files...")
    shutil.unpack_archive(zip_path, output_dir)
    
    print("3. Cleaning up...")
    zip_path.unlink()
    print(f"✓ ChartQA dataset successfully downloaded to {output_dir}")

def setup_vqa_v2(output_dir: Path) -> None:
    """Setup VQA-v2 dataset using ParlAI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Setting up VQA-v2 Dataset in {output_dir} ===")
    
    print("1. Initializing ParlAI...")
    opt = Opt({
        'task': 'vqa_v2',
        'datatype': 'train:ordered',
        'datapath': str(output_dir),
    })
    
    print("2. Downloading dataset (this may take a while)...")
    teacher = create_task_agent_from_taskname(opt)[0]
    
    print("3. Verifying download...")
    images_dir = output_dir / 'images'
    questions_dir = output_dir / 'questions'
    if images_dir.exists() and questions_dir.exists():
        print(f"✓ VQA-v2 dataset successfully set up in {output_dir}")
    else:
        print("! Warning: Some expected directories are missing")

def setup_mmmu(output_dir: Path) -> None:
    """Setup MMMU dataset using HuggingFace."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Setting up MMMU Dataset in {output_dir} ===")
    
    print("1. Downloading from HuggingFace...")
    dataset = load_dataset("MMMU/MMMU", 'Computer_Science', cache_dir=str(output_dir))
    
    print("2. Verifying splits...")
    for split in ['dev', 'validation', 'test']:
        if split in dataset:
            print(f"  ✓ Found {split} split with {len(dataset[split])} examples")
    print(f"✓ MMMU dataset successfully set up in {output_dir}")

def ensure_dataset(benchmark: str, data_dir: Optional[str] = None) -> Path:
    """
    Ensure dataset is downloaded and return its path.
    """
    print(f"\n=== Checking {benchmark.upper()} Dataset ===")
    
    if benchmark == 'chartqa':
        output_dir = Path(data_dir or CHARTQA_DIR)
        if not (output_dir / 'train' / 'train_augmented.json').exists():
            print("! ChartQA dataset not found. Starting download...")
            download_chartqa(output_dir)
        else:
            print("✓ ChartQA dataset already exists")
            
    elif benchmark == 'vqav2':
        output_dir = Path(data_dir or VQA_V2_DIR)
        if not (output_dir / 'images').exists():
            print("! VQA-v2 dataset not found. Starting setup...")
            setup_vqa_v2(output_dir)
        else:
            print("✓ VQA-v2 dataset already exists")
            
    elif benchmark == 'mmmu':
        output_dir = Path(data_dir or MMMU_DIR)
        if not output_dir.exists():
            print("! MMMU dataset not found. Starting setup...")
            setup_mmmu(output_dir)
        else:
            print("✓ MMMU dataset already exists")
            
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
        
    return output_dir

def evaluate_chartqa_answer(predicted: str, ground_truth: str) -> float:
    """
    ChartQA official evaluation metric.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer
    
    Returns:
        Score between 0 and 1
        
    Raises:
        MetricCalculationError: If there's an error calculating the metric
    """
    try:
        predicted = predicted.strip().lower()
        ground_truth = ground_truth.strip().lower()
        
        try:
            # Try numerical comparison first
            pred_num = float(predicted.replace(',', ''))
            true_num = float(ground_truth.replace(',', ''))
            # Allow 5% tolerance for numerical answers (official metric)
            return float(abs(pred_num - true_num) / max(abs(true_num), 1e-6) <= 0.05)
        except ValueError:
            # If not numerical, do exact string matching
            return float(predicted == ground_truth)
    except Exception as e:
        raise MetricCalculationError(f"Error calculating ChartQA metric: {str(e)}")

def evaluate_vqav2_answer(predicted: str, ground_truth: List[str]) -> float:
    """
    VQAv2 official evaluation metric.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: List of ground truth answers from different annotators
    
    Returns:
        Score between 0 and 1
        
    Raises:
        MetricCalculationError: If there's an error calculating the metric
    """
    try:
        predicted = predicted.strip().lower()
        answer_count: Dict[str, int] = {}
        
        for ans in ground_truth:
            ans = ans.strip().lower()
            answer_count[ans] = answer_count.get(ans, 0) + 1
        
        return min(answer_count.get(predicted, 0) / 3, 1)
    except Exception as e:
        raise MetricCalculationError(f"Error calculating VQAv2 metric: {str(e)}")

def evaluate_mmmu_answer(predicted: str, ground_truth: str, is_long_form: bool = False) -> float:
    """
    MMMU official evaluation metrics.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer
        is_long_form: Whether to use Rouge-L (True) or exact match (False)
    
    Returns:
        Score between 0 and 1
        
    Raises:
        MetricCalculationError: If there's an error calculating the metric
    """
    try:
        predicted = predicted.strip().lower()
        ground_truth = ground_truth.strip().lower()
        
        if is_long_form:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(ground_truth, predicted)
            return scores['rougeL'].fmeasure
        else:
            return float(predicted == ground_truth)
    except Exception as e:
        raise MetricCalculationError(f"Error calculating MMMU metric: {str(e)}")

def evaluate_model(
    model: Any,
    processor: Any,
    benchmark: str,
    split: str = "validation",
    num_samples: Optional[int] = None,
    prefix: str = "",
    data_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate model on benchmark dataset.
    """
    try:
        print(f"\n=== Starting Evaluation on {benchmark.upper()} ===")
        print(f"Split: {split}")
        print(f"Samples: {'all' if num_samples is None else num_samples}")
        
        # Ensure dataset is downloaded
        print("\n1. Preparing Dataset")
        dataset_dir = ensure_dataset(benchmark, data_dir)
        
        print("\n2. Loading Data")
        if benchmark == 'chartqa':
            dataset = get_chartqa_dataset(split, "local", str(dataset_dir))
            print(f"✓ Loaded {len(dataset)} examples from ChartQA")
            
        elif benchmark == 'vqav2':
            # Convert split names to ParlAI format
            split_map = {
                'train': 'train',
                'validation': 'valid',
                'test': 'test'
            }
            parlai_split = split_map.get(split)
            if not parlai_split:
                raise ValueError(f"Invalid split: {split}")
            
            # Initialize ParlAI options with our data directory
            opt = Opt({
                'task': 'vqa_v2',
                'datatype': f'{parlai_split}:ordered',
                'datapath': str(dataset_dir),
            })
            
            # Create teacher
            teacher = create_task_agent_from_taskname(opt)[0]
            
            # Load examples
            dataset = []
            num_examples = num_samples if num_samples else len(teacher)
            for _ in tqdm(range(num_examples), desc=f"Loading VQA-v2 {split}"):
                reply = teacher.act()
                if reply is None:
                    break
                dataset.append({
                    'question': reply['text'],
                    'answers': reply['labels'],
                    'image_path': reply['image']
                })
            
            print(f"✓ Loaded {len(dataset)} examples from VQA-v2")
            
        elif benchmark == 'mmmu':
            mmmu_split = {
                'train': 'dev',
                'validation': 'validation',
                'test': 'test'
            }.get(split, split)
            dataset = load_dataset(
                "MMMU/MMMU",
                'Computer_Science',
                split=mmmu_split,
                cache_dir=str(dataset_dir)
            )

            print(f"✓ Loaded {len(dataset)} examples from MMMU")

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Initialize metrics
        correct = 0
        total = 0
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        results: List[Dict[str, Any]] = []
        with torch.no_grad():
            for item in tqdm(dataset):
                try:
                    # Process input based on benchmark
                    if benchmark == 'chartqa':
                        image = Image.open(item['image_path'])
                        question = prefix + item['question']
                        ground_truth = str(item['answer'])
                        is_long_form = False
                        
                    elif benchmark == 'vqav2':
                        image = Image.open(item['image_path'])
                        question = prefix + item['question']
                        ground_truth = item['answers']
                        is_long_form = False
                        
                    elif benchmark == 'mmmu':
                        image = Image.open(BytesIO(requests.get(item['image_url']).content))
                        question = prefix + item['question']
                        ground_truth = str(item['answer'])
                        is_long_form = len(ground_truth.split()) > 15 or '\n' in ground_truth

                    # Generate prediction
                    inputs = processor(
                        images=image,
                        text=question,
                        return_tensors="pt"
                    ).to(device)

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50 if benchmark != 'mmmu' else 100,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    predicted = processor.decode(outputs[0], skip_special_tokens=True).lower()
                    
                    # Calculate score
                    if benchmark == 'chartqa':
                        score = evaluate_chartqa_answer(predicted, ground_truth)
                    elif benchmark == 'vqav2':
                        score = evaluate_vqav2_answer(predicted, ground_truth)
                    else:  # mmmu
                        score = evaluate_mmmu_answer(predicted, ground_truth, is_long_form)
                    
                    correct += score
                    total += 1
                    
                    results.append({
                        'question': question,
                        'ground_truth': ground_truth,
                        'predicted': predicted,
                        'score': float(score),
                        'is_long_form': is_long_form if benchmark == 'mmmu' else None
                    })

                except Exception as e:
                    logger.error(f"Error processing item: {str(e)}")
                    continue

        # Calculate metrics
        print("\n4. Computing Metrics")
        metrics: Dict[str, float] = {
            'accuracy': correct / total if total > 0 else 0.0,
            'total_samples': float(total),
            'correct_samples': float(correct)
        }
        
        print("\n=== Results ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Add benchmark-specific metrics
        if benchmark == 'mmmu':
            long_form_results = [r for r in results if r.get('is_long_form', False)]
            short_form_results = [r for r in results if not r.get('is_long_form', False)]
            
            if long_form_results:
                metrics['long_form_rouge_l'] = sum(r['score'] for r in long_form_results) / len(long_form_results)
                metrics['long_form_count'] = float(len(long_form_results))
            
            if short_form_results:
                metrics['short_form_accuracy'] = sum(r['score'] for r in short_form_results) / len(short_form_results)
                metrics['short_form_count'] = float(len(short_form_results))

        return metrics

    except Exception as e:
        print(f"\n! Error during evaluation: {str(e)}")
        raise EvaluationError(f"Error during evaluation: {str(e)}")

def save_results(benchmark: str, results: List[Dict], metrics: Dict[str, float]) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        benchmark: Name of the benchmark
        results: List of individual evaluation results
        metrics: Dictionary of computed metrics
        
    Raises:
        IOError: If there's an error saving the results
    """
    try:
        output = {
            'benchmark': benchmark,
            'metrics': metrics,
            'results': results
        }
        
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / f'eval_results_{benchmark}.json', 'w') as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        raise IOError(f"Error saving results: {str(e)}")

def load_results(benchmark: str) -> Dict:
    """
    Load previously saved evaluation results.
    
    Args:
        benchmark: Name of the benchmark
        
    Returns:
        Dictionary containing evaluation results
        
    Raises:
        IOError: If there's an error loading the results
    """
    try:
        with open(Path('results') / f'eval_results_{benchmark}.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Error loading results: {str(e)}") 