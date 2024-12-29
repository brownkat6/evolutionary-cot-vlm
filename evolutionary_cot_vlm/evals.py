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
from utils.dataset_loading import get_chartqa_dataset

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
        # Convert split names for MMMU
        if benchmark == 'mmmu':
            mmmu_split = {
                'train': 'dev',
                'validation': 'validation',
                'test': 'test'
            }.get(split, split)
            
            dataset = load_dataset("MMMU/MMMU", 'Computer_Science', split=mmmu_split)
            
        elif benchmark == 'chartqa':
            # Try methods in order until one works
            for method in ["local", "download", "huggingface"]:
                try:
                    dataset = get_chartqa_dataset(split, method, data_dir)
                    logger.info(f"Successfully loaded ChartQA dataset using {method} method")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load dataset using {method} method: {e}")
            else:
                raise DatasetLoadError("Failed to load ChartQA dataset using any method")
            
        elif benchmark == 'vqav2':
            dataset = load_dataset("vqa_v2", split=split)
            
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
                        image = Image.open(BytesIO(requests.get(item['image']).content))
                        question = prefix + item['question']
                        ground_truth = str(item['answer'])
                        is_long_form = False
                        
                    elif benchmark == 'vqav2':
                        image = Image.open(BytesIO(requests.get(item['image']).content))
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
        metrics: Dict[str, float] = {
            'accuracy': correct / total if total > 0 else 0.0,
            'total_samples': float(total),
            'correct_samples': float(correct)
        }
        
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