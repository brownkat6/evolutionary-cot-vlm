from typing import List, Dict, Any, Optional, Tuple, Union
import json
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
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
import zipfile

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

def download_chartqa(output_dir: Path = CHARTQA_DIR) -> None:
    """
    Download ChartQA dataset to specified directory.
    Default directory is CHARTQA_DIR from constants.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Downloading ChartQA Dataset to {output_dir} ===")
    
    # Download from HuggingFace
    url = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    zip_path = output_dir / "chartqa.zip"
    
    try:
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
        
        print(f"2. Extracting files... from {zip_path} to {output_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print("3. Cleaning up...")
        zip_path.unlink()
        print(f"✓ ChartQA dataset successfully downloaded to {output_dir}")
        
    except Exception as e:
        print(f"! Error during download: {str(e)}")
        if zip_path.exists():
            zip_path.unlink()
        raise

def setup_vqa_v2(output_dir: Path) -> None:
    """Setup VQA-v2 dataset using ParlAI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Setting up VQA-v2 Dataset ===")
    print(f"Target directory: {output_dir.absolute()}")
    
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
    print(f"Checking paths:")
    print(f"  - Images: {images_dir.absolute()}")
    print(f"  - Questions: {questions_dir.absolute()}")
    if images_dir.exists() and questions_dir.exists():
        print(f"✓ VQA-v2 dataset successfully set up in {output_dir}")
    else:
        print("! Warning: Some expected directories are missing")

def setup_mmmu(output_dir: Path) -> None:
    """Setup MMMU dataset by combining all subjects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Setting up MMMU Dataset in {output_dir} ===")
    
    # List of all MMMU subjects
    subjects = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    
    print("1. Downloading all MMMU subjects...")
    combined_datasets = {}
    
    for split in ['dev', 'validation', 'test']:
        print(f"\nLoading {split} split:")
        split_datasets = []
        
        for subject in subjects:
            print(f"  - Downloading {subject}...")
            try:
                dataset = load_dataset(
                    "MMMU/MMMU",
                    subject,
                    split=split,
                    cache_dir=str(output_dir)
                )
                split_datasets.append(dataset)
                print(f"    ✓ Found {len(dataset)} examples")
            except Exception as e:
                print(f"    ! Error loading {subject}: {str(e)}")
                continue
        
        # Combine all subjects for this split
        if split_datasets:
            combined_datasets[split] = concatenate_datasets(split_datasets)
            print(f"  ✓ Combined {split} split: {len(combined_datasets[split])} total examples")
    
    print("\n2. Verifying combined datasets:")
    for split, dataset in combined_datasets.items():
        print(f"  - {split}: {len(dataset)} examples")
    
    n_examples = sum(len(dataset) for dataset in combined_datasets.values())
    print(f"\n✓ MMMU dataset successfully set up in {output_dir} with {n_examples} examples")
    return combined_datasets

def ensure_dataset(benchmark: str, data_dir: Optional[str] = None) -> Path:
    """Ensure dataset is downloaded and return its path."""
    print(f"\n=== Checking {benchmark.upper()} Dataset ===")
    
    if benchmark == 'chartqa':
        output_dir = Path(data_dir or CHARTQA_DIR)
        print(f"ChartQA directory: {output_dir.absolute()}")
        dataset_dir = output_dir / 'ChartQA Dataset'
        train_file = dataset_dir / 'train' / 'train_augmented.json'
        print(f"Looking for dataset at: {train_file.absolute()}")
        
        if not train_file.exists():
            print("! ChartQA dataset not found. Starting download...")
            download_chartqa(output_dir)
            print(f"Verifying download at: {dataset_dir.absolute()}")
            if train_file.exists():
                print(f"✓ Found train file: {train_file.absolute()}")
            else:
                print(f"! Warning: Expected train file not found at {train_file.absolute()}")
        else:
            print("✓ ChartQA dataset already exists")
            print(f"  - Train file: {train_file.absolute()}")
            print(f"  - Dataset dir: {dataset_dir.absolute()}")
        return dataset_dir
        
    elif benchmark == 'vqav2':
        output_dir = Path(VQA_V2_DIR)
        print(f"VQA-v2 directory: {output_dir.absolute()}")
        if not (output_dir / 'images').exists():
            print("! VQA-v2 dataset not found. Starting setup...")
            setup_vqa_v2(output_dir)
        else:
            print("✓ VQA-v2 dataset already exists")
            print(f"  - Images: {(output_dir / 'images').absolute()}")
            print(f"  - Questions: {(output_dir / 'questions').absolute()}")
            
    elif benchmark == 'mmmu':
        output_dir = Path(MMMU_DIR)
        print(f"MMMU directory: {output_dir.absolute()}")
        if not output_dir.exists():
            print("! MMMU dataset not found. Starting setup...")
            setup_mmmu(output_dir)
        else:
            print("✓ MMMU dataset already exists")
            print(f"  - Cache directory: {output_dir.absolute()}")
            
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
    data_dir: Optional[str] = None,
    dataset: Optional[Dataset] = None,
    return_dataset: bool = False
) -> Union[Dict[str, float], Dataset]:
    """
    Evaluate model on benchmark dataset.
    
    Args:
        model: The model to evaluate
        processor: The model's processor/tokenizer
        benchmark: Name of the benchmark
        split: Split to evaluate
        num_samples: Number of samples to evaluate
        prefix: Prefix to evaluate
        data_dir: Data directory
        dataset: Optional pre-loaded dataset to use
        return_dataset: If True, return the loaded dataset instead of evaluating
        
    Returns:
        Either evaluation metrics or the loaded dataset
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
            dataset_path = str(dataset_dir)
            print(f"Loading ChartQA dataset from: {dataset_path}")
            dataset = get_chartqa_dataset(split, "local", dataset_path)
            print(f"✓ Loaded {len(dataset)} examples from ChartQA")
            print(f"  - Split: {split}")
            print(f"  - Path: {dataset_path}")
            
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
            
            # Load and combine all subjects
            print(f"Loading MMMU dataset from: {dataset_dir}")
            combined_dataset = setup_mmmu(dataset_dir)
            dataset = combined_dataset[mmmu_split]
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
        
        # Initialize counters
        processed_count = 0
        skipped_count = 0

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
                        if item.get('image_path') is None:
                            skipped_count += 1
                            continue
                        processed_count += 1
                            
                        image = Image.open(item['image_path'])
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

        # At the end of processing
        print(f"\nMMU Stats: {processed_count} items processed, {skipped_count} items skipped (no images)")

        if return_dataset:
            return dataset
        else:
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